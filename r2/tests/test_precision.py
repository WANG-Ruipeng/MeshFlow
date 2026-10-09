"""Small CPU precision contracts: no checkpoint, real model, CUDA or updates."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock
import unittest

from meshflow_control.precision import (
    precision_execution, current_precision, readout_execution_options, DtypeAudit)
from meshflow_control.models.readout import (
    CornerReadout, apply_readout, gather_known_corners, corner_indices)
from meshflow_control.models import native
import torch
from torch import nn


class ContractOnlyModel(nn.Module):
    """Runtime contract shape only; has no NativeModel forward."""
    def __init__(self):
        super().__init__()
        self.role_embedding = nn.Parameter(torch.zeros(2, 4), requires_grad=False)
        self.linear = nn.Linear(4, 4).requires_grad_(False)
        self.backbone = SimpleNamespace(layers=[SimpleNamespace(gradient_checkpointing=False)])
        self.experiment_trainable = False
        self.eval()


class CountResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, query, memory, *, observer=None):
        self.calls += 1
        return query.float() * .001 + .003


def fixture(dtype):
    hidden = torch.arange(2 * 12 * 4, dtype=torch.float32).reshape(2, 12, 4) / 11 + 1
    hidden = hidden.to(dtype)
    known = torch.tensor([[True, False, False, False], [False, True, True, False]])
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    return hidden, known, valid


def original_addition(hidden, known, valid, module, memory=None):
    memory = gather_known_corners(hidden, known, valid) if memory is None else memory
    rows = []
    for b in range(hidden.shape[0]):
        indices = corner_indices(valid[b] & ~known[b])
        query = hidden[b].index_select(0, indices)
        residual = module(query, memory[b])
        updated = query + residual.to(dtype=hidden.dtype)
        rows.append(hidden[b].index_copy(0, indices, updated))
    return torch.stack(rows)


class PrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertIsNone(current_precision())
        self.assertEqual(readout_execution_options(), {})

    def test_default_addition_bitwise_matches_original_and_preserves_masks(self):
        for dtype in (torch.float32, torch.bfloat16):
            h, k, v = fixture(dtype)
            before = h.clone()
            actual = apply_readout(h, k, v, CountResidual())
            expected = original_addition(h, k, v, CountResidual())
            self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
            self.assertTrue(torch.equal(h.view(torch.uint8), before.view(torch.uint8)))
            fixed = (k | ~v).repeat_interleave(3, dim=1)
            self.assertTrue(torch.equal(actual[fixed].view(torch.uint8), h[fixed].view(torch.uint8)))

    def test_off_skips_only_extra_computation_zero_keeps_real_r_and_actual_zeros(self):
        h, k, v = fixture(torch.bfloat16)
        module = CountResidual()
        off = apply_readout(h, k, v, module, addition_mode="off")
        self.assertIs(off, h)
        self.assertEqual(module.calls, 0)
        captures = []
        zero = apply_readout(h, k, v, module, addition_mode="zero", capture=captures.append)
        self.assertEqual(module.calls, 2)
        self.assertTrue(torch.equal(off.view(torch.uint8), zero.view(torch.uint8)))
        self.assertEqual(len(captures), 2)
        for i, record in enumerate(captures):
            idx = corner_indices(v[i] & ~k[i])
            self.assertTrue(torch.equal(record["free_corner_indices"], idx))
            self.assertTrue(torch.equal(record["before"], h[i, idx]))
            self.assertGreater(torch.count_nonzero(record["residual"]).item(), 0)
            self.assertGreater(torch.count_nonzero(record["cast"]).item(), 0)
            self.assertEqual(torch.count_nonzero(record["applied"]).item(), 0)
            self.assertTrue(torch.equal(record["after"], record["before"] + record["applied"]))

    def test_capture_is_actual_free_only_addition_and_clean_memory_not_mutated(self):
        h, k, v = fixture(torch.float32)
        memory = tuple(x + .7 for x in gather_known_corners(h, k, v))
        saved_memory = [x.clone() for x in memory]
        records = []
        result = apply_readout(h, k, v, CountResidual(), memory=memory,
                               capture=records.append, layer=6, source_mode="clean")
        for i, record in enumerate(records):
            idx = record["free_corner_indices"]
            self.assertEqual(record["layer"], 6)
            self.assertEqual(record["source_mode"], "clean")
            self.assertTrue(torch.equal(result[i, idx], record["after"]))
            self.assertTrue(torch.equal(record["cast"], record["applied"]))
            self.assertTrue(torch.equal(record["after"], record["before"] + record["cast"]))
            self.assertTrue(torch.equal(memory[i], saved_memory[i]))

    def test_scalar_cast_survives_but_bf16_addition_can_round_away(self):
        h = torch.tensor([1.], dtype=torch.bfloat16)
        r = torch.tensor([.001], dtype=torch.float32)
        c = r.to(h.dtype)
        self.assertGreater(c.item(), 0)
        self.assertEqual((h + c).item(), h.item())
        self.assertNotEqual((h.float() + r).item(), h.float().item())

    def test_routes_and_scope_restore_on_exception_without_parameter_change(self):
        model = ContractOnlyModel()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        seen = []

        @contextmanager
        def execution(m, *, coordinates=None, autocast=True):
            seen.append((m, coordinates, autocast, current_precision()))
            yield {}

        with mock.patch.object(native, "native_execution", execution):
            for precision in ("LEGACY_BF16", "FP32_REFERENCE"):
                for switch in ("on", "off", "zero"):
                    with precision_execution(model, precision, switch) as facts:
                        self.assertEqual(current_precision(), precision)
                        self.assertEqual(readout_execution_options()["addition_mode"], switch)
                        self.assertEqual(facts["readout_switch"], switch)
                        self.assertEqual(seen[-1][2], precision == "LEGACY_BF16")
            with self.assertRaisesRegex(RuntimeError, "scope test"):
                with precision_execution(model, "FP32_REFERENCE", "off"):
                    raise RuntimeError("scope test")
        self.assertTrue(all(torch.equal(v, state[k]) for k, v in model.state_dict().items()))

    def test_fp32_runtime_context_disables_nested_autocast_and_restores_selection(self):
        model = ContractOnlyModel()
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            with torch.autocast("cpu", dtype=torch.bfloat16):
                with precision_execution(model, "FP32_REFERENCE", "on",
                        coordinates=(torch.zeros(1, 4), torch.zeros(1))) as facts:
                    self.assertFalse(torch.is_autocast_enabled("cpu"))
                    self.assertFalse(torch.is_autocast_enabled("cuda"))
                    self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                    self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
                    self.assertEqual(facts["native_precision"], "FP32_REFERENCE")
                self.assertTrue(torch.is_autocast_enabled("cpu"))
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
        finally:
            torch.set_float32_matmul_precision(previous)
            torch.backends.cuda.matmul.allow_tf32 = False

    def test_real_small_readout_dtype_hooks_and_exception_cleanup(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            module = CornerReadout(8, 4, 2).requires_grad_(False).eval()
            module.W_O.weight.fill_(.1)
        q = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 13
        memory = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 17
        before = torch.get_rng_state().clone()
        with DtypeAudit(module, "FP32_REFERENCE") as audit:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = module(q, memory)
            audit.observe(dict(site="embedding", role=torch.zeros(1, 8),
                               known_mask=torch.tensor([True]), hidden=output))
        self.assertEqual(audit.summary()["status"], "PASS")
        self.assertEqual(audit.summary()["floating_dtypes"], ["torch.float32"])
        self.assertTrue({"norm", "W_Q", "W_K", "W_V", "W_O"}.issubset(
            {r.get("module") for r in audit.records}))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules()))
        with self.assertRaisesRegex(RuntimeError, "dtype violation"):
            with DtypeAudit(module, "FP32_REFERENCE"):
                module(q.to(torch.bfloat16), memory)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules()))

    def test_invalid_route_and_trainable_weights_refused_before_execution(self):
        model = ContractOnlyModel()
        for precision, switch in (("FP16", "on"), ("FP32_REFERENCE", "per_layer")):
            with self.assertRaises(ValueError):
                with precision_execution(model, precision, switch):
                    self.fail("invalid policy entered")
        model.linear.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            with precision_execution(model, "FP32_REFERENCE"):
                self.fail("trainable model accepted")


if __name__ == "__main__":
    unittest.main()
