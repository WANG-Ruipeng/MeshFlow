"""CPU-only contracts on reduced-width real Native/DiT arithmetic.

No production checkpoint is loaded and no CUDA context is initialized. These
operator checks are not additional large-model training/probe jobs.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import unittest
from unittest import mock

import torch
from torch import nn

from . import model as M
from meshflow_control.models.backbone.equidit import DiT


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(31519)
        model = M.NativeModel.__new__(M.NativeModel)
        nn.Module.__init__(model)
        model.backbone = DiT(hidden_dim=12, num_heads=3, num_layers=12,
            max_length=800, gradient_checkpointing=False, class_dropout_prob=0.,
            version=3, pe_freq=2, mixed_precision="fp32", use_dit_like_pe=False,
            face_cond=True, face_bin=20, use_rmsnorm=True, use_qknorm=True)
        model.role_embedding = nn.Parameter(torch.zeros(2, 12))
        model.context_encoder = None
        model.context_observer = None
        model.readout_mode = "none"
        model.readouts = nn.ModuleDict()
        model.readout_observer = None
        model.experiment_trainable = True
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name and parameter.ndim == 1:
                    parameter.fill_(1.)
                else:
                    parameter.normal_(0., .065)
        model.eval()
        return model


def inputs():
    generator = torch.Generator().manual_seed(12122)
    x = torch.randn(1, 128, 9, generator=generator, requires_grad=True)
    known = torch.zeros(1, 128, dtype=torch.bool)
    known[:, :4] = True
    tokens = torch.tensor([[[0., .2, .1, .4, .3, .1, 1., 0.],
                            [0., -.2, 0., .5, .3, .4, 0., 1.]]])
    return x, torch.tensor([.63]), torch.tensor([128]), torch.ones_like(known), known, tokens, torch.ones(1, 2, dtype=torch.bool)


class LayoutOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU tests must not inherit initialized CUDA")

    def setUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(mock.patch.object(M, "assert_native_runtime", lambda *a, **k: None))
        self.stack.enter_context(mock.patch.object(M.native, "assert_native_runtime", lambda *a, **k: None))
        self.addCleanup(self.stack.close)

    def test_branch_initialization_preserves_cpu_rng_and_is_seeded(self):
        state = torch.get_rng_state().clone()
        with mock.patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("Do not alter CUDA RNG")):
            a = M.LayoutReadout(12, seed=88)
            b = M.LayoutReadout(12, seed=88)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
        self.assertEqual(torch.count_nonzero(a.output.weight).item(), 0)

    def test_attach_preserves_inherited_parameter_identity_and_order(self):
        model = fixture()
        names = [n for n, _ in model.named_parameters()]
        parameters = list(model.parameters())
        M.attach_layout(model)
        self.assertEqual(names, [n for n, _ in model.named_parameters() if not n.startswith("layout_branch.")])
        self.assertTrue(all(a is b for a, b in zip(parameters, M.base_parameters(model))))
        self.assertTrue(all(not module.training for module in model.modules()))
        with self.assertRaises(ValueError):
            M.attach_layout(model)

    def test_zero_exit_exact_output_and_input_vjp(self):
        original = fixture()
        candidate = M.attach_layout(copy.deepcopy(original))
        x, t, y, valid, known, tokens, lv = inputs()
        a = original(x, t, y, valid, known)
        b = candidate(x, t, y, valid, known, layout_tokens=tokens, layout_valid=lv)
        self.assertTrue(torch.equal(a, b))
        cot = torch.linspace(-1., 1., a.numel()).reshape_as(a)
        ga = torch.autograd.grad((a * cot).sum(), x, retain_graph=True)[0]
        gb = torch.autograd.grad((b * cot).sum(), x)[0]
        self.assertTrue(torch.equal(ga, gb))

    def test_missing_layout_is_exact_after_nonzero_branch(self):
        original = fixture()
        candidate = M.attach_layout(copy.deepcopy(original))
        with torch.no_grad():
            candidate.layout_branch.output.weight.fill_(.1)
        x, t, y, valid, known, tokens, lv = inputs()
        a = original(x, t, y, valid, known)
        b = candidate(x, t, y, valid, known, layout_tokens=tokens, layout_valid=torch.zeros_like(lv))
        c = candidate(x, t, y, valid, known)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a, c))

    def test_known_padding_and_missing_rows_have_zero_delta(self):
        branch = M.LayoutReadout(12)
        with torch.no_grad():
            branch.output.weight.fill_(.15)
        x, _, _, valid, known, tokens, lv = inputs()
        valid[:, -5:] = False
        hidden = torch.randn(1, 384, 12)
        result, delta = branch(hidden, x, known, valid, tokens, lv)
        inactive = (known | ~valid).repeat_interleave(3, 1)
        self.assertTrue(torch.equal(result[inactive], hidden[inactive]))
        self.assertEqual(torch.count_nonzero(delta[inactive]).item(), 0)
        self.assertGreater(torch.count_nonzero(delta[~inactive]).item(), 0)

    def test_nonzero_layout_changes_real_model_prediction(self):
        candidate = M.attach_layout(fixture())
        with torch.no_grad():
            candidate.layout_branch.output.weight.normal_(0., .04)
        x, t, y, valid, known, tokens, lv = inputs()
        a = candidate(x, t, y, valid, known)
        b = candidate(x, t, y, valid, known, layout_tokens=tokens, layout_valid=lv)
        self.assertGreater((b-a).abs().max().item(), 1e-7)

    def test_zero_exit_has_real_backward_and_second_update_reaches_encoder(self):
        branch = M.LayoutReadout(12)
        optimizer = torch.optim.AdamW(branch.parameters(), lr=.001)
        x, _, _, valid, known, tokens, lv = inputs()
        hidden = torch.randn(1, 384, 12, requires_grad=True)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            result, _ = branch(hidden, x, known, valid, tokens, lv)
            result.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters()))
            self.assertGreater(branch.output.weight.grad.abs().sum().item(), 0.)
            if step == 1:
                self.assertGreater(branch.layout_embed[0].weight.grad.abs().sum().item(), 0.)
                self.assertGreater(branch.query_hidden.weight.grad.abs().sum().item(), 0.)
            optimizer.step()

    def test_null_training_still_has_zero_finite_branch_gradients(self):
        branch = M.LayoutReadout(12)
        x, _, _, valid, known, tokens, lv = inputs()
        hidden = torch.randn(1, 384, 12, requires_grad=True)
        result, delta = branch(hidden, x, known, valid, tokens, torch.zeros_like(lv))
        result.square().mean().backward()
        self.assertEqual(torch.count_nonzero(delta).item(), 0)
        for p in branch.parameters():
            self.assertIsNotNone(p.grad)
            self.assertEqual(torch.count_nonzero(p.grad).item(), 0)

    def test_context_is_scoped_no_state_change_and_nested_rejected(self):
        candidate = M.attach_layout(fixture())
        x, t, y, valid, known, tokens, lv = inputs()
        before = {k: v.clone() for k, v in candidate.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with candidate.layout_context(tokens, lv):
                candidate(x, t, y, valid, known)
                with self.assertRaises(RuntimeError):
                    with candidate.layout_context(tokens, lv):
                        pass
                with self.assertRaises(ValueError):
                    candidate(x, t, y, valid, known, layout_tokens=tokens, layout_valid=lv)
                raise RuntimeError("sentinel")
        self.assertIsNone(candidate._layout_payload)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in candidate.state_dict().items()))

    def test_invalid_tokens_and_extents_refused(self):
        candidate = M.attach_layout(fixture())
        x, t, y, valid, known, tokens, lv = inputs()
        for change in ("negative_extent", "wrong_role", "nan"):
            altered = tokens.clone()
            if change == "negative_extent": altered[0, 0, 3] = -1.
            elif change == "wrong_role": altered[0, 0, 6:] = 0.
            else: altered[0, 0, 0] = float("nan")
            with self.assertRaises(ValueError):
                candidate(x, t, y, valid, known, layout_tokens=altered, layout_valid=lv)

    def test_source_and_input_arrays_never_mutated(self):
        candidate = M.attach_layout(fixture())
        arrays = inputs()
        prior = [v.detach().clone() for v in arrays]
        x, t, y, valid, known, tokens, lv = arrays
        candidate(x, t, y, valid, known, layout_tokens=tokens, layout_valid=lv)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(arrays, prior)))

    def test_configure_replaces_microbatch_and_clear_restores_bypass(self):
        model = M.attach_layout(fixture())
        x, t, y, valid, known, tokens, lv = inputs()
        with torch.no_grad(): model.layout_branch.output.weight.normal_(0., .04)
        baseline = model(x, t, y, valid, known)
        model.configure_layout(tokens, lv)
        self.assertGreater((model(x, t, y, valid, known)-baseline).abs().max().item(), 0.)
        model.configure_layout(tokens, torch.zeros_like(lv))
        self.assertTrue(torch.equal(model(x, t, y, valid, known), baseline))
        model.clear_layout()
        self.assertIsNone(model._layout_payload)
        with model.layout_context(tokens, lv):
            with self.assertRaises(RuntimeError): model.configure_layout(tokens, lv)
            with self.assertRaises(RuntimeError): model.clear_layout()

    def test_state_split_roundtrip(self):
        model = M.attach_layout(fixture())
        base, layout = M.split_base_state(model.state_dict())
        other = fixture()
        other.load_state_dict(base, strict=True)
        M.attach_layout(other, seed=57)
        other.layout_branch.load_state_dict(layout, strict=True)
        self.assertTrue(all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items()))
        with self.assertRaises(ValueError): M.split_base_state(base)

    def test_area_centroid_and_global_translation_consistency(self):
        x, _, _, valid, known, tokens, lv = inputs()
        centroid = M.c_area_centroid(x, known)
        translated = x.reshape(1, 128, 3, 3) + torch.tensor([1., -2., 3.])
        shifted = M.c_area_centroid(translated.reshape_as(x), known)
        self.assertTrue(torch.allclose(shifted-centroid, torch.tensor([[1., -2., 3.]]), atol=2e-6, rtol=0.))

    def test_no_cuda_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())


def run_recorded_gate(path):
    """Record explicitly scoped small-CPU operator work, never production work."""
    import datetime
    import hashlib
    import json
    from pathlib import Path
    counts = dict(reduced_native_block_calls=0, layout_module_calls=0,
                  autograd_backward_calls=0, input_vjp_grad_calls=0,
                  tiny_AdamW_updates=0)
    def counted(key, function):
        def call(*args, **kwargs):
            counts[key] += 1
            return function(*args, **kwargs)
        return call
    block = counted("reduced_native_block_calls", M.native_block_forward)
    branch = counted("layout_module_calls", M.LayoutReadout.forward)
    backward = counted("autograd_backward_calls", torch.autograd.backward)
    grad = counted("input_vjp_grad_calls", torch.autograd.grad)
    step = counted("tiny_AdamW_updates", torch.optim.AdamW.step)
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(M, "native_block_forward", block))
        stack.enter_context(mock.patch.object(M.native, "native_block_forward", block))
        stack.enter_context(mock.patch.object(M.LayoutReadout, "forward", branch))
        stack.enter_context(mock.patch.object(torch.autograd, "backward", backward))
        stack.enter_context(mock.patch.object(torch.autograd, "grad", grad))
        stack.enter_context(mock.patch.object(torch.optim.AdamW, "step", step))
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(LayoutOperators)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    source = [Path(M.__file__).resolve(), Path(__file__).resolve()]
    receipt = dict(status="PASS" if result.wasSuccessful() else "FAIL",
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tests=result.testsRun, failures=[(str(t), e) for t, e in result.failures],
        errors=[(str(t), e) for t, e in result.errors],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        scope="CPU reduced-width operator tests; not a production neural experiment",
        fixture=dict(real_DiT_arithmetic=True, hidden_dimension=12, heads=3,
            blocks=12, face_slots=128, production_checkpoint_loaded=False,
            original_GPU_runtime_assertions_mocked=True,
            mocked_guards=["model.assert_native_runtime", "natural_N_runtime.assert_native_runtime"],
            mock_reason="CPU float32 reduced fixture; production files retain full guards"),
        counted_CPU_work=counts,
        production_work=dict(generator_forwards=0, generator_backwards=0,
            generator_optimizer_updates=0, checkpoint_loads=0, CUDA_initialized=torch.cuda.is_initialized()),
        earlier_development_runs=[dict(tests=14, reason="initial implementation"),
            dict(tests=15, reason="trainer prefix/configure API alignment"),
            dict(tests=15, reason="all-modules-eval contract")],
        earlier_CPU_call_counts="not individually instrumented; all reduced fixtures and CPU only",
        runtime_GPU_certification=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--receipt":
        raise SystemExit(0 if run_recorded_gate(sys.argv[2]) else 1)
    unittest.main()
