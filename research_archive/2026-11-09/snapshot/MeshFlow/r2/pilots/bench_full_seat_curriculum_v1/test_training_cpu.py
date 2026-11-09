"""CPU state-machine and real-checkpoint tests; no model forward or update.

Writes a preparation gate independent of the not-yet-frozen registration, so
the registration can include the gate hash without a circular dependency.
"""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "r2/src"))
from . import train as M
from . import training_state as S
from .common import OUT, SOURCE, HEAD, HEAD_SHA256
import numpy as np
import torch


def contract():
    return dict(stage_updates={"NATIVE": 2, "SEAM": 1},
                stage_full_steps={"NATIVE": [0, 2], "SEAM": [0, 1]})


def record_update(ledger, reg, stage, step):
    for micro in range(8):
        context = dict(**S.cursor(reg, stage, step), microbatch=micro)
        for kind in ("train_forward", "train_backward"):
            ledger.call(kind, lambda: None, context)
    ledger.call("optimizer_update", lambda: None, S.cursor(reg, stage, step))


class StateContractTests(unittest.TestCase):
    def test_native_to_seam_preserves_global_cursor(self):
        reg = contract()
        last = S.cursor(reg, "NATIVE", 2)
        first = S.cursor(reg, "SEAM", 0)
        self.assertEqual((last["global_step"], last["curriculum_step"]),
                         (first["global_step"], first["curriculum_step"]))
        self.assertEqual(S.cursor(reg, "SEAM", 1)["global_step"], 12003)
        self.assertEqual(first["seam_total_step"], 0)
        for stage in ("NATIVE", "SEAM"):
            with self.assertRaises(ValueError):
                S.cursor(reg, stage, reg["stage_updates"][stage] + 1)

    def test_checkpoint_budget_must_include_endpoints(self):
        reg = contract()
        reg["stage_full_steps"]["NATIVE"] = [0, 1]
        with self.assertRaises(ValueError):
            S.stage_spec(reg, "NATIVE")

    def test_exact_resume_rejects_stale_checkpoint_and_extra_head(self):
        reg = contract()
        with tempfile.TemporaryDirectory(prefix="ledger_", dir=OUT / "cpu_tests") as tmp:
            ledger = M.T.historical.CallLedger(Path(tmp) / "calls.jsonl", S.limits(reg, "NATIVE"),
                                              "NATIVE", "synthetic", synchronize=False)
            S.require_cursor(ledger, reg, "NATIVE", 0)
            record_update(ledger, reg, "NATIVE", 1)
            S.require_cursor(ledger, reg, "NATIVE", 1)
            with self.assertRaises(RuntimeError):
                S.require_cursor(ledger, reg, "NATIVE", 0)
            with self.assertRaises(RuntimeError):
                ledger.call("head_forward", lambda: None, {})
            ledger2 = M.T.historical.CallLedger(ledger.path, S.limits(reg, "NATIVE"),
                                               "NATIVE", "resume", synchronize=False)
            S.require_cursor(ledger2, reg, "NATIVE", 1)
            record_update(ledger2, reg, "NATIVE", 2)
            S.require_cursor(ledger2, reg, "NATIVE", 2)
            with self.assertRaises(RuntimeError):
                ledger2.call("optimizer_update", lambda: None, {})

    def test_partial_microbatch_and_wrong_global_cannot_resume(self):
        reg = contract()
        with tempfile.TemporaryDirectory(prefix="partial_", dir=OUT / "cpu_tests") as tmp:
            ledger = M.T.historical.CallLedger(Path(tmp) / "calls.jsonl", S.limits(reg, "SEAM"),
                                              "SEAM", "synthetic", synchronize=False)
            context = dict(**S.cursor(reg, "SEAM", 1), microbatch=0)
            ledger.call("train_forward", lambda: None, context)
            with self.assertRaises(RuntimeError):
                S.require_cursor(ledger, reg, "SEAM", 0)
            with self.assertRaises(RuntimeError):
                S.require_cursor(ledger, reg, "SEAM", 1)
        with tempfile.TemporaryDirectory(prefix="wrong_", dir=OUT / "cpu_tests") as tmp:
            ledger = M.T.historical.CallLedger(Path(tmp) / "calls.jsonl", S.limits(reg, "SEAM"),
                                              "SEAM", "synthetic", synchronize=False)
            record_update(ledger, reg, "SEAM", 1)
            ledger.rows[0]["context"]["global_step"] -= 2
            with self.assertRaises(RuntimeError):
                S.require_cursor(ledger, reg, "SEAM", 1)

    def test_native_owner_labels_do_not_constrain_free_corners(self):
        n, k = 128, 2
        anchors = np.full((n, 3), -1, np.int64)
        anchors[:k] = np.arange(k * 3).reshape(k, 3)
        known = np.zeros(n, bool)
        known[:k] = True
        sample = dict(N=n, K=k, anchor_ids=anchors, known_mask=known,
            valid_mask=np.ones(n, bool), occupied_labels=np.asarray([True, False]),
            occupied_for_FM=np.asarray([False, False]), free_coordinate_count=9 * (n-k))
        self.assertEqual(S.verify_sample(sample, "NATIVE")["active_seam_slots"], 0)
        with self.assertRaises(ValueError):
            S.verify_sample(sample, "SEAM")
        sample["occupied_for_FM"] = sample["occupied_labels"].copy()
        sample["anchor_ids"][k, :2] = [0, 1]
        sample["free_coordinate_count"] -= 6
        self.assertEqual(S.verify_sample(sample, "SEAM")["active_seam_slots"], 1)
        with self.assertRaises(ValueError):
            S.verify_sample(sample, "NATIVE")

    def test_old_adamw_state_restored_without_forward_or_update(self):
        # Use an inert tiny module to test the state path, never a neural forward.
        class ParametersOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
                self.readout_mode = "none"
        model = ParametersOnly().eval()
        optimizer = M.make_optimizer(model, None)
        optimizer.state[model.weight] = dict(step=torch.tensor(12000.),
            exp_avg=torch.tensor([.1, .2]), exp_avg_sq=torch.tensor([.3, .4]))
        saved = optimizer.state_dict()
        clone = M.make_optimizer(model, None)
        clone.load_state_dict(saved)
        S.validate_optimizer(model, clone, 12000)
        self.assertEqual(M.T.base.tree_hash(saved), M.T.base.tree_hash(clone.state_dict()))
        with self.assertRaises(ValueError):
            S.validate_optimizer(model, clone, 0)
        with self.assertRaises(ValueError):
            S.validate_optimizer(model, M.make_optimizer(model, None), 12000)


def main():
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation must not start with CUDA initialized")
    (OUT / "cpu_tests").mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StateContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    reg = dict(S.SOURCE_IDENTITIES, source_checkpoint=str(SOURCE), head_source_checkpoint=str(HEAD),
               head_source_sha256=HEAD_SHA256, head_state_sha256=S.HEAD_STATE_SHA256)
    if M.digest(SOURCE) != reg["source_checkpoint_sha256"] or M.digest(HEAD) != HEAD_SHA256:
        raise ValueError("Real protected input file hash differs")
    payload = M._common_payload(reg)
    model, optimizer = M._build_restore(payload, 12000, "cpu")
    head = M._head_state(reg)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation initialized CUDA")
    report = dict(status="PASS", schema="bench_full_seat_curriculum_training_cpu_gate_v1",
        tests_run=result.testsRun, real_source=dict(S.SOURCE_IDENTITIES),
        head_source_sha256=HEAD_SHA256, head_state_sha256=M.state_hash(head),
        optimizer_groups=len(optimizer.param_groups), optimizer_states=len(optimizer.state),
        optimizer_steps=sorted({float(v["step"]) for v in optimizer.state.values()}),
        generator_parameters=sum(p.numel() for p in model.parameters()),
        CPU_only=True, CUDA_initialized=False, real_model_forwards=0, real_backwards=0,
        real_optimizer_updates=0, head_forwards=0, OT_calls=0,
        source_hashes={str(Path(__file__).with_name(f)): M.digest(Path(__file__).with_name(f))
                      for f in ("train.py", "training_state.py", "test_training_cpu.py")},
        scope="Synthetic recovery/cursor/role invariants plus real COMMON CPU state/moment import. No neural repeatability or convergence claim.")
    M.T.write(OUT / "training_cpu_gate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
