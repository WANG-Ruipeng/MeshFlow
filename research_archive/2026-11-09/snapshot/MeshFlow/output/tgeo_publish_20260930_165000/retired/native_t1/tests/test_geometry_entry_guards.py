"""CPU mocks prove generation cannot bypass failed/incomplete prior stages."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from native_t1 import geometry_generation as generation

COUNTS = dict(blocked_generate_calls=0, metadata_only_passed_guard_calls=0,
              model_loads=0, model_forwards=0, backwards=0, optimizer_updates=0,
              complete_rollouts=0, evaluation_array_loads=0)


class _ReachedRegistration(Exception):
    pass


class GeometryEntryGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.args = SimpleNamespace(out=self.out, checkpoint=None)
        self.write("gates.json", dict(status="PASS"))
        self.write("training_summary.json", dict(status="PASS", total_optimizer_updates=1500))
        self.records = {}
        for arm in generation.ARMS:
            directory = self.out/arm
            directory.mkdir()
            endpoint = directory/"step500.pt"
            # Placeholder proves that the guard checks existence without loading.
            endpoint.write_bytes(b"CPU test placeholder: never torch.load")
            self.records[arm] = dict(status="PASS", completed_updates=500,
                counts=dict(optimizer_updates=500), update_in_progress=False,
                checkpoints=[dict(path=str(endpoint), cumulative_updates=2000)])
            self.write(arm+"/training.json", self.records[arm])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.forbidden = []
        for name in ("_load", "load_initial", "load_trained_checkpoint", "load_geometry_checkpoint",
                     "load_evaluation", "clamped_sample", "score_output", "file_sha256"):
            self.forbidden.append(self.stack.enter_context(mock.patch.object(
                generation, name, side_effect=AssertionError("forbidden before guard: "+name))))
        for owner, name in ((generation.np, "load"), (generation.np, "savez_compressed"),
                            (generation.torch, "load"), (generation.torch.cuda, "_lazy_init")):
            self.forbidden.append(self.stack.enter_context(mock.patch.object(
                owner, name, side_effect=AssertionError("forbidden IO/CUDA: "+name))))

    def write(self, name, data):
        (self.out/name).write_text(json.dumps(data), encoding="utf-8")

    def assert_blocked(self):
        COUNTS["blocked_generate_calls"] += 1
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            generation.generate(self.args)
        self.assertFalse((self.out/"raw").exists())
        self.assertFalse((self.out/"generation.json").exists())
        for operation in self.forbidden:
            operation.assert_not_called()

    def test_missing_gate_blocks_before_any_asset_or_model(self):
        (self.out/"gates.json").unlink()
        self.assert_blocked()

    def test_stopped_running_or_missing_gate_status_blocks(self):
        for status in ("STOPPED", "RUNNING", None):
            with self.subTest(status=status):
                self.write("gates.json", dict(status=status))
                self.assert_blocked()

    def test_missing_failed_or_partial_training_summary_blocks(self):
        (self.out/"training_summary.json").unlink()
        self.assert_blocked()
        for value in (dict(status="STOPPED", total_optimizer_updates=1500),
                      dict(status="PASS", total_optimizer_updates=1499),
                      dict(status="PASS")):
            with self.subTest(value=value):
                self.write("training_summary.json", value)
                self.assert_blocked()

    def test_every_arm_must_be_complete_with_500_actual_updates(self):
        for arm in generation.ARMS:
            path=self.out/arm/"training.json"
            path.unlink()
            self.assert_blocked()
            for change in (dict(status="STOPPED"), dict(completed_updates=499),
                           dict(counts=dict(optimizer_updates=499)),
                           dict(update_in_progress=True)):
                with self.subTest(arm=arm, change=change):
                    self.write(arm+"/training.json", dict(self.records[arm], **change))
                    self.assert_blocked()
            self.write(arm+"/training.json", self.records[arm])

    def test_each_final_checkpoint_must_be_present_and_cumulative2000(self):
        for arm in generation.ARMS:
            for checkpoints in ([], [dict(path=self.records[arm]["checkpoints"][-1]["path"], cumulative_updates=1750)],
                                [dict(path=str(self.out/arm/"missing.pt"), cumulative_updates=2000)]):
                with self.subTest(arm=arm, checkpoints=checkpoints):
                    row=copy.deepcopy(self.records[arm]);row["checkpoints"]=checkpoints
                    self.write(arm+"/training.json", row)
                    self.assert_blocked()
            self.write(arm+"/training.json", self.records[arm])

    def test_complete_metadata_can_reach_later_source_registration_check_only(self):
        original = generation.read_json
        def read(path):
            if Path(path).name == "registration.json":
                raise _ReachedRegistration()
            return original(path)
        COUNTS["metadata_only_passed_guard_calls"] += 1
        with mock.patch.object(generation, "read_json", side_effect=read), self.assertRaises(_ReachedRegistration):
            generation.generate(self.args)
        self.assertFalse((self.out/"raw").exists())
        self.assertFalse((self.out/"generation.json").exists())
        for operation in self.forbidden:
            operation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
