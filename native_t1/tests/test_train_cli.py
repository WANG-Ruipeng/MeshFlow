"""Portable trainer contract checks with no real model, OT or optimizer update."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import torch

from native_t1 import train_cli
from native_t1 import __main__ as cli
from native_t1.artifacts import state_sha256


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def set_trainable(self, value):
        self.requires_grad_(value)
        return self


class FakeStream:
    seed = 10
    last_audit = {"shared_batch_sha256": "mock-input", "official_OT_call_count": 8}

    def __init__(self):
        self.index = 0

    def state_dict(self):
        return {"seed": self.seed, "arm": "Native_correct", "batch_index": self.index}

    def next_effective_batch(self):
        self.index += 1
        return [None]*8


class TrainerTests(unittest.TestCase):
    def parse(self, directory, *extra):
        return train_cli.build_parser().parse_args(
            ["--inputs", str(directory/"inputs.npz"), "--official-checkpoint", str(directory/"last.pt"),
             "--out", str(directory/"run"), *extra])

    def test_budget_must_be_explicit_positive_or_check_only(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            root = Path(temp)
            for extra in ([], ["--updates", "0"], ["--updates", "-1"], ["--updates", "2", "--check-only"]):
                with self.assertRaises(SystemExit):
                    self.parse(root, *extra)
            self.assertTrue(self.parse(root, "--check-only").check_only)
            self.assertEqual(self.parse(root, "--updates", "500").updates, 500)

    def test_existing_output_and_invalid_stream_options_fail_before_asset_io(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.parse(root, "--updates", "2")
            args.out.mkdir()
            with mock.patch.object(train_cli, "load_archive", side_effect=AssertionError("asset IO")):
                with self.assertRaises(FileExistsError):
                    train_cli.run(args)
                args.out = root/"new_run"
                args.continue_stream = True
                with self.assertRaises(ValueError):
                    train_cli.run(args)

    def test_root_cli_dispatch_and_trained_profile(self):
        with mock.patch("native_t1.train_cli.main", return_value="train") as train:
            self.assertEqual(cli.main(["train", "--check-only"]), "train")
            train.assert_called_once_with(["--check-only"])
        with mock.patch("native_t1.prepare.main", return_value="prepare") as prepare:
            self.assertEqual(cli.main(["prepare", "--help"]), "prepare")
            prepare.assert_called_once_with(["--help"])
        args = cli.build_parser().parse_args(["sample", "--condition", "C.npy", "--seed", "1",
                                             "--out", "out", "--profile", "trained"])
        with self.assertRaises(ValueError):
            cli.resolve_profile(args)
        args.checkpoint = Path("endpoint.pt")
        loader, path = cli.resolve_profile(args)
        from native_t1.portable_checkpoint import load_trained_checkpoint
        self.assertIs(loader, load_trained_checkpoint)
        self.assertEqual(path, args.checkpoint)

    def test_controller_respects_exact_budget_and_saves_portable_metadata(self):
        from native_t1 import portable_checkpoint
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            root = Path(temp)
            args = self.parse(root, "--updates", "2", "--save-every", "1")
            archive = SimpleNamespace(path=args.inputs, sha256="d"*64)
            model, stream = FakeModel(), FakeStream()
            optimizer = SimpleNamespace(state_dict=lambda: {"state": {}, "param_groups": []})
            stack.enter_context(mock.patch.object(train_cli, "load_archive", return_value=archive))
            stack.enter_context(mock.patch.object(train_cli, "make_stream", return_value=(stream, "mock")))
            stack.enter_context(mock.patch.object(train_cli, "configure_stable_runtime"))
            stack.enter_context(mock.patch.object(portable_checkpoint, "load_official_initialization",
                                                  return_value=(model, {"source": "mock"})))
            stack.enter_context(mock.patch.object(train_cli, "make_optimizer", return_value=optimizer))
            def step(model_arg, optimizer_arg, samples, *, counts):
                self.assertIs(model_arg, model)
                self.assertIs(optimizer_arg, optimizer)
                self.assertEqual(len(samples), 8)
                for key in ("forward_attempts", "forward_returns", "backward_attempts", "backward_returns"):
                    counts[key] += 8
                counts["optimizer_step_attempts"] += 1
                counts["optimizer_updates"] += 1
                return {"loss": 1., "grad_norm_before_clip": 0.}
            mocked_step = stack.enter_context(mock.patch.object(train_cli, "train_step", side_effect=step))
            before = state_sha256(model)
            with contextlib.redirect_stdout(io.StringIO()):
                report = train_cli.run(args)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(mocked_step.call_count, 2)
            self.assertEqual(report["completed_updates"], 2)
            self.assertEqual(len(report["checkpoints"]), 2)
            self.assertEqual(state_sha256(model), before)  # no real optimizer or model work
            payload = torch.load(report["checkpoints"][-1]["path"], weights_only=True, map_location="cpu")
            self.assertEqual(payload["schema"], train_cli.SCHEMA)
            self.assertEqual(payload["native_meta"], portable_checkpoint.NATIVE_META)
            self.assertEqual(payload["completed_updates"], 2)
            self.assertEqual(payload["cumulative_updates"], 2)
            self.assertEqual(payload["loss"], "masked_free_fm")
            self.assertFalse(payload["optimizer_step_in_progress"])
            self.assertEqual(payload["stream_state"]["batch_index"], 2)


if __name__ == "__main__":
    unittest.main()
