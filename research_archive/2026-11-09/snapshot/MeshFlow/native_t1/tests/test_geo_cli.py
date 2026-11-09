"""CPU contract tests for explicit, portable T_geo entry points."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import torch

from native_t1 import train_cli, portable_checkpoint, geometry_checkpoint
from native_t1 import __main__ as cli


class GeoCliTests(unittest.TestCase):
    def args(self, root, *extra):
        return train_cli.build_parser().parse_args([
            "--inputs", str(root/"inputs.npz"), "--out", str(root/"run"),
            "--updates", "2", *extra])

    def test_invalid_branch_combinations_fail_before_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = [
                ["--init-geo", "geo.pt"],
                ["--init-t1", "pure.pt", "--seed", "-1"],
                ["--init-t1", "pure.pt", "--seed", str(2**32-70000)],
                ["--init-t1", "pure.pt", "--context-encoder", "geo", "--encoder-seed", "-1"],
                ["--init-t1", "pure.pt", "--context-encoder", "geo", "--encoder-seed", str(2**63)],
                ["--init-t1", "pure.pt", "--encoder-seed", "5"],
                ["--init-geo", "geo.pt", "--context-encoder", "geo", "--encoder-seed", "5"],
                ["--official-checkpoint", "last.pt", "--continue-stream"],
                ["--init-geo", "geo.pt", "--context-encoder", "geo", "--continue-stream", "--seed", "5"]]
            with mock.patch.object(train_cli, "load_archive", side_effect=AssertionError("IO before validation")):
                for case in cases:
                    with self.subTest(case=case), self.assertRaises(ValueError):
                        train_cli.run(self.args(root, *case))

    def test_geo_sample_profile_is_explicit(self):
        args = cli.build_parser().parse_args([
            "sample", "--condition", "C.npy", "--out", "out", "--seed", "1", "--profile", "trained-geo"])
        with self.assertRaises(ValueError):
            cli.resolve_profile(args)
        args.checkpoint = Path("geo.pt")
        loader, checkpoint = cli.resolve_profile(args)
        self.assertIs(loader, geometry_checkpoint.load_geometry_checkpoint)
        self.assertEqual(checkpoint, args.checkpoint)

    def test_fresh_and_existing_geo_use_correct_loader_and_saver(self):
        for resume in (False, True):
            with self.subTest(resume=resume), tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
                root = Path(temporary)
                args = self.args(root, "--context-encoder", "geo",
                                 "--init-geo" if resume else "--init-t1", "source.pt")
                model = torch.nn.Module()
                model.context_encoder = torch.nn.Linear(2, 3, bias=False)
                model.set_trainable = lambda value: model.requires_grad_(value)
                stream = SimpleNamespace(seed=1010, state_dict=lambda: {}, last_audit={},
                                         next_effective_batch=lambda: [None]*8)
                archive = SimpleNamespace(path=args.inputs, sha256="a"*64)
                stack.enter_context(mock.patch.object(train_cli, "load_archive", return_value=archive))
                stack.enter_context(mock.patch.object(train_cli, "make_stream", return_value=(stream, "test")))
                stack.enter_context(mock.patch.object(train_cli, "configure_stable_runtime"))
                identity = {"cumulative_updates": 1500, "state_sha256": "b"*64, "checkpoint_sha256": "c"*64}
                pure = stack.enter_context(mock.patch.object(portable_checkpoint, "load_trained_checkpoint",
                                                              return_value=(model, identity)))
                geo = stack.enter_context(mock.patch.object(geometry_checkpoint, "load_geometry_checkpoint",
                                                             return_value=(model, identity)))
                attach = stack.enter_context(mock.patch("native_t1.context_geometry.attach_context_encoder",
                                                         return_value={"mode": "geo"}))
                optimizer = object()
                stack.enter_context(mock.patch.object(train_cli, "make_optimizer", return_value=optimizer))
                stack.enter_context(mock.patch.object(train_cli, "train_step", return_value={"loss": 1.}))
                save = stack.enter_context(mock.patch.object(geometry_checkpoint, "save_geometry_checkpoint",
                                                              return_value={"schema": geometry_checkpoint.SCHEMA}))
                pure_save = stack.enter_context(mock.patch.object(train_cli, "save_checkpoint",
                                                                   side_effect=AssertionError("pure saver used for geo")))
                with contextlib.redirect_stdout(io.StringIO()):
                    report = train_cli.run(args)
                self.assertEqual(pure.call_count, int(not resume))
                self.assertEqual(geo.call_count, int(resume))
                self.assertEqual(attach.call_count, int(not resume))
                if not resume:
                    attach.assert_called_once_with(model, "geo", seed=1010)
                self.assertEqual(report["completed_updates"], 2)
                self.assertEqual(report["base_cumulative_updates"], 1500)
                self.assertEqual(report["context_encoder"], "geo")
                self.assertEqual(save.call_count, 1)
                self.assertEqual(save.call_args.kwargs["completed"], 2)
                self.assertEqual(save.call_args.kwargs["base_updates"], 1500)
                self.assertEqual(save.call_args.kwargs["base_identity"], identity)
                pure_save.assert_not_called()
                self.assertIn("encoder", report["logs"][-1])

    def test_update_metrics_measure_real_delta_and_postclip_gradient(self):
        encoder = torch.nn.Linear(2, 3, bias=False)
        before = {"weight": encoder.weight.detach().clone()}
        encoder.weight.grad = torch.ones_like(encoder.weight)*2
        with torch.no_grad():
            encoder.weight.add_(.25)
        result = train_cli.encoder_update_metrics(encoder, before)["weight"]
        self.assertAlmostEqual(result["actual_update_l2"], (6*.25**2)**.5, places=6)
        self.assertAlmostEqual(result["gradient_l2_after_clip"], (6*4)**.5, places=6)


if __name__ == "__main__":
    unittest.main()
