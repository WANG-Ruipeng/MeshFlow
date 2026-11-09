"""CPU-only identity/strict-load tests; no backbone forward or training."""
import copy
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
from native_t1 import portable_checkpoint as pc
from native_t1.artifacts import file_sha256, state_sha256


class TinyBackbone(torch.nn.Module):
    """Small state carrier used only to exercise PyTorch strict assign loading."""
    def __init__(self, **kwargs):
        super().__init__()
        self.x_embedder = torch.nn.Linear(2, 2)

    def forward(self, *args, **kwargs):
        raise AssertionError("A model forward is forbidden in these loader tests")


class TinyNative(torch.nn.Module):
    def __init__(self, backbone, trainable=False):
        super().__init__()
        self.backbone = backbone
        self.role_embedding = torch.nn.Parameter(torch.zeros(2, 768))
        self.set_trainable(trainable)

    def set_trainable(self, value):
        self.requires_grad_(value)
        return self.eval()

    def forward(self, *args, **kwargs):
        raise AssertionError("A model forward is forbidden in these loader tests")


def metadata(state=None):
    return dict(schema=pc.TRAINING_SCHEMA, native_meta=copy.deepcopy(pc.NATIVE_META),
                model={} if state is None else state, optimizer={}, stream_state={},
                config_sha256=pc.OFFICIAL_CONFIG_SHA256, completed_updates=500,
                base_cumulative_updates=0, cumulative_updates=500, stream_seed=10,
                input_sha256="1" * 64, loss="masked_free_fm", num_faces=112,
                train_K=[4, 8, 12], optimizer_reset_at_start=True,
                model_state_sha256="2" * 64, optimizer_step_in_progress=False,
                batch_in_progress=False)


class PortableCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = TinyNative(TinyBackbone())
        with torch.no_grad():
            self.model.backbone.x_embedder.weight.fill_(0.25)
            self.model.backbone.x_embedder.bias.zero_()
        self.state = self.model.state_dict()
        self.sha = state_sha256(self.model)
        self.backbone_sha = state_sha256(self.model.backbone)

    def tiny_constructors(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(pc, "DiT", TinyBackbone))
        stack.enter_context(mock.patch.object(pc, "NativeInpaintingModel", TinyNative))
        return stack

    def test_portable_identity_and_progress(self):
        self.assertEqual(pc.validate_training_metadata(metadata())["cumulative_updates"], 500)
        changes = [dict(schema="a_then_fc_branch_v1"), dict(native_meta={}),
                   dict(loss="FM_plus_FC"), dict(train_K=[12]), dict(num_faces=800),
                   dict(cumulative_updates=501), dict(completed_updates=True),
                   dict(stream_seed=-1), dict(optimizer_reset_at_start=False),
                   dict(optimizer_step_in_progress=True), dict(batch_in_progress=True),
                   dict(model_state_sha256="bad"), dict(optimizer=None), dict(stream_state=None)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                pc.validate_training_metadata(dict(metadata(), **change))
        with self.assertRaises(ValueError):
            pc.validate_training_metadata(metadata(), "0" * 64)

    def test_canonical_config_lf_crlf_and_tamper(self):
        source = pc.DEFAULT_CONFIG.read_text(encoding="utf-8-sig")
        hashes = []
        for name, content in (("lf.yaml", source.encode()),
                              ("crlf.yaml", source.replace("\n", "\r\n").encode()),
                              ("bom.yaml", b"\xef\xbb\xbf" + source.encode())):
            path = self.root / name
            path.write_bytes(content)
            hashes.append(pc.config_sha256(path))
            self.assertEqual(pc._read_config(path)["hidden_dim"], 768)
        self.assertEqual(hashes, [pc.OFFICIAL_CONFIG_SHA256] * 3)
        wrong = self.root / "wrong.yaml"
        wrong.write_text(source.replace("hidden_dim: 768", "hidden_dim: 384"))
        with mock.patch.object(pc.torch, "load") as load, self.assertRaises(ValueError):
            pc.load_trained_checkpoint(self.root / "unused.pt", wrong, device="cpu")
        load.assert_not_called()

    def test_wrong_official_file_rejected_before_read(self):
        path = self.root / "wrong.pt"
        path.write_bytes(b"not official weights")
        with mock.patch.object(pc.torch, "load") as load, \
             mock.patch.object(pc, "DiT") as ctor, self.assertRaises(ValueError):
            pc.load_official_initialization(path, device="cpu")
        load.assert_not_called()
        ctor.assert_not_called()

    def test_ema_selection_requires_ema_and_uniform_prefix(self):
        for payload in ({"model": self.state}, {"ema": {}},
                        {"ema": {"module.x": torch.zeros(1), "y": torch.zeros(1)}}):
            with self.subTest(payload=list(payload)), self.assertRaises(ValueError):
                pc._ema_state(payload)
        x = torch.zeros(1)
        self.assertIs(pc._ema_state({"ema": {"module.x": x}, "model": {}})["x"], x)

    def test_official_strict_assign_zero_role(self):
        path = self.root / "ema.pt"
        torch.save({"ema": self.model.backbone.state_dict(), "model": {}}, path)
        rng = torch.get_rng_state().clone()
        with self.tiny_constructors(), \
             mock.patch.object(pc, "OFFICIAL_FILE_SHA256", file_sha256(path)), \
             mock.patch.object(pc, "OFFICIAL_NATIVE_SHA256", self.sha), \
             mock.patch.object(pc, "OFFICIAL_BACKBONE_SHA256", self.backbone_sha):
            loaded, audit = pc.load_official_initialization(path, device="cpu")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(audit["selected_state"], "ema")
        self.assertEqual(audit["role_nonzero"], 0)
        self.assertEqual(audit["missing_keys"], [])
        self.assertFalse(loaded.training)
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
        self.assertEqual(state_sha256(loaded), self.sha)

    def test_strict_rejects_missing_extra_shape_dtype(self):
        config = pc._read_config(pc.DEFAULT_CONFIG)
        missing = dict(self.state)
        missing.pop("role_embedding")
        extra = dict(self.state, unexpected=torch.zeros(1))
        shape = dict(self.state, role_embedding=torch.zeros(1, 768))
        dtype = dict(self.state, role_embedding=torch.zeros(2, 768, dtype=torch.bfloat16))
        for state in (missing, extra, shape, dtype):
            with self.subTest(keys=list(state)), self.tiny_constructors(), \
                 self.assertRaises((ValueError, RuntimeError)):
                pc._strict_native(state, config, self.sha)

    def test_trained_strict_load_and_frozen_default(self):
        payload = metadata(self.state)
        payload["model_state_sha256"] = self.sha
        path = self.root / "trained.pt"
        torch.save(payload, path)
        with self.tiny_constructors():
            loaded, audit = pc.load_trained_checkpoint(path, device="cpu")
        self.assertEqual(audit["cumulative_updates"], 500)
        self.assertEqual(audit["checkpoint_sha256"], file_sha256(path))
        self.assertEqual(audit["state_sha256"], self.sha)
        self.assertFalse(audit["official_EMA_loaded"])
        self.assertFalse(loaded.training)
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()))

    def test_bad_trained_identity_rejected_before_model(self):
        for change in (dict(native_meta={}), dict(schema="native_connected_patch_v1"),
                       dict(config_sha256="0" * 64)):
            path = self.root / "invalid.pt"
            torch.save(dict(metadata(self.state), **change), path)
            with mock.patch.object(pc, "DiT") as ctor, self.assertRaises(ValueError):
                pc.load_trained_checkpoint(path, device="cpu")
            ctor.assert_not_called()

    def test_state_digest_and_nonfinite_rejected(self):
        config = pc._read_config(pc.DEFAULT_CONFIG)
        with self.tiny_constructors(), self.assertRaises(ValueError):
            pc._strict_native(self.state, config, "0" * 64)
        with torch.no_grad():
            self.model.role_embedding[0, 0] = float("nan")
        with self.tiny_constructors(), self.assertRaisesRegex(ValueError, "Nonfinite"):
            pc._strict_native(self.model.state_dict(), config, state_sha256(self.model))

    def test_import_no_asset_read_model_execution_or_cuda(self):
        code = """
import sys, torch
from unittest import mock
with mock.patch.object(torch, 'load', side_effect=AssertionError('asset read')), mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA')):
    import native_t1.portable_checkpoint
assert not torch.cuda.is_initialized()
assert not any(x == 'experiments' or x.startswith('experiments.') for x in sys.modules)
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=pc.DEFAULT_CONFIG.parents[2],
                                env=env, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
