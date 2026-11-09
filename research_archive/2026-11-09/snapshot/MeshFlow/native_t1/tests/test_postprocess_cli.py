"""Synthetic CPU-only file/CLI contracts for optional GEO_C postprocessing."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from native_t1 import postprocess_cli as cli


def triangles():
    return np.asarray([
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        [[.01, 0., 0.], [0., 0., 1.], [1., 0., 1.]],
        [[3., 0., 0.], [4., 0., 0.], [3., 1., 0.]],
    ], dtype=np.float32)


def same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class PostprocessFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-postprocess-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, raw, name="input.npy"):
        path = self.root / name
        np.save(path, raw, allow_pickle=False)
        return path

    def test_flat_npy_roundtrip_all_source_maps_and_unchanged_input(self):
        raw = triangles()
        source = self.save(raw.reshape(-1, 9))
        before = source.read_bytes()
        out = self.root / "flat"
        receipt = cli.run(source, 1, out)
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["model_calls"], 0)
        self.assertEqual(receipt["stages"], {"input": "RAW", "output": "GEO_C"})
        self.assertEqual(receipt["C_original_index_graph"], "NOT_PROVIDED")
        self.assertEqual(receipt["input"]["original_shape"], [3, 9])
        self.assertEqual(source.read_bytes(), before)
        persisted = json.loads((out / "postprocess.json").read_text())
        self.assertEqual(receipt, persisted)
        with np.load(out / "post.npz", allow_pickle=False) as p:
            self.assertTrue(all(not p[k].dtype.hasobject for k in p.files))
            self.assertTrue(same_bits(p["C"], raw[:1]))
            self.assertTrue(same_bits(p["vertices"][p["faces"]], p["output"]))
            self.assertTrue(same_bits(raw.reshape(-1, 3)[p["corner_target_input_ids"]][p["source_face_ids"]], p["output"]))
            self.assertTrue(same_bits(p["corner_vertex_ids"][p["source_face_ids"]], p["faces"]))
            self.assertTrue(np.all(p["corner_displacement"] <= cli.RADIUS))
            self.assertGreater(float(p["corner_displacement"].max()), 0.)
            self.assertEqual(p["corner_target_input_ids"][1, 0], 0)
            pv, pf = cli.read_indexed_ply(out / "post.ply")
            self.assertTrue(same_bits(pv, p["vertices"]))
            self.assertTrue(same_bits(pf, p["faces"]))
            self.assertTrue(same_bits(pv[pf], p["output"]))
        for filename, record in receipt["files"].items():
            self.assertEqual(record["sha256"], cli.file_sha256(out / filename))
        self.assertIn(b"format binary_little_endian 1.0\n", (out / "post.ply").read_bytes())

    def test_npz_explicit_keys_default_output_and_raw(self):
        source = self.root / "input.npz"
        raw = triangles()
        np.savez(source, output=raw, raw=raw.reshape(-1, 9), unrelated=np.asarray([1]))
        before = source.read_bytes()
        for key in ("output", "raw"):
            with self.subTest(key=key):
                report = cli.run(source, 1, self.root / key, input_key=key)
                self.assertEqual(report["input"]["key"], key)
                self.assertEqual(report["status"], "PASS")
        self.assertEqual(source.read_bytes(), before)

    def test_signed_zero_and_duplicate_known_faces_survive_indexed_export(self):
        base = triangles()
        negative = base[0].copy()
        negative[0, 0] = np.float32(-0.)
        raw = np.stack([base[0], base[0], negative, base[2]])
        source = self.save(raw)
        out = self.root / "signed"
        receipt = cli.run(source, 3, out)
        self.assertEqual(receipt["output_faces"], 4)
        with np.load(out / "post.npz", allow_pickle=False) as p:
            self.assertTrue(same_bits(p["output"][:3], raw[:3]))
            self.assertTrue(same_bits(p["C"], raw[:3]))
            self.assertTrue(np.array_equal(p["faces"][0], p["faces"][1]))
            self.assertNotEqual(p["faces"][0, 0], p["faces"][2, 0])
            self.assertFalse(np.signbit(p["vertices"][p["faces"][0, 0], 0]))
            self.assertTrue(np.signbit(p["vertices"][p["faces"][2, 0], 0]))
            vertices, faces = cli.read_indexed_ply(out / "post.ply")
            self.assertTrue(same_bits(vertices[faces[:3]], raw[:3]))

    def test_deletion_provenance_only_free_and_all_free_removed_is_valid(self):
        raw = triangles()
        repeated = np.stack([raw[2, 0], raw[2, 0], raw[2, 1]])
        collinear = np.asarray([[5., 0., 0.], [6., 0., 0.], [7., 0., 0.]], dtype=np.float32)
        fixture = np.stack([raw[0], raw[0, ::-1], raw[2], raw[2, ::-1], repeated, collinear])
        report = cli.run(self.save(fixture), 1, self.root / "deletions")
        self.assertEqual([r["reason"] for r in report["face_records"]],
                         ["kept", "duplicate_C", "kept", "duplicate_free",
                          "exact_repeated_vertex", "exact_collinear"])
        with np.load(self.root / "deletions" / "post.npz", allow_pickle=False) as p:
            np.testing.assert_array_equal(p["source_face_ids"], [0, 2])
            np.testing.assert_array_equal(p["duplicate_of_source_face"], [-1, 0, -1, 2, -1, -1])
        removed = np.stack([raw[0], raw[0, ::-1]])
        report = cli.run(self.save(removed, "all.npy"), 1, self.root / "all")
        self.assertEqual(report["output_faces"], 1)
        vertices, faces = cli.read_indexed_ply(self.root / "all" / "post.ply")
        self.assertTrue(same_bits(vertices[faces], removed[:1]))

    def test_invalid_inputs_are_rejected_before_directory_creation(self):
        raw = triangles()
        nan = raw.copy()
        nan[1, 0, 0] = np.nan
        examples = [
            ("k0", raw, 0, ValueError), ("kN", raw, len(raw), ValueError),
            ("negative", raw, -1, ValueError), ("bool", raw, True, ValueError),
            ("fp64", raw.astype(np.float64), 1, TypeError),
            ("nan", nan, 1, ValueError), ("shape", raw.reshape(9, 3), 1, ValueError),
        ]
        for name, array, K, error in examples:
            with self.subTest(name=name):
                source = self.save(array, name + ".npy")
                original = source.read_bytes()
                out = self.root / name
                with self.assertRaises(error):
                    cli.run(source, K, out)
                self.assertFalse(out.exists())
                self.assertEqual(source.read_bytes(), original)
        missing = self.root / "missing.npz"
        np.savez(missing, raw=raw)
        with self.assertRaises(ValueError):
            cli.run(missing, 1, self.root / "missing")
        self.assertFalse((self.root / "missing").exists())
        objects = self.root / "object.npz"
        np.savez(objects, output=np.asarray([{"not": "geometry"}], dtype=object))
        with self.assertRaises(ValueError):
            cli.run(objects, 1, self.root / "object")
        self.assertFalse((self.root / "object").exists())

    def test_existing_output_is_never_overwritten(self):
        source = self.save(triangles())
        out = self.root / "existing"
        out.mkdir()
        marker = out / "postprocess.json"
        marker.write_bytes(b"existing evidence")
        with self.assertRaises(FileExistsError):
            cli.run(source, 1, out)
        self.assertEqual(marker.read_bytes(), b"existing evidence")
        self.assertEqual(list(out.iterdir()), [marker])

    def test_export_failure_leaves_stopped_receipt_and_preserves_raw(self):
        source = self.save(triangles())
        before = source.read_bytes()
        out = self.root / "failed"
        with patch.object(cli, "write_indexed_ply", side_effect=OSError("injected export failure")):
            with contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["--input", str(source), "--known-faces", "1", "--out", str(out)])
        self.assertEqual(code, 1)
        report = json.loads((out / "postprocess.json").read_text())
        self.assertEqual(report["status"], "STOPPED")
        self.assertEqual(report["error"]["type"], "OSError")
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(source.read_bytes(), before)
        self.assertIn("post.npz", report["files"])
        self.assertNotIn("contracts", report)

    def test_ply_reader_rejects_corrupt_body(self):
        path = self.root / "mesh.ply"
        cli.write_indexed_ply(path, triangles()[0], np.asarray([[0, 1, 2]], dtype=np.int64))
        good = path.read_bytes()
        for bad in (good[:-1], good + b"x"):
            path.write_bytes(bad)
            with self.assertRaises(ValueError):
                cli.read_indexed_ply(path)

    def test_top_level_command_without_model_imports_and_failure_exit_codes(self):
        guard = self.root / "import_guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text(
            "import sys\n"
            "class NoModelImports:\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname.split('.')[0] in {'torch', 'models', 'experiments'}:\n"
            "            raise RuntimeError('Forbidden model dependency: ' + fullname)\n"
            "sys.meta_path.insert(0, NoModelImports())\n",
            encoding="utf-8")
        repo = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join([str(guard), str(repo)]),
                   PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        # Prove the fresh interpreter activates the guard, rather than merely
        # assuming sitecustomize was found.
        guarded = subprocess.run([sys.executable, "-B", "-c", "import torch"],
                                 cwd=repo, env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(guarded.returncode, 0)
        self.assertIn("Forbidden model dependency: torch", guarded.stderr)
        source = self.save(triangles())
        out = self.root / "subprocess"
        prefix = [sys.executable, "-B", "-m", "native_t1", "postprocess"]
        args = ["--input", str(source), "--known-faces", "1", "--out", str(out)]
        success = subprocess.run(prefix + args, cwd=repo, env=env,
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(success.returncode, 0, success.stdout + success.stderr)
        self.assertEqual(json.loads((out / "postprocess.json").read_text())["status"], "PASS")
        original_receipt = (out / "postprocess.json").read_bytes()
        duplicate = subprocess.run(prefix + args, cwd=repo, env=env,
                                   capture_output=True, text=True, timeout=60)
        self.assertNotEqual(duplicate.returncode, 0, duplicate.stdout + duplicate.stderr)
        self.assertIn("FileExistsError", duplicate.stderr)
        self.assertEqual((out / "postprocess.json").read_bytes(), original_receipt)
        bad_source = self.save(triangles().astype(np.float64), "bad.npy")
        bad_out = self.root / "bad_subprocess"
        invalid = subprocess.run(prefix + ["--input", str(bad_source), "--known-faces", "1",
                                         "--out", str(bad_out)],
                                 cwd=repo, env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(invalid.returncode, 0, invalid.stdout + invalid.stderr)
        self.assertIn("TypeError", invalid.stderr)
        self.assertFalse(bad_out.exists())


if __name__ == "__main__":
    unittest.main()
