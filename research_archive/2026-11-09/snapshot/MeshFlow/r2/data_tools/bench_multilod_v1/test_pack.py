"""CPU tests of provenance, fixed-C identity and source-role refresh.

Runs on one retained real training case; creates no data files or model objects.
"""
import json
from pathlib import Path
import unittest

import numpy as np

from .pack import C_IMMUTABLE, pack_case, validate_case, _same

ROOT = Path(__file__).resolve().parents[3]


class RealCasePackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m = json.loads((ROOT / "experiments/r2_bench_high_joint_coverage_v1/data/manifest.json").read_text())
        row = m["train"][0]
        path = Path(row["case_path"])
        if not path.exists():
            path = ROOT / row["case_path"].split("/MeshFlow/", 1)[1]
        with np.load(path, allow_pickle=False) as a:
            cls.case = {k: a[k].copy() for k in a.files}

    def test_reordered_faces_preserve_c_and_reindex_new_domain(self):
        a = self.case
        kept = np.arange(len(a["source_faces"]), dtype=np.int64)[::-1]
        before = {k: x.tobytes() for k, x in a.items()}
        out, proof = pack_case(a, a["model_vertices"], a["source_faces"][kept], kept)
        self.assertEqual(proof["status"], "PASS")
        self.assertFalse(np.array_equal(out["known_ids"], a["known_ids"]))
        self.assertTrue(_same(kept[out["known_ids"]], a["known_ids"]))
        self.assertTrue(_same(out["source_face_ids"], out["known_ids"]))
        for key in C_IMMUTABLE:
            self.assertTrue(_same(out[key], a[key]), key)
        self.assertNotEqual(str(out["target_version_id"]), str(a["target_version_id"]))
        self.assertEqual(before, {k: x.tobytes() for k, x in a.items()})
        validate_case(out, a)

    def test_removed_owner_is_not_left_as_stale_supervision(self):
        a = self.case
        edge = int(np.flatnonzero(a["occupied_labels"])[0])
        face = int(a["edge_owner_face"][edge])
        kept = np.setdiff1d(np.arange(int(a["N"]), dtype=np.int64), [face])
        out, _ = pack_case(a, a["model_vertices"], a["source_faces"][kept], kept)
        self.assertFalse(bool(out["source_supported_labels"][edge]))
        self.assertEqual(int(out["edge_owner_face"][edge]), -1)
        self.assertEqual(int(out["N"]), int(a["N"]) - 1)

    def test_moved_free_vertex_uses_recorded_inverse_only(self):
        a = self.case
        v = a["model_vertices"].copy()
        moved = int(np.setdiff1d(np.arange(len(v)), a["C_source_vertex_ids"])[0])
        v[moved, 0] += np.float32(.00001)
        kept = np.arange(int(a["N"]), dtype=np.int64)
        out, proof = pack_case(a, v, a["source_faces"], kept)
        self.assertEqual(proof["moved_vertex_slots"], [moved])
        self.assertTrue(_same(out["transform_json"], a["transform_json"]))
        unchanged = np.arange(len(v)) != moved
        self.assertTrue(_same(out["normalized_vertices"][unchanged], a["normalized_vertices"][unchanged]))
        self.assertTrue(_same(out["source_vertices"][unchanged], a["source_vertices"][unchanged]))
        tr = json.loads(str(a["transform_json"]))
        norm = v[moved] * np.float32(tr["dataset_std"] / tr["model_scale"])
        self.assertTrue(_same(out["normalized_vertices"][moved], norm))

    def test_reject_changed_c_coordinates(self):
        a = self.case
        v = a["model_vertices"].copy()
        v[a["C_source_vertex_ids"][0], 0] += np.float32(.01)
        with self.assertRaisesRegex(ValueError, "fixed C vertex"):
            pack_case(a, v, a["source_faces"], np.arange(int(a["N"]), dtype=np.int64))

    def test_reject_changed_c_winding(self):
        a = self.case
        f = a["source_faces"].copy()
        f[a["known_ids"][0]] = f[a["known_ids"][0]][[0, 2, 1]]
        with self.assertRaisesRegex(ValueError, "winding"):
            pack_case(a, a["model_vertices"], f, np.arange(int(a["N"]), dtype=np.int64))

    def test_reject_missing_c_or_unknown_transform(self):
        a = self.case
        kept = np.setdiff1d(np.arange(int(a["N"]), dtype=np.int64), a["known_ids"][:1])
        with self.assertRaisesRegex(ValueError, "C face was deleted"):
            pack_case(a, a["model_vertices"], a["source_faces"][kept], kept)
        unknown = dict(a, transform_json=np.asarray("{}"))
        with self.assertRaisesRegex(ValueError, "Unknown source transform"):
            pack_case(unknown, a["model_vertices"], a["source_faces"], np.arange(int(a["N"]), dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
