"""Small CPU checks for the protected proposal contract, not mesh quality claims."""
import unittest

import numpy as np

from r2.data_tools.bench_multilod_v1.simplifier import simplify


def grid(side=12):
    vertices = np.asarray([(x, y, 0.0) for y in range(side) for x in range(side)], np.float32)
    faces = []
    for y in range(side - 1):
        for x in range(side - 1):
            a = y * side + x
            faces.extend(((a, a + 1, a + side), (a + 1, a + side + 1, a + side)))
    return vertices, np.asarray(faces, np.int64)


class ProtectedSimplifierTests(unittest.TestCase):
    def test_fixed_faces_determinism_and_budget(self):
        v, f = grid()
        original_v, original_f = v.copy(), f.copy()
        locked = [110]
        target = 194
        a = simplify(v, f, locked, target, 20.0)
        b = simplify(v, f, locked, target, 20.0)
        self.assertGreater(a[3]["accepted_collapses"], 0)
        self.assertGreaterEqual(len(a[1]), target)
        self.assertLessEqual(len(a[1]), target + 1)
        self.assertEqual(a[0].shape, v.shape)
        for i in range(3):
            self.assertEqual(a[i].tobytes(), b[i].tobytes())
        self.assertEqual(a[3], b[3])
        index = {int(source_id): out for out, source_id in enumerate(a[2])}
        self.assertGreater(len(a[3]["effective_locked_face_ids"]), len(locked))
        for fi in a[3]["effective_locked_face_ids"]:
            self.assertEqual(a[1][index[fi]].tobytes(), f[fi].tobytes())
            self.assertEqual(a[0][a[1][index[fi]]].tobytes(), v[f[fi]].tobytes())
        self.assertEqual(v.tobytes(), original_v.tobytes())
        self.assertEqual(f.tobytes(), original_f.tobytes())

    def test_all_locked_is_unchanged(self):
        v, f = grid(5)
        vv, ff, ids, ledger = simplify(v, f, range(len(f)), len(f) - 8, 10.0)
        self.assertEqual(vv.tobytes(), v.tobytes())
        self.assertEqual(ff.tobytes(), f.tobytes())
        np.testing.assert_array_equal(ids, np.arange(len(f)))
        self.assertEqual(ledger["accepted_collapses"], 0)
        self.assertEqual(ledger["status"], "CONSTRAINT_LIMITED_PROPOSAL")

    def test_preexisting_duplicate_is_retained_not_deduplicated(self):
        v, f = grid(12)
        f = np.concatenate((f, f[[0]][:, [0, 2, 1]]), axis=0)
        vv, ff, ids, ledger = simplify(v, f, [], 201, 20.0)
        for source_id in (0, len(f) - 1):
            out = int(np.flatnonzero(ids == source_id)[0])
            self.assertEqual(ff[out].tobytes(), f[source_id].tobytes())
            self.assertEqual(vv[ff[out]].tobytes(), v[f[source_id]].tobytes())
        self.assertEqual(ledger["preexisting_duplicate_face_ids"], [0, len(f) - 1])
        self.assertGreaterEqual(len(ff), 201)


if __name__ == "__main__":
    unittest.main()
