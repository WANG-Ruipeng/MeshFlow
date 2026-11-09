"""Synthetic CPU checks for topology-preserving boundary chain changes."""
import unittest
import numpy as np
from .boundary_simplify import analyze, simplify_boundaries, edgebook, _single_loop, _collinear_chain


def folded():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [0, 0, 1], [2, 0, 1]], float)
    faces = np.asarray([[0, 1, 4], [1, 3, 4], [1, 2, 3], [2, 1, 6], [1, 5, 6], [1, 0, 5]], int)
    return dict(model_vertices=vertices, source_faces=faces)


class BoundarySimplificationTests(unittest.TestCase):
    def test_two_perpendicular_patches_remove_shared_midpoint_together(self):
        original = folded()
        result, lineage, ledger = simplify_boundaries(original, np.asarray([0, 0, 0, 1, 1, 1]), L=1)
        self.assertEqual(len(result["source_faces"]), 4)
        self.assertNotIn(1, result["source_faces"])
        self.assertEqual(len(edgebook(result["source_faces"])[(0, 2)]), 2)
        self.assertEqual(ledger["removed_vertex_ids"], [1])
        self.assertTrue(np.array_equal(original["model_vertices"], result["model_vertices"]))
        self.assertEqual(len(original["source_faces"]), 6)
        self.assertTrue(all(len(set((0 if x < 3 else 1) for x in group)) == 1 for group in lineage))

    def test_semantic_boundary_junction_is_preserved(self):
        original = folded()
        result, _, ledger = simplify_boundaries(original, np.asarray([2, 0, 0, 1, 1, 1]), L=1)
        self.assertIn(1, result["source_faces"])
        self.assertEqual(ledger["vertex_rejections"]["1"], "PATCH_OR_SEMANTIC_JUNCTION")

    def test_disconnected_component_touching_midpoint_protects_contact(self):
        original = folded()
        original["model_vertices"] = np.concatenate([original["model_vertices"], [[1, 0, 0], [1, -1, 0], [1, -1, 1]]])
        original["source_faces"] = np.concatenate([original["source_faces"], [[7, 8, 9]]])
        audit = analyze(original["model_vertices"], original["source_faces"], L=1)
        self.assertNotIn(1, audit["candidates"])
        self.assertEqual(audit["rejection_reasons"][1], "UNINDEXED_SURFACE_CONTACT_ENDPOINT")

    def test_topological_point_contact_is_preserved(self):
        original = folded()
        original["model_vertices"] = np.concatenate([original["model_vertices"], [[1, -1, 0], [1, -1, 1]]])
        original["source_faces"] = np.concatenate([original["source_faces"], [[1, 7, 8]]])
        audit = analyze(original["model_vertices"], original["source_faces"], L=1)
        self.assertEqual(audit["rejection_reasons"][1], "NONMANIFOLD_VERTEX_OR_CONTACT")

    def test_hole_is_not_silently_filled(self):
        v = np.asarray([[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0], [1, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0]], float)
        f = np.asarray([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
        self.assertEqual(_single_loop(f)[1], "MULTI_LOOP_OR_HOLE")
        result, _, _ = simplify_boundaries(dict(model_vertices=v, source_faces=f), L=1)
        self.assertEqual({tuple(x) for x in result["source_faces"]}, {tuple(x) for x in f})

    def test_global_chain_prevents_accumulated_curve_straightening(self):
        x = np.arange(21, dtype=float)
        v = np.stack([x, .01 * x * x, np.zeros_like(x)], axis=1)
        self.assertTrue(all(_collinear_chain(v, [i - 1, i, i + 1], .011) for i in range(1, 20)))
        self.assertFalse(_collinear_chain(v, list(range(21)), .011))

    def test_near_curve_is_not_a_shared_collinear_chain(self):
        original = folded()
        original["model_vertices"][1, 1] = .01
        result, _, ledger = simplify_boundaries(original, L=1)
        self.assertIn(1, result["source_faces"])
        self.assertEqual(ledger["summary"]["removed_vertices"], 0)


if __name__ == "__main__":
    unittest.main()
