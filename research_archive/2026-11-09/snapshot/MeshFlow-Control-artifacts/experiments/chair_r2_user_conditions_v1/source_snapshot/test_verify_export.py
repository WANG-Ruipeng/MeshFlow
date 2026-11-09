"""Negative controls for preservation and original-ID connectivity checks."""
import unittest

import numpy as np

from verify_export import connected_components, same_bytes, triangle_areas


class IndependentVerifierTests(unittest.TestCase):
    def test_bitwise_detects_signed_zero_dtype_and_shape(self):
        positive = np.asarray([0.0], dtype=np.float32)
        negative = np.asarray([-0.0], dtype=np.float32)
        self.assertTrue(np.array_equal(positive, negative))
        self.assertFalse(same_bytes(positive, negative))
        self.assertFalse(same_bytes(positive, positive.astype(np.float64)))
        self.assertFalse(same_bytes(positive, positive.reshape(1, 1)))
        self.assertTrue(same_bytes(positive, positive.copy()))

    def test_shared_vertex_is_not_edge_connectivity(self):
        faces = np.asarray([[0, 1, 2], [2, 1, 3], [3, 4, 5], [5, 4, 6]])
        self.assertEqual(connected_components(faces, [0, 1]), [[0, 1]])
        self.assertEqual(connected_components(faces, [0, 1, 2, 3]), [[0, 1], [2, 3]])
        self.assertEqual(connected_components(faces, [0, 3]), [[0], [3]])

    def test_degenerate_and_small_positive_triangles_remain_distinct(self):
        faces = np.asarray([[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                            [[0, 0, 0], [1e-8, 0, 0], [0, 1e-8, 0]]], dtype=np.float32)
        values = triangle_areas(faces)
        self.assertEqual(values[0], 0)
        self.assertGreater(values[1], 0)


if __name__ == "__main__":
    unittest.main()
