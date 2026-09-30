"""CPU-only counter fixtures; every OT callable is a replacement, never real OT."""
import random
import unittest
from unittest.mock import patch

import numpy as np
from utils import ot_utils

from native_t1.accounting import count_ot_calls


class GeometryAccountingTests(unittest.TestCase):
    def test_success_preserves_arguments_result_rng_and_restores(self):
        result, data, noise = object(), object(), object()
        calls = []

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return result

        counts = {"OT_calls": 4, "OT_returns": 3}
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        with patch.object(ot_utils, "optimal_sum_numpy", original):
            with count_ot_calls(counts):
                value = ot_utils.optimal_sum_numpy(data, noise, optimal=True,
                                                   dimension=3, flip_face=False)
                self.assertIs(value, result)
                self.assertIsNot(ot_utils.optimal_sum_numpy, original)
            self.assertIs(ot_utils.optimal_sum_numpy, original)
        self.assertEqual(counts, {"OT_calls": 5, "OT_returns": 4})
        self.assertEqual(calls, [((data, noise), dict(optimal=True, dimension=3,
                                                    flip_face=False))])
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(python_before, random.getstate())

    def test_failure_counts_attempt_and_restores_exception_identity(self):
        error = ValueError("replacement OT failure")

        def original(*args, **kwargs):
            raise error

        counts = {}
        with patch.object(ot_utils, "optimal_sum_numpy", original):
            with self.assertRaises(ValueError) as caught:
                with count_ot_calls(counts):
                    ot_utils.optimal_sum_numpy(None, None)
            self.assertIs(caught.exception, error)
            self.assertIs(ot_utils.optimal_sum_numpy, original)
        self.assertEqual(counts, {"OT_calls": 1, "OT_returns": 0})

    def test_body_failure_restores_and_later_scope_can_run(self):
        def original(*args, **kwargs):
            return None

        counts = {}
        with patch.object(ot_utils, "optimal_sum_numpy", original):
            with self.assertRaisesRegex(RuntimeError, "after preprocessing"):
                with count_ot_calls(counts):
                    ot_utils.optimal_sum_numpy(None, None)
                    raise RuntimeError("after preprocessing")
            self.assertIs(ot_utils.optimal_sum_numpy, original)
            with count_ot_calls(counts):
                ot_utils.optimal_sum_numpy(None, None)
            self.assertIs(ot_utils.optimal_sum_numpy, original)
        self.assertEqual(counts, {"OT_calls": 2, "OT_returns": 2})

    def test_nested_scope_rejected_without_double_counting(self):
        def original(*args, **kwargs):
            return None

        counts = {}
        with patch.object(ot_utils, "optimal_sum_numpy", original):
            with count_ot_calls(counts):
                wrapped = ot_utils.optimal_sum_numpy
                with self.assertRaisesRegex(RuntimeError, "cannot be nested"):
                    with count_ot_calls(counts):
                        self.fail("nested scope must not execute")
                self.assertIs(ot_utils.optimal_sum_numpy, wrapped)
                ot_utils.optimal_sum_numpy(None, None)
            self.assertIs(ot_utils.optimal_sum_numpy, original)
        self.assertEqual(counts, {"OT_calls": 1, "OT_returns": 1})


if __name__ == "__main__":
    unittest.main()
