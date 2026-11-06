"""Metric contract checks; no model checkpoint or GPU required."""
import unittest

import numpy as np

from qwen4_vision_ref import compare


class VisionComparisonTests(unittest.TestCase):
    def test_grid_mismatch_with_identical_shape_fails(self):
        values = np.ones((6, 4))
        self.assertFalse(compare(values, values, (2, 3), (3, 2), 0.99)["pass"])

    def test_nonfinite_reference_or_actual_fails(self):
        good = np.ones((2, 4))
        for value in [float("nan"), float("inf")]:
            bad = good.copy()
            bad[0, 0] = value
            for actual, reference in [(bad, good), (good, bad)]:
                self.assertFalse(compare(actual, reference, (1, 2), (1, 2), 0.99)["pass"])

    def test_one_bad_token_fails_despite_high_mean(self):
        reference = np.tile([1.0, 0.0], (1000, 1))
        actual = reference.copy()
        actual[73] = [0.0, 1.0]
        result = compare(actual, reference, (20, 50), (20, 50), 0.99)
        self.assertGreater(result["mean_cos"], 0.99)
        self.assertFalse(result["pass"])
        self.assertEqual(result["worst_token"], 73)

    def test_scaled_vectors_pass_cosine_but_report_absolute_error(self):
        reference = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = compare(reference * 2, reference, (1, 2), (1, 2), 0.99)
        self.assertTrue(result["pass"])
        self.assertEqual(result["max_abs"], 4.0)


if __name__ == "__main__":
    unittest.main()
