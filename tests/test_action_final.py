"""Tests for Stage-1 final action-audit helpers."""

import unittest

from cve2attack.evaluation.action_final import classify_case, spearman_correlation


class ActionFinalAuditTests(unittest.TestCase):
    def test_case_classes_use_the_controlled_top20_boundary(self):
        self.assertEqual(classify_case(30, 10), "new_v5_hit")
        self.assertEqual(classify_case(10, 30), "lost_v1_hit")
        self.assertEqual(classify_case(5, 12), "retained_hit")
        self.assertEqual(classify_case(80, 45), "unresolved_rank_21_50")
        self.assertEqual(classify_case(None, 80), "unresolved_beyond_50")

    def test_spearman_handles_ties_and_monotonic_values(self):
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [4, 5, 6]), 1.0)
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [6, 5, 4]), -1.0)
        self.assertIsNone(spearman_correlation([1, 1, 1], [1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
