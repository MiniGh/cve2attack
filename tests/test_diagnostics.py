"""Focused tests for long-ranking and complementarity diagnostics."""

import unittest

from cve2attack.evaluation.diagnostics import (
    practical_failure_diagnosis,
    rank_distribution,
    recall_curve,
    union_oracle,
)


class CandidateDiagnosticTests(unittest.TestCase):
    def test_recall_curve_marks_points_beyond_publication_limit_unavailable(self):
        truth = {"CVE-2024-1": {"T1001", "T1002"}}
        predictions = {"CVE-2024-1": ["", "T1001", "T1002"]}

        curve = recall_curve(
            predictions,
            truth,
            cutoffs=(1, 3, 5),
            observable_through=3,
        )

        self.assertEqual(curve["points"]["1"]["micro_recall"], 0.0)
        self.assertEqual(curve["points"]["3"]["micro_recall"], 1.0)
        self.assertFalse(curve["points"]["5"]["observable"])
        self.assertIsNone(curve["points"]["5"]["micro_recall"])

    def test_rank_distribution_and_practical_failure_use_best_source_rank(self):
        rows = [
            {"cve_id": "CVE-2024-1", "technique_id": "T1001", "ranks": {"V1": 2, "V2": 40}},
            {"cve_id": "CVE-2024-2", "technique_id": "T1002", "ranks": {"V1": 70, "V2": 35}},
            {"cve_id": "CVE-2024-3", "technique_id": "T1003", "ranks": {"V1": None, "V2": 80}},
            {"cve_id": "CVE-2024-4", "technique_id": "T1004", "ranks": {"V1": None, "V2": None}},
        ]

        distribution = rank_distribution(rows, "V1")
        diagnosis = practical_failure_diagnosis(rows, ["V1", "V2"])

        self.assertEqual(distribution["rank_2_3"], 1)
        self.assertEqual(distribution["rank_over_50"], 1)
        self.assertEqual(distribution["unranked"], 2)
        self.assertEqual(
            diagnosis["counts"],
            {"top_20": 1, "rank_21_50": 1, "rank_over_50": 1, "unranked": 1},
        )

    def test_union_oracle_reports_real_union_budget_and_filters_invalid_slots(self):
        truth = {"CVE-2024-1": {"T1003"}}
        rankings = {
            "V1": {"CVE-2024-1": ["T1001", "T1002"]},
            "SMET": {"CVE-2024-1": ["", "T1003"]},
        }

        result = union_oracle(
            rankings=rankings,
            truth=truth,
            source_names=["V1", "SMET"],
            cutoffs=(2, 3),
            source_limits={"V1": 2, "SMET": 2},
        )

        self.assertEqual(result["points"]["2"]["micro_recall"], 1.0)
        self.assertEqual(result["points"]["2"]["mean_union_candidates"], 3.0)
        self.assertFalse(result["points"]["3"]["observable"])


if __name__ == "__main__":
    unittest.main()
