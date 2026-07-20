import unittest

from cve2attack.evaluation.metrics import evaluate
from cve2attack.evaluation.ranking import evaluate_rankings
from cve2attack.schemas import CandidateRecord


class EvaluationTests(unittest.TestCase):
    def test_missing_prediction_is_a_miss_and_reduces_coverage(self):
        records = [
            CandidateRecord.from_dict(
                {"cve_id": "CVE-2024-1", "techniques": ["T1190", "T1059"]}
            )
        ]
        truth = {
            "CVE-2024-1": {"T1190"},
            "CVE-2024-2": {"T1059"},
        }
        metrics = evaluate(records, truth)
        self.assertEqual(metrics.benchmark_cves, 2)
        self.assertEqual(metrics.predicted_cves, 1)
        self.assertEqual(metrics.coverage, 0.5)
        self.assertEqual(metrics.recall_at_10, 0.5)

    def test_ranking_metrics_name_macro_and_micro_recall_separately(self):
        predictions = {
            "CVE-2024-1": ["T1000", "T1001"],
            "CVE-2024-2": ["T9999"],
        }
        truth = {
            "CVE-2024-1": {"T1000", "T1001"},
            "CVE-2024-2": {"T1002"},
        }
        metrics = evaluate_rankings(predictions, truth)

        # Per-CVE recall is (1 + 0) / 2, while pooled-label recall is 2 / 3.
        self.assertEqual(metrics.macro_recall_at_5, 0.5)
        self.assertAlmostEqual(metrics.micro_recall_at_5, 2 / 3)
        self.assertEqual(metrics.hit_rate_at_5, 0.5)
        self.assertEqual(metrics.relevant_labels, 3)


if __name__ == "__main__":
    unittest.main()
