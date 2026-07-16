import unittest

from cve2attack.evaluation.metrics import evaluate
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


if __name__ == "__main__":
    unittest.main()
