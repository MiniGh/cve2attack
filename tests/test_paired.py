"""Tests for paired CVE-level candidate comparison uncertainty."""

import unittest

from cve2attack.evaluation.paired import paired_recall_comparison
from cve2attack.schemas import CandidateRecord, TechniqueCandidate


def record(cve_id, techniques):
    return CandidateRecord(
        cve_id=cve_id,
        domain="Enterprise",
        candidates=tuple(
            TechniqueCandidate(technique_id=value, score=1.0, sources=("test",))
            for value in techniques
        ),
    )


class PairedComparisonTests(unittest.TestCase):
    def test_reports_direction_counts_and_positive_interval(self):
        truth = {
            "CVE-2024-1": {"T1001"},
            "CVE-2024-2": {"T1002"},
            "CVE-2024-3": {"T1003"},
        }
        left = [record(cve_id, ["T9999"]) for cve_id in truth]
        right = [record(cve_id, [next(iter(labels))]) for cve_id, labels in truth.items()]

        result = paired_recall_comparison(
            left,
            right,
            truth,
            left_name="V1",
            right_name="V5",
            bootstrap_iterations=100,
            seed=7,
        )["cutoffs"]["10"]

        self.assertEqual(result["delta"], 1.0)
        self.assertEqual(result["ci95_low"], 1.0)
        self.assertEqual(result["improved_cves"], 3)
        self.assertEqual(result["same_cves"], 0)
        self.assertEqual(result["worse_cves"], 0)


if __name__ == "__main__":
    unittest.main()
