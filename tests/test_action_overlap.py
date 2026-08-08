"""Tests for exact benchmark-CVE overlap auditing in ATT&CK procedures."""

import json
import tempfile
import unittest
from pathlib import Path

from cve2attack.evaluation.action_overlap import audit_procedure_overlap


class ActionOverlapAuditTests(unittest.TestCase):
    def test_counts_only_direct_benchmark_true_label_pairs(self):
        bundle = {
            "objects": [
                {
                    "type": "attack-pattern",
                    "id": "attack-pattern--parent",
                    "external_references": [
                        {"source_name": "mitre-attack", "external_id": "T1000"}
                    ],
                },
                {
                    "type": "relationship",
                    "id": "relationship--true",
                    "relationship_type": "uses",
                    "source_ref": "malware--one",
                    "target_ref": "attack-pattern--parent",
                    "description": "Actor exploited CAN-2024-12345.",
                },
                {
                    "type": "relationship",
                    "id": "relationship--outside",
                    "relationship_type": "uses",
                    "source_ref": "malware--two",
                    "target_ref": "attack-pattern--parent",
                    "description": "Actor exploited CVE-2024-99999.",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attack_bundle = root / "attack.json"
            attack_bundle.write_text(json.dumps(bundle), encoding="utf-8")
            benchmark = root / "benchmark"
            benchmark.mkdir()
            (benchmark / "CVE-2024.jsonl").write_text(
                json.dumps({"cve_id": "CVE-2024-12345", "techniques": ["T1000"]}) + "\n",
                encoding="utf-8",
            )
            summary, rows = audit_procedure_overlap(
                attack_bundle=attack_bundle,
                benchmark_dir=benchmark,
            )

        self.assertEqual(summary["directly_mentioned_cves"], 1)
        self.assertEqual(summary["direct_true_label_pairs"], 1)
        self.assertEqual(summary["cves_with_direct_true_label"], 1)
        self.assertTrue(rows[0]["is_benchmark_label"])


if __name__ == "__main__":
    unittest.main()
