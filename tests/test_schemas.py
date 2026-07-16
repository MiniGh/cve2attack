import unittest

from cve2attack.schemas import CandidateRecord


class CandidateSchemaTests(unittest.TestCase):
    def test_reads_legacy_string_list(self):
        record = CandidateRecord.from_dict(
            {"cve_id": "CVE-2024-1", "techniques": ["T1190", "T1059.001"]}
        )
        self.assertEqual([item.technique_id for item in record.candidates], ["T1190", "T1059"])

    def test_reads_legacy_scored_list(self):
        record = CandidateRecord.from_dict(
            {"cve_id": "CVE-2024-1", "techniques": [{"id": "T1190", "score": 0.75}]}
        )
        self.assertEqual(record.candidates[0].technique_id, "T1190")
        self.assertEqual(record.candidates[0].score, 0.75)

    def test_writes_only_canonical_schema(self):
        record = CandidateRecord.from_dict(
            {"cve_id": "CVE-2024-1", "techniques": [{"id": "T1190", "score": 0.75}]}
        )
        payload = record.to_dict()
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertIn("candidates", payload)
        self.assertNotIn("techniques", payload)


if __name__ == "__main__":
    unittest.main()
