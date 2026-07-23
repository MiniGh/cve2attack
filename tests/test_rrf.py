"""Tests for deterministic, label-free Reciprocal Rank Fusion."""

import unittest

from cve2attack.fusion.rrf import fuse_rrf_records
from cve2attack.schemas import CandidateRecord, TechniqueCandidate


def _record(cve_id: str, technique_ids: list[str]) -> CandidateRecord:
    return CandidateRecord(
        cve_id=cve_id,
        candidates=tuple(
            TechniqueCandidate(technique_id, score=1.0 / rank)
            for rank, technique_id in enumerate(technique_ids, start=1)
        ),
        domain="Enterprise",
    )


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_consensus_candidates_outrank_single_source_candidates(self):
        sources = {
            "raw": [_record("CVE-2024-1", ["T1001", "T1002", "T1003"])],
            "rewrite": [_record("CVE-2024-1", ["T1002", "T1003", "T1004"])],
        }

        records = fuse_rrf_records(
            sources,
            cohort=["CVE-2024-1"],
            top_k=2,
            source_depth=3,
            rank_constant=60.0,
        )

        self.assertEqual(
            [candidate.technique_id for candidate in records[0].candidates],
            ["T1002", "T1003"],
        )
        self.assertEqual(
            records[0].candidates[0].metadata["source_ranks"],
            {"raw": 2, "rewrite": 1},
        )
        self.assertEqual(records[0].metadata["source_depth"], 3)

    def test_source_depth_limits_the_internal_candidate_pool(self):
        sources = {
            "raw": [_record("CVE-2024-1", ["T1001", "T1003"])],
            "rewrite": [_record("CVE-2024-1", ["T1002", "T1003"])],
        }

        records = fuse_rrf_records(
            sources,
            cohort=["CVE-2024-1"],
            top_k=2,
            source_depth=1,
            rank_constant=60.0,
        )

        self.assertEqual(
            [candidate.technique_id for candidate in records[0].candidates],
            ["T1001", "T1002"],
        )
        self.assertNotIn("T1003", [candidate.technique_id for candidate in records[0].candidates])

    def test_rejects_non_positive_weights(self):
        with self.assertRaisesRegex(ValueError, "weight must be positive"):
            fuse_rrf_records(
                {"raw": [_record("CVE-2024-1", ["T1001"])]},
                cohort=["CVE-2024-1"],
                top_k=1,
                source_depth=1,
                rank_constant=60.0,
                weights={"raw": 0.0},
            )


if __name__ == "__main__":
    unittest.main()
