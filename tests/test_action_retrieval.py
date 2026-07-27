"""Tests for action-corpus construction and parent-Technique aggregation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cve2attack.retrieval.action_generator import (
    aggregate_action_scores,
    retrieve_action_candidates,
)
from cve2attack.retrieval.action_kb import (
    ActionDocument,
    load_action_documents,
    sanitize_action_text,
)


class FakeEmbedder:
    model_name = "fake"

    def encode(self, texts, batch_size):
        return np.asarray(
            [[1.0, 0.0] if "web" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def action(action_id, technique_id, text, vulnerability_ids=()):
    return ActionDocument(
        action_id=action_id,
        technique_id=technique_id,
        technique_name=technique_id,
        tactics=(),
        source_type="procedure",
        source_stix_id=action_id,
        text=text,
        vulnerability_ids=tuple(vulnerability_ids),
    )


class ActionKnowledgeBaseTests(unittest.TestCase):
    def test_sanitizes_current_and_legacy_vulnerability_identifiers(self):
        text = sanitize_action_text(
            "Exploit CVE-2024-12345 and can-2002-0392. (Citation: Example)",
            max_chars=0,
        )
        self.assertEqual(text.count("[VULNERABILITY]"), 2)
        self.assertNotIn("CVE-2024-12345", text)
        self.assertNotIn("Citation", text)

    def test_loads_parent_and_subtechnique_actions_and_deduplicates(self):
        bundle = {
            "objects": [
                {
                    "type": "attack-pattern",
                    "id": "attack-pattern--parent",
                    "name": "Parent",
                    "description": "Parent behavior that is long enough.",
                    "kill_chain_phases": [{"phase_name": "execution"}],
                    "external_references": [
                        {"source_name": "mitre-attack", "external_id": "T1000"}
                    ],
                },
                {
                    "type": "attack-pattern",
                    "id": "attack-pattern--child",
                    "name": "Child",
                    "description": "Child behavior for CVE-2024-12345.",
                    "x_mitre_is_subtechnique": True,
                    "external_references": [
                        {"source_name": "mitre-attack", "external_id": "T1000.001"}
                    ],
                },
                {
                    "type": "relationship",
                    "id": "relationship--one",
                    "relationship_type": "uses",
                    "target_ref": "attack-pattern--child",
                    "description": "Actor exploited CAN-2002-0392 to run code.",
                },
                {
                    "type": "relationship",
                    "id": "relationship--duplicate",
                    "relationship_type": "uses",
                    "target_ref": "attack-pattern--child",
                    "description": "Actor exploited CVE-2023-9999 to run code.",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attack.json"
            path.write_text(json.dumps(bundle), encoding="utf-8")
            actions = load_action_documents(path, min_chars=5, max_chars=0)

        self.assertTrue(actions)
        self.assertEqual({item.technique_id for item in actions}, {"T1000"})
        self.assertTrue(any(item.source_type == "subtechnique_description" for item in actions))
        procedure_actions = [item for item in actions if item.source_type == "procedure"]
        self.assertEqual(len(procedure_actions), 1)
        self.assertEqual(
            procedure_actions[0].vulnerability_ids,
            ("CVE-2002-0392", "CVE-2023-9999"),
        )
        self.assertNotIn("CVE-", " ".join(item.text for item in actions))
        self.assertNotIn("CAN-", " ".join(item.text for item in actions))


class ActionRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.actions = [
            action("a", "T1000", "web exploit"),
            action("b", "T2000", "command execution"),
            action("c", "T2000", "another command"),
        ]

    def test_max_aggregation_returns_canonical_evidence(self):
        records = retrieve_action_candidates(
            queries={"CVE-2024-1": "web vulnerability"},
            actions=self.actions,
            action_embeddings=np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
            ),
            embedder=FakeEmbedder(),
            top_k=2,
            batch_size=2,
            aggregation="max",
        )
        best = records[0].candidates[0]
        self.assertEqual(best.technique_id, "T1000")
        self.assertEqual(best.sources, ("action_embedding",))
        self.assertEqual(best.metadata["action_evidence"][0]["action_id"], "a")
        self.assertEqual(records[0].to_dict()["schema_version"], "1.0")

    def test_rank_rrf_can_reward_two_independent_action_hits(self):
        candidates = aggregate_action_scores(
            scores=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            actions=self.actions,
            aggregation="rank_rrf",
            aggregation_top_m=2,
            rank_constant=1.0,
            top_k=2,
            evidence_limit=1,
            evidence_text_limit=100,
        )
        self.assertEqual([item.technique_id for item in candidates], ["T2000", "T1000"])

    def test_equal_scores_use_deterministic_technique_id_tie_break(self):
        candidates = aggregate_action_scores(
            scores=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
            actions=self.actions,
            aggregation="max",
            aggregation_top_m=3,
            rank_constant=60.0,
            top_k=2,
            evidence_limit=1,
            evidence_text_limit=100,
        )
        self.assertEqual([item.technique_id for item in candidates], ["T1000", "T2000"])

    def test_query_specific_procedure_is_excluded_without_removing_other_actions(self):
        actions = [
            action("direct", "T1000", "web exploit", ("CVE-2024-1",)),
            action("other", "T2000", "command execution"),
        ]
        records = retrieve_action_candidates(
            queries={"CVE-2024-1": "web vulnerability"},
            actions=actions,
            action_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            embedder=FakeEmbedder(),
            top_k=2,
            batch_size=2,
            aggregation="max",
            exclude_query_cve_actions=True,
        )
        self.assertEqual([item.technique_id for item in records[0].candidates], ["T2000"])
        self.assertEqual(records[0].metadata["excluded_query_cve_actions"], 1)


if __name__ == "__main__":
    unittest.main()
