"""Tests for the minimal stage-1 -> graph context -> reranking loop."""

import json
import tempfile
import unittest
from pathlib import Path

from cve2attack.data.loaders import benchmark_truth, candidate_records
from cve2attack.stage2.candidate_joiner import join_contexts_with_candidates
from cve2attack.stage2.context_extractor import extract_all_cve_contexts
from cve2attack.stage2.evaluation import evaluate_reranking
from cve2attack.stage2.graph_parser import parse_xml_to_graph, reverse_for_analysis
from cve2attack.stage2.pipeline import run_stage2_experiment
from cve2attack.stage2.reranker import rerank_joined_records
from cve2attack.schemas import CandidateRecord, TechniqueCandidate


ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures" / "stage2"
ATTACK_GRAPH = FIXTURES / "public_facing" / "AttackGraph.xml"
STAGE1_RUN = FIXTURES / "stage1_run"
BENCHMARK = FIXTURES / "benchmark"


def _context_document() -> dict:
    graph = reverse_for_analysis(parse_xml_to_graph(ATTACK_GRAPH))
    return {"schema_version": "1.0", "contexts": extract_all_cve_contexts(graph)}


class CandidateJoinerTests(unittest.TestCase):
    def test_joins_context_and_preserves_candidate_order(self):
        records = candidate_records(STAGE1_RUN)
        joined, stats = join_contexts_with_candidates(_context_document(), records)

        self.assertEqual(stats.matched, 1)
        self.assertEqual(stats.missing_candidates, ())
        self.assertEqual(joined[0]["cve_id"], "CVE-2023-20887")
        self.assertEqual(
            [candidate["technique_id"] for candidate in joined[0]["candidates"]],
            ["T1072", "T1210", "T1127", "T1505", "T1190", "T1059"],
        )

    def test_reports_unresolved_and_missing_context_ids(self):
        document = {
            "contexts": [
                {"cve_id": "vulID", "candidates": []},
                {"cve_id": "CVE-2024-0001", "candidates": []},
            ]
        }
        joined, stats = join_contexts_with_candidates(document, [])
        self.assertEqual(joined, [])
        self.assertEqual(stats.unresolved_context_ids, ("vulID",))
        self.assertEqual(stats.missing_candidates, ("CVE-2024-0001",))

    def test_rejects_duplicate_candidate_records(self):
        record = CandidateRecord(
            cve_id="CVE-2024-0001",
            candidates=(TechniqueCandidate("T1190"),),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate stage-1 candidate"):
            join_contexts_with_candidates({"contexts": []}, [record, record])


class TopologyRerankerTests(unittest.TestCase):
    def test_public_facing_rule_promotes_t1190_without_changing_candidates(self):
        joined, _stats = join_contexts_with_candidates(
            _context_document(), candidate_records(STAGE1_RUN)
        )
        reranked = rerank_joined_records(joined)

        original_ids = [candidate["technique_id"] for candidate in joined[0]["candidates"]]
        reranked_ids = [candidate["technique_id"] for candidate in reranked[0]["candidates"]]
        self.assertEqual(reranked_ids[0], "T1190")
        self.assertEqual(set(original_ids), set(reranked_ids))
        self.assertEqual(
            [rule["rule_id"] for rule in reranked[0]["reranker"]["detected_rules"]],
            ["public_facing_service"],
        )

    def test_evaluation_reports_rank_gain_on_same_candidate_set(self):
        joined, _stats = join_contexts_with_candidates(
            _context_document(), candidate_records(STAGE1_RUN)
        )
        reranked = rerank_joined_records(joined)
        metrics = evaluate_reranking(joined, reranked, benchmark_truth(BENCHMARK))

        self.assertTrue(metrics["candidate_sets_preserved"])
        self.assertEqual(metrics["cases"][0]["best_original_rank"], 5)
        self.assertEqual(metrics["cases"][0]["best_reranked_rank"], 1)
        self.assertEqual(metrics["wins"], 1)

    def test_lateral_rule_prioritizes_remote_service_exploitation(self):
        record = {
            "cve_id": "CVE-2024-0002",
            "local_context": {
                "target_host": "serverB",
                "direct_consequences": [{"fact": "execCode(serverB,root)"}],
            },
            "graph_context": {
                "upstream_requirements": [
                    {"fact": "hacl(serverA,serverB,tcp,445)"},
                    {"fact": "execCode(serverA,root)"},
                ]
            },
            "candidates": [
                {"technique_id": "T1000", "sources": ["fixture"]},
                {"technique_id": "T1210", "sources": ["fixture"]},
            ],
        }
        reranked = rerank_joined_records([record])[0]
        self.assertEqual(reranked["candidates"][0]["technique_id"], "T1210")
        self.assertEqual(
            [rule["rule_id"] for rule in reranked["reranker"]["detected_rules"]],
            ["lateral_remote_service"],
        )

    def test_local_privilege_rule_requires_prior_non_root_execution(self):
        record = {
            "cve_id": "CVE-2024-0003",
            "local_context": {
                "target_host": "workstation",
                "direct_consequences": [{"fact": "execCode(workstation,root)"}],
            },
            "graph_context": {
                "upstream_requirements": [{"fact": "execCode(workstation,user)"}]
            },
            "candidates": [
                {"technique_id": "T1000", "sources": ["fixture"]},
                {"technique_id": "T1068", "sources": ["fixture"]},
            ],
        }
        reranked = rerank_joined_records([record])[0]
        self.assertEqual(reranked["candidates"][0]["technique_id"], "T1068")
        self.assertEqual(
            [rule["rule_id"] for rule in reranked["reranker"]["detected_rules"]],
            ["local_privilege_transition"],
        )

    def test_local_privilege_rule_uses_stable_tactic_priority_group(self):
        record = {
            "cve_id": "CVE-2024-0005",
            "local_context": {
                "target_host": "workstation",
                "direct_consequences": [{"fact": "execCode(workstation,root)"}],
            },
            "graph_context": {
                "upstream_requirements": [{"fact": "execCode(workstation,user)"}]
            },
            "candidates": [
                {"technique_id": "T1059", "metadata": {"tactics": ["execution"]}},
                {
                    "technique_id": "T1548",
                    "metadata": {"tactics": ["privilege-escalation", "defense-evasion"]},
                },
                {"technique_id": "T1068", "metadata": {"tactics": ["privilege-escalation"]}},
                {"technique_id": "T1134", "metadata": {"tactics": ["privilege-escalation"]}},
            ],
        }
        reranked = rerank_joined_records([record])[0]
        self.assertEqual(
            [candidate["technique_id"] for candidate in reranked["candidates"]],
            ["T1548", "T1068", "T1134", "T1059"],
        )
        self.assertTrue(reranked["candidates"][0]["metadata"]["stage2"]["topology_match"])
        self.assertEqual(
            reranked["reranker"]["detected_rules"][0]["match_scope"],
            "tactic",
        )

    def test_target_semantic_fields_alone_do_not_change_ranking(self):
        record = {
            "cve_id": "CVE-2024-0004",
            "local_context": {
                "target_host": "server",
                "exploit_type": "remoteExploit",
                "expected_impact": "privEscalation",
                "direct_consequences": [],
            },
            "graph_context": {"upstream_requirements": []},
            "candidates": [
                {"technique_id": "T1000", "sources": ["fixture"]},
                {"technique_id": "T1190", "sources": ["fixture"]},
                {"technique_id": "T1068", "sources": ["fixture"]},
            ],
        }
        reranked = rerank_joined_records([record])[0]
        self.assertEqual(
            [candidate["technique_id"] for candidate in reranked["candidates"]],
            ["T1000", "T1190", "T1068"],
        )
        self.assertEqual(reranked["reranker"]["detected_rules"], [])

    def test_end_to_end_pipeline_writes_a_complete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = run_stage2_experiment(
                stage1_run=STAGE1_RUN,
                attack_graph_path=ATTACK_GRAPH,
                benchmark_dir=BENCHMARK,
                output_root=directory,
                run_id="public-facing-smoke",
                project_root=ROOT.parent,
                scenario_kind="synthetic_public_facing_smoke",
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(metrics["reranked"]["top1"], 1.0)
            self.assertTrue((output / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
