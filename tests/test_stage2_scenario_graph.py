"""Tests for normalized public-scenario to MulVAL graph conversion."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from cve2attack.stage2.context_extractor import extract_all_cve_contexts
from cve2attack.stage2.graph_parser import parse_xml_to_graph, reverse_for_analysis
from cve2attack.stage2.pipeline import run_stage2_experiment
from cve2attack.stage2.reranker import detect_topology_rules
from cve2attack.stage2.scenario_graph import (
    build_attack_graph_from_scenario,
    load_scenario,
    render_attack_graph_xml,
)


ROOT = Path(__file__).parent.parent
MANTIS_DIR = ROOT / "data" / "stage2_scenarios" / "mantis"
CASE_DIR = MANTIS_DIR / "zerologon"
SCENARIO = CASE_DIR / "scenario.yaml"
CHECKED_GRAPH = CASE_DIR / "AttackGraph.xml"
STAGE1_SNAPSHOT = CASE_DIR / "stage1_snapshot"
TATSU_DIR = MANTIS_DIR / "tatsu_rce"
SUDO_DIR = MANTIS_DIR / "sudo_cve_2021_3156"
BENCHMARK = ROOT / "data" / "benchmarks" / "stage2_mantis_scenarios"


class ScenarioGraphTests(unittest.TestCase):
    def test_checked_graph_is_reproducible_from_scenario(self):
        for case_dir in (CASE_DIR, TATSU_DIR, SUDO_DIR):
            with self.subTest(case=case_dir.name):
                scenario = load_scenario(case_dir / "scenario.yaml")
                self.assertEqual(
                    render_attack_graph_xml(scenario),
                    (case_dir / "AttackGraph.xml").read_text(encoding="utf-8"),
                )

    def test_zerologon_graph_preserves_lateral_trace_evidence(self):
        graph = reverse_for_analysis(parse_xml_to_graph(CHECKED_GRAPH))
        contexts = extract_all_cve_contexts(graph)
        self.assertEqual(len(contexts), 1)
        record = contexts[0]
        self.assertEqual(record["cve_id"], "CVE-2020-1472")
        self.assertEqual(record["local_context"]["target_host"], "dcserver")
        rules = detect_topology_rules(record)
        self.assertEqual([rule["rule_id"] for rule in rules], ["lateral_remote_service"])
        self.assertEqual(rules[0]["technique_id"], "T1210")
        self.assertIn("execCode(TARGET,user)", rules[0]["evidence"])
        self.assertIn("hacl(TARGET,dcserver,tcp,445)", rules[0]["evidence"])

    def test_tatsu_graph_preserves_public_facing_trace_evidence(self):
        graph = reverse_for_analysis(parse_xml_to_graph(TATSU_DIR / "AttackGraph.xml"))
        contexts = extract_all_cve_contexts(graph)
        self.assertEqual(len(contexts), 1)
        record = contexts[0]
        self.assertEqual(record["cve_id"], "CVE-2021-25094")
        rules = detect_topology_rules(record)
        self.assertEqual([rule["rule_id"] for rule in rules], ["public_facing_service"])
        self.assertEqual(rules[0]["technique_id"], "T1190")
        self.assertIn("attackerLocated(internet)", rules[0]["evidence"])
        self.assertIn("networkServiceInfo(TARGET,wordpress_tatsu,tcp,80,www-data)", rules[0]["evidence"])

    def test_sudo_graph_preserves_local_privilege_trace_evidence(self):
        graph = reverse_for_analysis(parse_xml_to_graph(SUDO_DIR / "AttackGraph.xml"))
        contexts = extract_all_cve_contexts(graph)
        self.assertEqual(len(contexts), 1)
        record = contexts[0]
        self.assertEqual(record["cve_id"], "CVE-2021-3156")
        rules = detect_topology_rules(record)
        self.assertEqual([rule["rule_id"] for rule in rules], ["local_privilege_transition"])
        self.assertEqual(rules[0]["technique_id"], "T1068")
        self.assertIn("execCode(TARGET,user)", rules[0]["evidence"])
        self.assertIn("execCode(TARGET,root)", rules[0]["evidence"])

    def test_evaluation_label_cannot_change_generated_graph(self):
        for case_dir in (CASE_DIR, TATSU_DIR, SUDO_DIR):
            with self.subTest(case=case_dir.name):
                scenario = load_scenario(case_dir / "scenario.yaml")
                original = render_attack_graph_xml(scenario)
                changed_label = deepcopy(scenario)
                changed_label["evaluation"]["expected_techniques"] = ["T0000"]
                self.assertEqual(render_attack_graph_xml(changed_label), original)

    def test_builder_refuses_to_silently_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "AttackGraph.xml"
            build_attack_graph_from_scenario(SCENARIO, output)
            with self.assertRaises(FileExistsError):
                build_attack_graph_from_scenario(SCENARIO, output)

    def test_trace_derived_case_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            output = run_stage2_experiment(
                stage1_run=STAGE1_SNAPSHOT,
                attack_graph_path=CHECKED_GRAPH,
                benchmark_dir=BENCHMARK,
                output_root=directory,
                run_id="mantis-zerologon-test",
                project_root=ROOT,
                scenario_kind="trace_derived_mantis_lateral_movement",
            )
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertEqual(metrics["cases"][0]["best_original_rank"], 2)
            self.assertEqual(metrics["cases"][0]["best_reranked_rank"], 1)
            self.assertEqual(metrics["wins"], 1)
            self.assertIn("public scenario execution trace", report)

    def test_tatsu_missing_stage1_candidate_is_reported_as_unrecoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = run_stage2_experiment(
                stage1_run=TATSU_DIR / "stage1_snapshot",
                attack_graph_path=TATSU_DIR / "AttackGraph.xml",
                benchmark_dir=BENCHMARK,
                output_root=directory,
                run_id="mantis-tatsu-test",
                project_root=ROOT,
                scenario_kind="trace_derived_public_facing",
            )
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            case = metrics["cases"][0]
            self.assertIsNone(case["best_original_rank"])
            self.assertIsNone(case["best_reranked_rank"])
            self.assertEqual(case["outcome"], "unrecoverable")
            self.assertEqual(metrics["unrecoverable"], 1)
            self.assertTrue(metrics["candidate_sets_preserved"])

    def test_sudo_rule_degradation_is_kept_as_a_regression_case(self):
        with tempfile.TemporaryDirectory() as directory:
            output = run_stage2_experiment(
                stage1_run=SUDO_DIR / "stage1_snapshot",
                attack_graph_path=SUDO_DIR / "AttackGraph.xml",
                benchmark_dir=BENCHMARK,
                output_root=directory,
                run_id="mantis-sudo-test",
                project_root=ROOT,
                scenario_kind="trace_derived_local_privilege",
            )
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            case = metrics["cases"][0]
            self.assertEqual(case["best_original_rank"], 1)
            self.assertEqual(case["best_reranked_rank"], 2)
            self.assertEqual(case["original_top1"], "T1548")
            self.assertEqual(case["reranked_top1"], "T1068")
            self.assertEqual(case["outcome"], "degraded")
            self.assertEqual(metrics["losses"], 1)
            self.assertTrue(metrics["candidate_sets_preserved"])


if __name__ == "__main__":
    unittest.main()
