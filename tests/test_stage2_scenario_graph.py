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
CASE_DIR = ROOT / "data" / "stage2_scenarios" / "mantis" / "zerologon"
SCENARIO = CASE_DIR / "scenario.yaml"
CHECKED_GRAPH = CASE_DIR / "AttackGraph.xml"
STAGE1_SNAPSHOT = CASE_DIR / "stage1_snapshot"
BENCHMARK = ROOT / "data" / "benchmarks" / "stage2_mantis_scenarios"


class ScenarioGraphTests(unittest.TestCase):
    def test_checked_graph_is_reproducible_from_scenario(self):
        scenario = load_scenario(SCENARIO)
        self.assertEqual(render_attack_graph_xml(scenario), CHECKED_GRAPH.read_text(encoding="utf-8"))

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

    def test_evaluation_label_cannot_change_generated_graph(self):
        scenario = load_scenario(SCENARIO)
        original = render_attack_graph_xml(scenario)
        changed_label = deepcopy(scenario)
        changed_label["evaluation"]["expected_techniques"] = ["T1190"]
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


if __name__ == "__main__":
    unittest.main()
