"""Tests for normalized public-scenario to MulVAL graph conversion."""

import hashlib
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
ATTACKMATE_DIR = (
    ROOT / "data" / "stage2_scenarios" / "attackmate" / "pwnkit_cve_2021_4034"
)
MANTIS_DIR = ROOT / "data" / "stage2_scenarios" / "mantis"
CASE_DIR = MANTIS_DIR / "zerologon"
SCENARIO = CASE_DIR / "scenario.yaml"
CHECKED_GRAPH = CASE_DIR / "AttackGraph.xml"
STAGE1_SNAPSHOT = CASE_DIR / "stage1_snapshot"
TATSU_DIR = MANTIS_DIR / "tatsu_rce"
SUDO_DIR = MANTIS_DIR / "sudo_cve_2021_3156"
BENCHMARK = ROOT / "data" / "benchmarks" / "stage2_mantis_scenarios"
EXTENDED_LPE_DIR = ROOT / "data" / "stage2_scenarios" / "extended_lpe"
EXTENDED_LPE_CASES = {
    "cve_2020_0787": ("CVE-2020-0787", "bits"),
    "cve_2021_40449": ("CVE-2021-40449", "win32k"),
    "cve_2022_21999": ("CVE-2022-21999", "spooler"),
    "cve_2022_26904": ("CVE-2022-26904", "profsvc"),
    "cve_2010_3856": ("CVE-2010-3856", "glibc_ld_audit"),
}
EXTENDED_LPE_GRAPH_SHA256 = {
    "cve_2020_0787": "a713a989dbea94741f9bf6d6e9fd86dd66b8606048b0d40c4cbf6733e6eb78b9",
    "cve_2021_40449": "dfe4eae077c177529d77488505aba8da5d399ed9b9b5fe013a288565120f7b36",
    "cve_2022_21999": "21742c98e2da07075b89fb1dc462caa1fcce6dad0fbd97071ffc64ef437e0237",
    "cve_2022_26904": "691554323a8e6644605698ec5c07d5b489245ea78ba6fd953214131c0a89fbdf",
    "cve_2010_3856": "a30f11a363c8070a49cfb878bcc5a60922d3a62cf185bbf82073cfbbcdf356aa",
}


class ScenarioGraphTests(unittest.TestCase):
    def test_checked_graph_is_reproducible_from_scenario(self):
        for case_dir in (
            CASE_DIR,
            TATSU_DIR,
            SUDO_DIR,
            ATTACKMATE_DIR,
            *(EXTENDED_LPE_DIR / name for name in sorted(EXTENDED_LPE_CASES)),
        ):
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
        self.assertEqual(rules[0]["match_scope"], "technique")
        self.assertEqual(rules[0]["match_values"], ["T1210"])
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
        self.assertEqual(rules[0]["match_scope"], "technique")
        self.assertEqual(rules[0]["match_values"], ["T1190"])
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
        self.assertEqual(rules[0]["match_scope"], "tactic")
        self.assertEqual(rules[0]["match_values"], ["privilege-escalation"])
        self.assertEqual(rules[0]["fallback_technique_ids"], ["T1068"])
        self.assertIn("execCode(TARGET,user)", rules[0]["evidence"])
        self.assertIn("execCode(TARGET,root)", rules[0]["evidence"])

    def test_pwnkit_graph_preserves_independent_local_privilege_evidence(self):
        graph = reverse_for_analysis(
            parse_xml_to_graph(ATTACKMATE_DIR / "AttackGraph.xml")
        )
        contexts = extract_all_cve_contexts(graph)
        self.assertEqual(len(contexts), 1)
        record = contexts[0]
        self.assertEqual(record["cve_id"], "CVE-2021-4034")
        rules = detect_topology_rules(record)
        self.assertEqual([rule["rule_id"] for rule in rules], ["local_privilege_transition"])
        self.assertEqual(rules[0]["match_scope"], "tactic")
        self.assertEqual(rules[0]["match_values"], ["privilege-escalation"])
        self.assertIn("execCode(TARGET,user)", rules[0]["evidence"])
        self.assertIn("execCode(TARGET,root)", rules[0]["evidence"])

    def test_evaluation_label_cannot_change_generated_graph(self):
        for case_dir in (
            CASE_DIR,
            TATSU_DIR,
            SUDO_DIR,
            ATTACKMATE_DIR,
            *(EXTENDED_LPE_DIR / name for name in sorted(EXTENDED_LPE_CASES)),
        ):
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

    def test_sudo_tactic_guard_preserves_specific_stage1_top1(self):
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
            self.assertEqual(case["best_reranked_rank"], 1)
            self.assertEqual(case["original_top1"], "T1548")
            self.assertEqual(case["reranked_top1"], "T1548")
            self.assertEqual(case["outcome"], "unchanged")
            self.assertEqual(metrics["losses"], 0)
            self.assertEqual(metrics["ties"], 1)
            self.assertTrue(metrics["candidate_sets_preserved"])


class ExtendedLocalPrivilegeGraphTests(unittest.TestCase):
    """Frozen label-blind graphs for the extended local privilege escalation cohort."""

    def test_frozen_graph_hashes_are_stable(self):
        for name, expected in sorted(EXTENDED_LPE_GRAPH_SHA256.items()):
            with self.subTest(case=name):
                digest = hashlib.sha256(
                    (EXTENDED_LPE_DIR / name / "AttackGraph.xml").read_bytes()
                ).hexdigest()
                self.assertEqual(digest, expected)

    def test_each_graph_exposes_one_cve_and_local_privilege_evidence(self):
        for name, (cve_id, service) in sorted(EXTENDED_LPE_CASES.items()):
            with self.subTest(case=name):
                graph = reverse_for_analysis(
                    parse_xml_to_graph(EXTENDED_LPE_DIR / name / "AttackGraph.xml")
                )
                contexts = extract_all_cve_contexts(graph)
                self.assertEqual(len(contexts), 1)
                record = contexts[0]
                self.assertEqual(record["cve_id"], cve_id)
                self.assertEqual(record["local_context"]["target_host"], "TARGET")
                self.assertEqual(record["local_context"]["target_service"], service)
                self.assertEqual(record["local_context"]["exploit_type"], "localExploit")
                self.assertEqual(
                    record["local_context"]["expected_impact"], "privilegeEscalation"
                )

                rules = detect_topology_rules(record)
                self.assertEqual(
                    [rule["rule_id"] for rule in rules], ["local_privilege_transition"]
                )
                self.assertEqual(rules[0]["match_scope"], "tactic")
                self.assertEqual(rules[0]["match_values"], ["privilege-escalation"])
                self.assertEqual(rules[0]["fallback_technique_ids"], ["T1068"])
                self.assertIn("execCode(TARGET,user)", rules[0]["evidence"])
                self.assertIn("execCode(TARGET,root)", rules[0]["evidence"])

    def test_graphs_carry_no_network_entry_or_lateral_evidence(self):
        # A local privilege escalation graph must not smuggle in internet entry or
        # cross-host reachability, which would trigger a different topology rule.
        for name in sorted(EXTENDED_LPE_CASES):
            with self.subTest(case=name):
                text = (EXTENDED_LPE_DIR / name / "AttackGraph.xml").read_text(
                    encoding="utf-8"
                )
                self.assertNotIn("attackerLocated", text)
                self.assertNotIn("hacl(", text)
                self.assertNotIn("netAccess(", text)
                self.assertNotIn("networkServiceInfo(", text)

    def test_expected_technique_never_appears_in_graph(self):
        for name in sorted(EXTENDED_LPE_CASES):
            with self.subTest(case=name):
                scenario = load_scenario(EXTENDED_LPE_DIR / name / "scenario.yaml")
                self.assertEqual(scenario["evaluation"]["expected_techniques"], ["T1068"])
                self.assertNotIn("T1068", render_attack_graph_xml(scenario))


if __name__ == "__main__":
    unittest.main()
