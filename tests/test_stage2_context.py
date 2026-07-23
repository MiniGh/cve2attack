"""Regression tests for the migrated MulVAL context extractor."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from cve2attack.stage2.context_extractor import extract_all_cve_contexts
from cve2attack.stage2.graph_parser import (
    parse_xml_to_graph,
    reverse_for_analysis,
    summarize_graph,
)
from cve2attack.stage2.pipeline import run_context_extraction


FIXTURE = Path(__file__).parent / "fixtures" / "mulval" / "AttackGraph.xml"


def _facts(value: Any) -> set[str]:
    """Collect every serialized fact from a nested graph-context record."""
    result: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("fact"), str):
            result.add(value["fact"])
        for child in value.values():
            result.update(_facts(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_facts(child))
    return result


class Stage2ContextTests(unittest.TestCase):
    def setUp(self):
        self.raw_graph = parse_xml_to_graph(FIXTURE)
        self.graph = reverse_for_analysis(self.raw_graph)
        self.contexts = extract_all_cve_contexts(self.graph, max_graph_depth=2)

    def test_parses_current_mulval_fixture(self):
        self.assertEqual(
            summarize_graph(self.raw_graph),
            {
                "node_count": 44,
                "edge_count": 52,
                "type_counts": {"AND": 17, "LEAF": 19, "OR": 8},
            },
        )

    def test_extracts_local_context_without_collapsing_required_states(self):
        web_context = self.contexts[0]
        self.assertEqual(web_context["cve_id"], "CVE-2002-0392")
        self.assertEqual(web_context["local_context"]["target_host"], "webServer")
        self.assertEqual(
            {item["fact"] for item in web_context["local_context"]["required_facts"]},
            {
                "netAccess(webServer,tcp,80)",
                "networkServiceInfo(webServer,httpd,tcp,80,apache)",
            },
        )
        self.assertEqual(
            [item["fact"] for item in web_context["local_context"]["direct_consequences"]],
            ["execCode(webServer,apache)"],
        )

    def test_preserves_all_branches_and_prior_exploit_evidence(self):
        file_server_context = self.contexts[1]
        evidence_facts = _facts(file_server_context["graph_context"])

        # The old prototype selected only hacl(fileServer,fileServer,...).
        # The migrated extractor retains lateral paths and the earlier CVE.
        self.assertIn("hacl(fileServer,fileServer,rpc,100005)", evidence_facts)
        self.assertIn("hacl(webServer,fileServer,rpc,100005)", evidence_facts)
        self.assertIn("hacl(workStation,fileServer,rpc,100005)", evidence_facts)
        self.assertIn(
            "vulExists(webServer,'CAN-2002-0392',httpd,remoteExploit,privEscalation)",
            evidence_facts,
        )

    def test_file_pipeline_writes_versioned_document(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "contexts.json"
            run_context_extraction(FIXTURE, output, max_graph_depth=2)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(len(payload["contexts"]), 2)
            with self.assertRaises(FileExistsError):
                run_context_extraction(FIXTURE, output)


if __name__ == "__main__":
    unittest.main()
