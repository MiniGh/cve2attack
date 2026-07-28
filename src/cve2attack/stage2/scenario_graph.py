"""Convert a normalized, trace-derived scenario into MulVAL-compatible XML.

The public scenario packages used by stage 2 contain network topology and
ordered attack reports, but they are not MulVAL ``AttackGraph.xml`` files.
This module provides the narrow, deterministic bridge between those formats.

Expected ATT&CK labels live under the scenario's ``evaluation`` section.  The
graph renderer deliberately never reads that section: labels are reserved for
after-the-fact evaluation and cannot influence the generated topology.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from cve2attack.stage2.context_extractor import (
    extract_all_cve_contexts,
    normalize_cve_id,
    parse_fact,
)
from cve2attack.stage2.graph_parser import parse_xml_to_graph, reverse_for_analysis


SCENARIO_SCHEMA_VERSION = "1.0"
_SAFE_SCENARIO_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_CVE_ID = re.compile(r"^CVE-\d{4}-\d{4,}$")
_TECHNIQUE_ID = re.compile(r"^T\d{4}(?:\.\d{3})?$")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a YAML object")
    return value


def _nonempty(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty YAML list")
    result = [_nonempty(item, f"{field}[]") for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _validate_fact(fact: str, field: str) -> None:
    predicate, arguments = parse_fact(fact)
    if not predicate or not arguments:
        raise ValueError(f"{field} is not a predicate with arguments: {fact}")


def validate_scenario(scenario: Mapping[str, Any]) -> None:
    """Validate the versioned scenario contract before rendering a graph."""
    if str(scenario.get("schema_version")) != SCENARIO_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported scenario schema_version: {scenario.get('schema_version')!r}"
        )

    scenario_id = _nonempty(scenario.get("scenario_id"), "scenario_id")
    if not _SAFE_SCENARIO_ID.fullmatch(scenario_id):
        raise ValueError("scenario_id may contain only lowercase letters, digits, '_' and '-'")

    cve_id = normalize_cve_id(_nonempty(scenario.get("cve_id"), "cve_id"))
    if not _CVE_ID.fullmatch(cve_id):
        raise ValueError(f"cve_id must be a canonical CVE identifier: {cve_id}")

    source = _mapping(scenario.get("source"), "source")
    _nonempty(source.get("provider"), "source.provider")
    _nonempty(source.get("dataset_id"), "source.dataset_id")

    context = _mapping(scenario.get("context"), "context")
    initial_facts = _string_list(context.get("initial_facts"), "context.initial_facts")
    for fact in initial_facts:
        _validate_fact(fact, "context.initial_facts[]")

    vulnerability = _mapping(context.get("vulnerability"), "context.vulnerability")
    vulnerability_host = _nonempty(
        vulnerability.get("host"), "context.vulnerability.host"
    )
    vulnerability_service = _nonempty(
        vulnerability.get("service"), "context.vulnerability.service"
    )
    _nonempty(vulnerability.get("exploit_type"), "context.vulnerability.exploit_type")
    _nonempty(
        vulnerability.get("expected_impact"), "context.vulnerability.expected_impact"
    )

    consequences = _string_list(context.get("consequences"), "context.consequences")
    for fact in consequences:
        _validate_fact(fact, "context.consequences[]")

    network_access = context.get("network_access")
    target_service = context.get("target_service")
    if network_access is not None:
        network = _mapping(network_access, "context.network_access")
        source_host = _nonempty(
            network.get("source_host"), "context.network_access.source_host"
        )
        target_host = _nonempty(
            network.get("target_host"), "context.network_access.target_host"
        )
        protocol = _nonempty(network.get("protocol"), "context.network_access.protocol")
        port = _nonempty(network.get("port"), "context.network_access.port")
        service = _mapping(target_service, "context.target_service")
        if _nonempty(service.get("host"), "context.target_service.host") != target_host:
            raise ValueError("target_service.host must match network_access.target_host")
        if vulnerability_host != target_host:
            raise ValueError("vulnerability.host must match network_access.target_host")
        if _nonempty(service.get("name"), "context.target_service.name") != vulnerability_service:
            raise ValueError("target_service.name must match vulnerability.service")
        if _nonempty(service.get("protocol"), "context.target_service.protocol") != protocol:
            raise ValueError("target_service.protocol must match network_access.protocol")
        if _nonempty(service.get("port"), "context.target_service.port") != port:
            raise ValueError("target_service.port must match network_access.port")
        _nonempty(service.get("account"), "context.target_service.account")

        # A trace-derived network edge must be anchored in an observed state:
        # attackerLocated(internet) for public access or execCode(source,...) for
        # lateral movement.  This prevents an ungrounded hacl edge from silently
        # becoming sufficient evidence.
        parsed_initial = [parse_fact(fact) for fact in initial_facts]
        has_source_basis = any(
            (predicate == "attackerLocated" and source_host == "internet" and arguments[0] == "internet")
            or (predicate == "execCode" and arguments[0] == source_host)
            for predicate, arguments in parsed_initial
            if arguments
        )
        if not has_source_basis:
            raise ValueError(
                "network_access requires attackerLocated(internet) or execCode(source_host,...)"
            )
    elif target_service is not None:
        raise ValueError("target_service requires context.network_access")

    evaluation = _mapping(scenario.get("evaluation"), "evaluation")
    if evaluation.get("evaluation_only") is not True:
        raise ValueError("evaluation.evaluation_only must be true")
    labels = _string_list(
        evaluation.get("expected_techniques"), "evaluation.expected_techniques"
    )
    for technique_id in labels:
        if not _TECHNIQUE_ID.fullmatch(technique_id):
            raise ValueError(f"Invalid ATT&CK Technique ID: {technique_id}")
    _nonempty(evaluation.get("label_source"), "evaluation.label_source")


def load_scenario(path: str | Path) -> dict[str, Any]:
    """Load and validate one scenario YAML document."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Stage-2 scenario does not exist: {source}")
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Stage-2 scenario must be a YAML object: {source}")
    validate_scenario(value)
    return value


class _GraphBuilder:
    """Small deterministic builder for MulVAL's effect-to-requirement XML."""

    def __init__(self) -> None:
        self.vertices: list[dict[str, Any]] = []
        self.arcs: list[tuple[int, int]] = []

    def vertex(self, fact: str, node_type: str, metric: int) -> int:
        node_id = len(self.vertices) + 1
        self.vertices.append(
            {"id": node_id, "fact": fact, "type": node_type, "metric": metric}
        )
        return node_id

    def arc(self, source: int, target: int) -> None:
        self.arcs.append((source, target))

    def xml(self) -> str:
        root = ET.Element("attack_graph")
        arcs = ET.SubElement(root, "arcs")
        for source, target in self.arcs:
            arc = ET.SubElement(arcs, "arc")
            ET.SubElement(arc, "src").text = str(source)
            ET.SubElement(arc, "dst").text = str(target)

        vertices = ET.SubElement(root, "vertices")
        for item in self.vertices:
            vertex = ET.SubElement(vertices, "vertex")
            ET.SubElement(vertex, "id").text = str(item["id"])
            ET.SubElement(vertex, "fact").text = str(item["fact"])
            ET.SubElement(vertex, "metric").text = str(item["metric"])
            ET.SubElement(vertex, "type").text = str(item["type"])
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode") + "\n"


def render_attack_graph_xml(scenario: Mapping[str, Any]) -> str:
    """Render XML from context fields without reading evaluation labels."""
    validate_scenario(scenario)
    # Copy only the non-evaluation section to make the leakage boundary
    # explicit in code as well as documentation.
    graph_input = deepcopy(_mapping(scenario.get("context"), "context"))
    cve_id = normalize_cve_id(str(scenario["cve_id"]))
    builder = _GraphBuilder()

    consequence_ids = [
        builder.vertex(fact, "OR", 0) for fact in graph_input["consequences"]
    ]
    exploit_rule_id = builder.vertex(
        str(graph_input.get("exploit_rule") or "RULE 1 (trace-derived vulnerability exploitation)"),
        "AND",
        0,
    )
    for consequence_id in consequence_ids:
        builder.arc(consequence_id, exploit_rule_id)

    initial_ids = [
        builder.vertex(fact, "LEAF", 1) for fact in graph_input["initial_facts"]
    ]
    exploit_requirement_ids: list[int] = []

    network = graph_input.get("network_access")
    service = graph_input.get("target_service")
    if isinstance(network, Mapping):
        source_host = str(network["source_host"])
        target_host = str(network["target_host"])
        protocol = str(network["protocol"])
        port = str(network["port"])
        net_access_id = builder.vertex(
            f"netAccess({target_host},{protocol},{port})", "OR", 0
        )
        access_rule_id = builder.vertex(
            str(graph_input.get("access_rule") or "RULE 2 (trace-derived network access)"),
            "AND",
            0,
        )
        hacl_id = builder.vertex(
            f"hacl({source_host},{target_host},{protocol},{port})", "LEAF", 1
        )
        builder.arc(net_access_id, access_rule_id)
        builder.arc(access_rule_id, hacl_id)
        for initial_id in initial_ids:
            builder.arc(access_rule_id, initial_id)
        exploit_requirement_ids.append(net_access_id)

        assert isinstance(service, Mapping)  # guaranteed by validate_scenario
        service_id = builder.vertex(
            "networkServiceInfo("
            f"{service['host']},{service['name']},{service['protocol']},"
            f"{service['port']},{service['account']})",
            "LEAF",
            1,
        )
        exploit_requirement_ids.append(service_id)
    else:
        exploit_requirement_ids.extend(initial_ids)

    vulnerability = _mapping(graph_input["vulnerability"], "context.vulnerability")
    vulnerability_id = builder.vertex(
        "vulExists("
        f"{vulnerability['host']},'{cve_id}',{vulnerability['service']},"
        f"{vulnerability['exploit_type']},{vulnerability['expected_impact']})",
        "LEAF",
        1,
    )
    exploit_requirement_ids.append(vulnerability_id)
    for requirement_id in exploit_requirement_ids:
        builder.arc(exploit_rule_id, requirement_id)

    return builder.xml()


def build_attack_graph_from_scenario(
    scenario_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Build, validate and atomically write one MulVAL-compatible graph."""
    source = Path(scenario_path)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --force to replace it")

    print(f"[1/4] Loading normalized stage-2 scenario: {source}")
    scenario = load_scenario(source)
    print(
        f"      scenario={scenario['scenario_id']} cve={scenario['cve_id']} "
        f"provider={scenario['source']['provider']}"
    )

    print("[2/4] Rendering label-isolated MulVAL-compatible graph")
    xml_text = render_attack_graph_xml(scenario)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(xml_text, encoding="utf-8")

    print("[3/4] Validating graph parser and extracted CVE context")
    try:
        raw_graph = parse_xml_to_graph(temporary)
        contexts = extract_all_cve_contexts(reverse_for_analysis(raw_graph))
        if [record["cve_id"] for record in contexts] != [scenario["cve_id"]]:
            raise ValueError("Generated graph CVE does not match scenario.cve_id")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(
        f"[4/4] Wrote graph: {output} "
        f"nodes={raw_graph.number_of_nodes()} edges={raw_graph.number_of_edges()}"
    )
    return output
