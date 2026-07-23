"""Extract local and graph-wide evidence for every MulVAL vulnerability node."""

from __future__ import annotations

import csv
from typing import Any

import networkx as nx

from cve2attack.stage2.path_expander import expand_upstream_evidence, node_payload


CONTEXT_SCHEMA_VERSION = "1.0"


def parse_fact(fact: str) -> tuple[str, tuple[str, ...]]:
    """Split a simple MulVAL predicate into its name and arguments.

    MulVAL facts use single quotes for values such as paths and CVE IDs.  The
    CSV parser handles quoted commas without imposing assumptions about the
    predicate name or number of arguments.
    """
    text = fact.strip()
    open_parenthesis = text.find("(")
    if open_parenthesis < 1 or not text.endswith(")"):
        return text, ()
    predicate = text[:open_parenthesis].strip()
    body = text[open_parenthesis + 1 : -1]
    arguments = tuple(
        value.strip()
        for value in next(
            csv.reader([body], delimiter=",", quotechar="'", skipinitialspace=True)
        )
    )
    return predicate, arguments


def normalize_cve_id(value: str) -> str:
    """Normalize historical CAN identifiers so they join stage-1 CVE records."""
    identifier = value.strip().strip("'\"")
    upper = identifier.upper()
    if upper.startswith("CAN-"):
        return f"CVE-{upper[4:]}"
    if upper.startswith("CVE-"):
        return upper
    return identifier


def find_cve_nodes(graph: nx.DiGraph) -> list[int]:
    """Return all ``vulExists`` node IDs in deterministic order."""
    result: list[int] = []
    for node_id, attributes in graph.nodes(data=True):
        predicate, _arguments = parse_fact(str(attributes.get("fact") or ""))
        if predicate == "vulExists":
            result.append(node_id)
    return sorted(result)


def _argument(arguments: tuple[str, ...], index: int) -> str | None:
    return arguments[index] if len(arguments) > index else None


def _unique_node_ids(values: list[int]) -> list[int]:
    """Deduplicate node IDs while retaining deterministic ascending order."""
    return sorted(set(values))


def extract_cve_context(
    graph: nx.DiGraph,
    cve_node_id: int,
    *,
    max_graph_depth: int = 2,
) -> dict[str, Any]:
    """Extract one versioned context record from an analysis-direction graph.

    Local context contains only facts directly required by the exploit rule.
    Graph context expands every producer branch for those facts.  No branch is
    selected or ranked during extraction, and all rules/consequences are kept.
    """
    if cve_node_id not in graph:
        raise KeyError(f"Unknown CVE node: {cve_node_id}")
    cve_fact = str(graph.nodes[cve_node_id].get("fact") or "")
    predicate, arguments = parse_fact(cve_fact)
    if predicate != "vulExists":
        raise ValueError(f"Node {cve_node_id} is not a vulExists fact: {cve_fact}")

    rule_ids = sorted(
        node_id
        for node_id in graph.successors(cve_node_id)
        if graph.nodes[node_id].get("type") == "AND"
    )
    if not rule_ids:
        raise ValueError(f"CVE node {cve_node_id} has no triggered AND rule")

    consequence_ids = _unique_node_ids(
        [
            consequence_id
            for rule_id in rule_ids
            for consequence_id in graph.successors(rule_id)
            if graph.nodes[consequence_id].get("type") == "OR"
        ]
    )
    if not consequence_ids:
        raise ValueError(f"CVE node {cve_node_id} has no direct OR consequence")

    requirement_ids = _unique_node_ids(
        [
            requirement_id
            for rule_id in rule_ids
            for requirement_id in graph.predecessors(rule_id)
            if requirement_id != cve_node_id
        ]
    )
    boundary_by_node = {cve_node_id: "current_cve"}
    boundary_by_node.update({rule_id: "current_rule" for rule_id in rule_ids})
    boundary_by_node.update(
        {consequence_id: "current_consequence" for consequence_id in consequence_ids}
    )

    raw_identifier = _argument(arguments, 1) or ""
    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "cve_node_id": cve_node_id,
        "cve_id": normalize_cve_id(raw_identifier),
        "vulnerability_id_raw": raw_identifier,
        "cve_fact": cve_fact,
        "local_context": {
            "target_host": _argument(arguments, 0),
            "target_service": _argument(arguments, 2),
            "exploit_type": _argument(arguments, 3),
            "expected_impact": _argument(arguments, 4),
            "required_facts": [node_payload(graph, node_id) for node_id in requirement_ids],
            "triggered_rules": [node_payload(graph, node_id) for node_id in rule_ids],
            "direct_consequences": [
                node_payload(graph, node_id) for node_id in consequence_ids
            ],
        },
        "graph_context": {
            "max_depth": max_graph_depth,
            "upstream_requirements": [
                expand_upstream_evidence(
                    graph,
                    node_id,
                    max_depth=max_graph_depth,
                    boundary_by_node=boundary_by_node,
                )
                for node_id in requirement_ids
            ],
        },
        # Stage 1 will populate this field in the candidate-joining work package.
        "candidates": [],
    }


def extract_all_cve_contexts(
    graph: nx.DiGraph,
    *,
    max_graph_depth: int = 2,
) -> list[dict[str, Any]]:
    """Extract all vulnerability contexts in graph-node order."""
    return [
        extract_cve_context(graph, node_id, max_graph_depth=max_graph_depth)
        for node_id in find_cve_nodes(graph)
    ]
