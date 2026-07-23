"""Expand every upstream proof branch for a required MulVAL state.

The old prototype selected one producer rule for an OR node.  That discarded
lateral-movement evidence and could choose a circular self-access branch.  The
stage-2 contract instead records every producer rule and marks cycles or the
current exploit's own nodes as boundaries.  A later reranker can then decide
which evidence matters without losing it during extraction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx


def node_payload(graph: nx.DiGraph, node_id: int) -> dict[str, Any]:
    """Serialize the stable, human-readable attributes of one graph node."""
    attributes = graph.nodes[node_id]
    return {
        "node_id": node_id,
        "fact": str(attributes.get("fact") or ""),
        "type": str(attributes.get("type") or ""),
        "metric": attributes.get("metric", 0),
    }


def expand_upstream_evidence(
    graph: nx.DiGraph,
    node_id: int,
    *,
    max_depth: int,
    boundary_by_node: Mapping[int, str] | None = None,
    _path: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Return all upstream alternatives for ``node_id`` up to ``max_depth``.

    ``graph`` must use requirement -> rule -> effect direction.  OR nodes are
    states that may be produced by several AND rules, while an AND node lists
    requirements that must all hold.  ``boundary_by_node`` identifies nodes
    belonging to the exploit currently being analysed; they remain visible in
    the JSON but are not expanded, which prevents circular explanations.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if node_id not in graph:
        raise KeyError(f"Unknown graph node: {node_id}")

    payload = node_payload(graph, node_id)
    boundaries = boundary_by_node or {}

    if node_id in boundaries:
        payload["boundary"] = boundaries[node_id]
        return payload
    if node_id in _path:
        payload["cycle"] = True
        return payload

    node_type = payload["type"]
    upstream = sorted(graph.predecessors(node_id))
    if node_type == "LEAF" or not upstream:
        return payload
    if max_depth == 0:
        payload["truncated"] = True
        return payload

    next_path = _path + (node_id,)
    if node_type == "OR":
        producer_rules = [
            predecessor
            for predecessor in upstream
            if graph.nodes[predecessor].get("type") == "AND"
        ]
        alternatives: list[dict[str, Any]] = []
        for rule_id in producer_rules:
            requirements = [
                expand_upstream_evidence(
                    graph,
                    requirement_id,
                    max_depth=max_depth - 1,
                    boundary_by_node=boundaries,
                    _path=next_path + (rule_id,),
                )
                for requirement_id in sorted(graph.predecessors(rule_id))
            ]
            alternatives.append(
                {"rule": node_payload(graph, rule_id), "requirements": requirements}
            )
        if alternatives:
            payload["producer_rules"] = alternatives
        return payload

    if node_type == "AND":
        payload["requirements"] = [
            expand_upstream_evidence(
                graph,
                requirement_id,
                max_depth=max_depth - 1,
                boundary_by_node=boundaries,
                _path=next_path,
            )
            for requirement_id in upstream
        ]
    return payload
