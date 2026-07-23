"""Parse a MulVAL ``AttackGraph.xml`` file into a NetworkX graph.

MulVAL's XML edges point from an effect to the rule and then to the rule's
requirements.  Context extraction is easier in the opposite direction:
requirement -> rule -> effect.  The two directions are kept as explicit
operations so callers cannot accidentally analyse a graph in the wrong
direction.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx


def _required_text(element: ET.Element, field: str, *, owner: str) -> str:
    """Read one required XML child and report a useful validation error."""
    value = element.findtext(field)
    if value is None or not value.strip():
        raise ValueError(f"{owner} is missing a non-empty <{field}> value")
    return value.strip()


def _parse_metric(raw_value: str) -> int | float:
    """Preserve integer metrics while accepting MulVAL files with decimals."""
    value = raw_value.strip() or "0"
    try:
        return int(value)
    except ValueError:
        return float(value)


def parse_xml_to_graph(xml_path: str | Path) -> nx.DiGraph:
    """Return the graph in the original MulVAL XML edge direction.

    Every node contains ``fact``, ``type`` and ``metric`` attributes.  Node
    identifiers are integers.  The function validates missing sections,
    malformed identifiers and arcs that reference undefined vertices.
    """
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"MulVAL attack graph does not exist: {path}")

    root = ET.parse(path).getroot()
    vertices = root.find("vertices")
    arcs = root.find("arcs")
    if vertices is None:
        raise ValueError("MulVAL XML is missing the <vertices> section")
    if arcs is None:
        raise ValueError("MulVAL XML is missing the <arcs> section")

    graph = nx.DiGraph()
    for index, vertex in enumerate(vertices.findall("vertex"), start=1):
        owner = f"vertex #{index}"
        node_id = int(_required_text(vertex, "id", owner=owner))
        if node_id in graph:
            raise ValueError(f"MulVAL XML contains duplicate vertex id {node_id}")
        graph.add_node(
            node_id,
            fact=_required_text(vertex, "fact", owner=owner),
            type=_required_text(vertex, "type", owner=owner).upper(),
            metric=_parse_metric(vertex.findtext("metric", "0")),
        )

    for index, arc in enumerate(arcs.findall("arc"), start=1):
        owner = f"arc #{index}"
        source = int(_required_text(arc, "src", owner=owner))
        target = int(_required_text(arc, "dst", owner=owner))
        missing = [node_id for node_id in (source, target) if node_id not in graph]
        if missing:
            raise ValueError(f"{owner} references undefined vertices: {missing}")
        graph.add_edge(source, target)

    return graph


def reverse_for_analysis(graph: nx.DiGraph) -> nx.DiGraph:
    """Return a copy whose edges mean requirement -> rule -> effect."""
    analysis_graph = graph.reverse(copy=True)
    analysis_graph.graph.update(graph.graph)
    analysis_graph.graph["edge_direction"] = "requirement_to_effect"
    return analysis_graph


def summarize_graph(graph: nx.DiGraph) -> dict[str, object]:
    """Return small deterministic statistics used by CLI progress and tests."""
    type_counts: dict[str, int] = {}
    for _node_id, attributes in graph.nodes(data=True):
        node_type = str(attributes.get("type") or "UNKNOWN")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "type_counts": dict(sorted(type_counts.items())),
    }
