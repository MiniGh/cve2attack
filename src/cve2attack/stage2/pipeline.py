"""File-level orchestration and progress reporting for stage-2 extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cve2attack.stage2.context_extractor import (
    CONTEXT_SCHEMA_VERSION,
    extract_cve_context,
    find_cve_nodes,
)
from cve2attack.stage2.graph_parser import (
    parse_xml_to_graph,
    reverse_for_analysis,
    summarize_graph,
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file to avoid partial results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_context_extraction(
    attack_graph_path: str | Path,
    output_path: str | Path,
    *,
    max_graph_depth: int = 2,
    overwrite: bool = False,
) -> Path:
    """Parse one attack graph and write the versioned context JSON document."""
    source = Path(attack_graph_path)
    output = Path(output_path)
    if max_graph_depth < 0:
        raise ValueError("max_graph_depth must be non-negative")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --force to replace it")

    print(f"[1/5] Parsing MulVAL attack graph: {source}")
    raw_graph = parse_xml_to_graph(source)
    summary = summarize_graph(raw_graph)
    print(
        "      "
        f"nodes={summary['node_count']} edges={summary['edge_count']} "
        f"types={summary['type_counts']}"
    )

    print("[2/5] Reversing edges to requirement -> rule -> effect")
    graph = reverse_for_analysis(raw_graph)

    print("[3/5] Locating vulExists nodes")
    cve_nodes = find_cve_nodes(graph)
    print(f"      found={len(cve_nodes)} node_ids={cve_nodes}")

    print(f"[4/5] Extracting local and graph context (max_depth={max_graph_depth})")
    contexts: list[dict[str, Any]] = []
    for position, node_id in enumerate(cve_nodes, start=1):
        context = extract_cve_context(
            graph,
            node_id,
            max_graph_depth=max_graph_depth,
        )
        contexts.append(context)
        print(
            f"      [{position}/{len(cve_nodes)}] "
            f"node={node_id} cve={context['cve_id']}"
        )

    payload = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "attack_graph": {
            "source": str(source.resolve()),
            **summary,
            "edge_direction": "requirement_to_effect",
        },
        "contexts": contexts,
    }
    print(f"[5/5] Writing context JSON: {output}")
    _write_json_atomic(output, payload)
    print(f"      complete contexts={len(contexts)}")
    return output
