"""File-level orchestration and progress reporting for stage-2 extraction."""

from __future__ import annotations

import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cve2attack.data.loaders import benchmark_truth, candidate_records
from cve2attack.stage2.candidate_joiner import join_contexts_with_candidates
from cve2attack.stage2.context_extractor import (
    CONTEXT_SCHEMA_VERSION,
    extract_cve_context,
    find_cve_nodes,
)
from cve2attack.stage2.evaluation import evaluate_reranking
from cve2attack.stage2.graph_parser import (
    parse_xml_to_graph,
    reverse_for_analysis,
    summarize_graph,
)
from cve2attack.stage2.reranker import RULESET_VERSION, rerank_joined_records


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file to avoid partial results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str | None:
    """Return the current commit without failing runs outside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_stage2_report(
    path: Path,
    *,
    metrics: Mapping[str, Any],
    join_stats: Mapping[str, Any],
    scenario_kind: str,
    stage1_run: Path,
    attack_graph: Path,
) -> None:
    """Write a compact human-readable before/after case report."""
    original = metrics["original"]
    reranked = metrics["reranked"]
    if "synthetic" in scenario_kind:
        claim_boundary = (
            "This run is an engineering smoke validation. A synthetic topology "
            "does not constitute independent experimental evidence."
        )
    elif "trace_derived" in scenario_kind:
        claim_boundary = (
            "This graph was normalized from a public scenario execution trace. "
            "It is a reproducible case study, not a population-level benchmark."
        )
    else:
        claim_boundary = (
            "Interpret this run according to the provenance recorded by its "
            "scenario kind and input manifest."
        )
    lines = [
        "# Stage-2 closed-loop report",
        "",
        f"> {claim_boundary}",
        "",
        f"- Scenario kind: `{scenario_kind}`",
        f"- Stage-1 run: `{stage1_run}`",
        f"- Attack graph: `{attack_graph}`",
        f"- Reranker: `{RULESET_VERSION}` (topology only)",
        f"- Matched graph CVEs: {join_stats['matched']}",
        "",
        "## Summary",
        "",
        "| Ranking | Top-1 | Top-3 | Top-5 | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Stage 1 | {original['top1']:.3f} | {original['top3']:.3f} | {original['top5']:.3f} | {original['mrr']:.3f} |",
        f"| Stage 2 | {reranked['top1']:.3f} | {reranked['top3']:.3f} | {reranked['top5']:.3f} | {reranked['mrr']:.3f} |",
        "",
        f"Candidate sets preserved: `{metrics['candidate_sets_preserved']}`",
        "",
        "## Cases",
        "",
        "| CVE | Labels | Original best rank | Reranked best rank | Original Top-1 | Reranked Top-1 | Outcome |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for case in metrics["cases"]:
        labels = ", ".join(case["labels"])
        lines.append(
            f"| {case['cve_id']} | {labels} | {case['best_original_rank']} | "
            f"{case['best_reranked_rank']} | {case['original_top1']} | "
            f"{case['reranked_top1']} | {case['outcome']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def run_stage2_experiment(
    *,
    stage1_run: str | Path,
    attack_graph_path: str | Path,
    benchmark_dir: str | Path,
    output_root: str | Path,
    run_id: str,
    project_root: str | Path,
    scenario_kind: str,
    max_graph_depth: int = 2,
) -> Path:
    """Run the first reproducible stage-1 -> context -> reranking loop."""
    stage1_path = Path(stage1_run).resolve()
    graph_path = Path(attack_graph_path).resolve()
    truth_path = Path(benchmark_dir).resolve()
    root = Path(project_root).resolve()
    if not stage1_path.is_dir():
        raise FileNotFoundError(f"Stage-1 run directory does not exist: {stage1_path}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"Attack graph does not exist: {graph_path}")
    if not truth_path.is_dir():
        raise FileNotFoundError(f"Benchmark directory does not exist: {truth_path}")
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one safe directory name")
    if not scenario_kind.strip():
        raise ValueError("scenario_kind must be non-empty")

    stage1_manifest_path = stage1_path / "manifest.json"
    stage1_manifest: dict[str, Any] = {}
    if stage1_manifest_path.is_file():
        raw_manifest = json.loads(stage1_manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw_manifest, dict):
            stage1_manifest = raw_manifest
        status = str(stage1_manifest.get("status") or "")
        if status and status not in {"complete", "imported"}:
            raise RuntimeError(f"Stage-1 run is not complete: status={status}")

    output_dir = Path(output_root).resolve() / run_id
    if output_dir.exists():
        raise FileExistsError(f"Stage-2 run directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "scenario_kind": scenario_kind,
        "reranker": RULESET_VERSION,
        "uses_target_semantics": False,
        "inputs": {
            "stage1_run": str(stage1_path),
            "stage1_run_id": stage1_manifest.get("run_id"),
            "stage1_git_commit": stage1_manifest.get("git_commit"),
            "attack_graph": str(graph_path),
            "attack_graph_sha256": _sha256(graph_path),
            "benchmark": str(truth_path),
        },
        "parameters": {"max_graph_depth": max_graph_depth},
    }
    _write_json_atomic(manifest_path, manifest)

    try:
        print(f"[stage2 1/6] Extracting graph context for run={run_id}")
        contexts_path = output_dir / "contexts.json"
        run_context_extraction(
            graph_path,
            contexts_path,
            max_graph_depth=max_graph_depth,
        )
        context_document = json.loads(contexts_path.read_text(encoding="utf-8"))

        print(f"[stage2 2/6] Loading stage-1 candidates: {stage1_path}")
        stage1_records = candidate_records(stage1_path)
        if not stage1_records:
            raise RuntimeError(f"No candidate records found in stage-1 run: {stage1_path}")
        print(f"      candidate_records={len(stage1_records)}")

        print("[stage2 3/6] Joining CVE contexts with CandidateRecord")
        joined_records, stats = join_contexts_with_candidates(context_document, stage1_records)
        join_stats = stats.to_dict()
        print(
            f"      matched={stats.matched} missing_candidates={len(stats.missing_candidates)} "
            f"unresolved={len(stats.unresolved_context_ids)}"
        )
        if not joined_records:
            raise RuntimeError("No graph CVE could be joined with stage-1 candidates")
        _write_jsonl_atomic(output_dir / "joined_records.jsonl", joined_records)
        _write_json_atomic(output_dir / "join_stats.json", join_stats)

        print(f"[stage2 4/6] Applying deterministic topology rules: {RULESET_VERSION}")
        reranked_records = rerank_joined_records(joined_records)
        _write_jsonl_atomic(output_dir / "reranked_records.jsonl", reranked_records)
        for record in reranked_records:
            detected = record["reranker"]["detected_rules"]
            print(
                f"      cve={record['cve_id']} rules="
                f"{[rule['rule_id'] for rule in detected]} top1={record['candidates'][0]['technique_id']}"
            )

        print(f"[stage2 5/6] Comparing rankings on benchmark: {truth_path.name}")
        truth = benchmark_truth(truth_path)
        if not truth:
            raise RuntimeError(f"Benchmark contains no CVE labels: {truth_path}")
        metrics = evaluate_reranking(joined_records, reranked_records, truth)
        if metrics["evaluated_cves"] == 0:
            raise RuntimeError("No joined CVE has labels in the selected benchmark")
        _write_json_atomic(output_dir / "metrics.json", metrics)
        _write_stage2_report(
            output_dir / "report.md",
            metrics=metrics,
            join_stats=join_stats,
            scenario_kind=scenario_kind,
            stage1_run=stage1_path,
            attack_graph=graph_path,
        )
        print(
            f"      evaluated={metrics['evaluated_cves']} wins={metrics['wins']} "
            f"losses={metrics['losses']} ties={metrics['ties']}"
        )

        print(f"[stage2 6/6] Finalizing run: {output_dir}")
        manifest.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "join_stats": join_stats,
                "metrics_summary": {
                    key: metrics[key]
                    for key in ("evaluated_cves", "original", "reranked", "wins", "ties", "losses", "unrecoverable")
                },
            }
        )
        _write_json_atomic(manifest_path, manifest)
        print("      complete")
        return output_dir
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _write_json_atomic(manifest_path, manifest)
        raise
