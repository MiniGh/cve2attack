"""Audit direct CVE references shared by ATT&CK procedures and a benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cve2attack.data.loaders import benchmark_truth
from cve2attack.retrieval.action_kb import extract_vulnerability_ids
from cve2attack.retrieval.technique_kb import normalize_text
from cve2attack.schemas import parent_technique_id


def _external_id(references: Sequence[Mapping[str, Any]]) -> str | None:
    for reference in references or []:
        if reference.get("source_name") == "mitre-attack":
            value = str(reference.get("external_id") or "").strip().upper()
            if value:
                return value
    return None


def audit_procedure_overlap(
    *, attack_bundle: Path, benchmark_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure exact benchmark-CVE references in ATT&CK procedure relationships."""
    truth = benchmark_truth(benchmark_dir)
    if not truth:
        raise ValueError(f"Benchmark has no labels: {benchmark_dir}")
    with attack_bundle.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    objects = bundle.get("objects", []) if isinstance(bundle, Mapping) else []
    if not isinstance(objects, list):
        raise ValueError(f"ATT&CK bundle objects must be a list: {attack_bundle}")

    targets: dict[str, str] = {}
    for obj in objects:
        if not isinstance(obj, Mapping) or obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue
        stix_id = str(obj.get("id") or "")
        external_id = _external_id(obj.get("external_references", []))
        if stix_id and external_id and external_id.startswith("T"):
            targets[stix_id] = parent_technique_id(external_id)

    rows: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        if obj.get("type") != "relationship" or obj.get("relationship_type") != "uses":
            continue
        technique_id = targets.get(str(obj.get("target_ref") or ""))
        if not technique_id:
            continue
        description = str(obj.get("description") or "")
        for cve_id in extract_vulnerability_ids(description):
            if cve_id not in truth:
                continue
            rows.append(
                {
                    "cve_id": cve_id,
                    "technique_id": technique_id,
                    "is_benchmark_label": technique_id in truth[cve_id],
                    "relationship_id": str(obj.get("id") or ""),
                    "source_ref": str(obj.get("source_ref") or ""),
                    "target_ref": str(obj.get("target_ref") or ""),
                    "description": normalize_text(description)[:800],
                }
            )
    rows.sort(key=lambda item: (item["cve_id"], item["technique_id"], item["relationship_id"]))

    direct_pairs = {
        (row["cve_id"], row["technique_id"])
        for row in rows
        if row["is_benchmark_label"]
    }
    mentioned_cves = {row["cve_id"] for row in rows}
    direct_cves = {cve_id for cve_id, _ in direct_pairs}
    benchmark_pairs = sum(len(labels) for labels in truth.values())
    summary = {
        "benchmark": benchmark_dir.name,
        "benchmark_cves": len(truth),
        "benchmark_label_pairs": benchmark_pairs,
        "procedure_reference_rows": len(rows),
        "directly_mentioned_cves": len(mentioned_cves),
        "direct_true_label_pairs": len(direct_pairs),
        "cves_with_direct_true_label": len(direct_cves),
        "direct_true_label_fraction": len(direct_pairs) / benchmark_pairs if benchmark_pairs else 0.0,
    }
    return summary, rows


def write_action_overlap_audit(
    *, attack_bundle: Path, benchmark_dir: Path, output_dir: Path
) -> Path:
    """Write a machine-readable and human-readable procedure-overlap audit."""
    if output_dir.exists():
        raise FileExistsError(f"Comparison directory already exists: {output_dir}")
    print(
        f"[action-audit] benchmark={benchmark_dir.name}; corpus={attack_bundle}",
        flush=True,
    )
    summary, rows = audit_procedure_overlap(
        attack_bundle=attack_bundle,
        benchmark_dir=benchmark_dir,
    )
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "overlaps.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = (
        "# ATT&CK procedure / benchmark CVE overlap audit\n\n"
        "This report detects exact CVE or legacy CAN identifiers in ATT&CK `uses` "
        "relationship descriptions. A direct true-label pair is evidence of corpus / "
        "benchmark overlap and must not be presented as independent retrieval evidence.\n\n"
        "| Benchmark CVEs | Benchmark labels | Mentioned CVEs | Direct true-label CVEs | "
        "Direct true-label pairs | Label fraction |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        f"| {summary['benchmark_cves']} | {summary['benchmark_label_pairs']} | "
        f"{summary['directly_mentioned_cves']} | {summary['cves_with_direct_true_label']} | "
        f"{summary['direct_true_label_pairs']} | "
        f"{summary['direct_true_label_fraction']:.2%} |\n\n"
        "Detailed relationship rows are stored in `overlaps.jsonl`.\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(
        f"[action-audit] mentioned_cves={summary['directly_mentioned_cves']}; "
        f"direct_true_pairs={summary['direct_true_label_pairs']}/"
        f"{summary['benchmark_label_pairs']}; output={output_dir}",
        flush=True,
    )
    return output_dir
