"""Final frozen-method audit for Stage-1 action-level retrieval.

The report combines three questions that should be answered before V5c is
declared the frozen Stage-1 method: which ATT&CK source types provide recall,
whether procedure-rich Techniques receive disproportionate exposure, and
which true labels are gained, lost, or still outside the controlled Top-20.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cve2attack.data.loaders import benchmark_truth, candidate_records
from cve2attack.evaluation.ranking import evaluate_rankings, rankings_from_records
from cve2attack.retrieval.action_kb import load_action_documents
from cve2attack.schemas import CandidateRecord, records_by_id


ProgressReporter = Callable[[str], None]


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Return one-based average ranks and assign tied values the same rank."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Compute Spearman's rho without adding a new SciPy dependency."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def classify_case(v1_rank: int | None, v5_rank: int | None) -> str:
    """Classify one true label using the controlled Top-20 decision boundary."""
    v1_hit = v1_rank is not None and v1_rank <= 20
    v5_hit = v5_rank is not None and v5_rank <= 20
    if v5_hit and not v1_hit:
        return "new_v5_hit"
    if v1_hit and not v5_hit:
        return "lost_v1_hit"
    if v1_hit and v5_hit:
        return "retained_hit"
    if v5_rank is not None and v5_rank <= 50:
        return "unresolved_rank_21_50"
    return "unresolved_beyond_50"


def _rank_maps(records: Sequence[CandidateRecord]) -> dict[str, dict[str, int]]:
    return {
        record.cve_id: {
            candidate.technique_id: rank
            for rank, candidate in enumerate(record.candidates, start=1)
        }
        for record in records
    }


def _candidate_evidence(
    records: Mapping[str, CandidateRecord], cve_id: str, technique_id: str
) -> list[dict[str, Any]]:
    record = records.get(cve_id)
    if record is None:
        return []
    for candidate in record.candidates:
        if candidate.technique_id == technique_id:
            raw = candidate.metadata.get("action_evidence", [])
            return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_action_final_audit(
    *,
    attack_bundle: Path,
    benchmark_dir: Path,
    run_dirs: Mapping[str, Path],
    output_dir: Path,
    progress: ProgressReporter | None = None,
) -> Path:
    """Write corpus ablation, procedure-bias and label-level case artifacts."""
    required = {"v1", "parent", "subtechnique", "descriptions", "procedure", "full"}
    missing = required - set(run_dirs)
    if missing:
        raise ValueError(f"Missing required action-final runs: {sorted(missing)}")
    if output_dir.exists():
        raise FileExistsError(f"Comparison directory already exists: {output_dir}")

    _report(progress, "loading frozen benchmark and six candidate rankings")
    truth = benchmark_truth(benchmark_dir)
    if not truth:
        raise RuntimeError(f"Benchmark has no records: {benchmark_dir}")
    records = {name: candidate_records(path) for name, path in run_dirs.items()}
    predictions = {
        name: rankings_from_records(run_records)
        for name, run_records in records.items()
    }
    metrics = {
        name: evaluate_rankings(ranking, truth).to_dict()
        for name, ranking in predictions.items()
    }

    _report(progress, "counting ATT&CK source documents per parent Technique")
    actions = load_action_documents(attack_bundle)
    source_counts: dict[str, Counter[str]] = {}
    for action in actions:
        source_counts.setdefault(action.technique_id, Counter())[action.source_type] += 1
    all_techniques = sorted(source_counts)

    v1_top20 = Counter(
        technique_id
        for ranking in predictions["v1"].values()
        for technique_id in ranking[:20]
    )
    v5_top20 = Counter(
        technique_id
        for ranking in predictions["full"].values()
        for technique_id in ranking[:20]
    )
    truth_occurrences = Counter(
        technique_id for labels in truth.values() for technique_id in labels
    )
    v5_ranks = _rank_maps(records["full"])
    v1_ranks = _rank_maps(records["v1"])
    true_hits = Counter()
    truth_rank_values: dict[str, list[int]] = {}
    for cve_id, labels in truth.items():
        for technique_id in labels:
            rank = v5_ranks.get(cve_id, {}).get(technique_id)
            if rank is not None:
                truth_rank_values.setdefault(technique_id, []).append(rank)
            if rank is not None and rank <= 20:
                true_hits[technique_id] += 1

    technique_rows: list[dict[str, Any]] = []
    for technique_id in all_techniques:
        counts = source_counts[technique_id]
        occurrences = truth_occurrences[technique_id]
        hits = true_hits[technique_id]
        ranks = truth_rank_values.get(technique_id, [])
        technique_rows.append(
            {
                "technique_id": technique_id,
                "procedure_count": counts["procedure"],
                "parent_description_count": counts["technique_description"],
                "subtechnique_description_count": counts["subtechnique_description"],
                "v1_top20_exposure": v1_top20[technique_id],
                "v5_top20_exposure": v5_top20[technique_id],
                "v5_false_positive_exposure": max(0, v5_top20[technique_id] - hits),
                "truth_occurrences": occurrences,
                "v5_true_hits_at_20": hits,
                "v5_label_recall_at_20": hits / occurrences if occurrences else None,
                "v5_mean_truth_rank": sum(ranks) / len(ranks) if ranks else None,
            }
        )

    procedure_counts = [row["procedure_count"] for row in technique_rows]
    bias = {
        "techniques": len(technique_rows),
        "procedure_count_quantiles": {
            "q25": float(np.quantile(procedure_counts, 0.25)),
            "q50": float(np.quantile(procedure_counts, 0.50)),
            "q75": float(np.quantile(procedure_counts, 0.75)),
        },
        "spearman_procedure_vs_v5_top20_exposure": spearman_correlation(
            procedure_counts, [row["v5_top20_exposure"] for row in technique_rows]
        ),
        "spearman_procedure_vs_v5_false_positive_exposure": spearman_correlation(
            procedure_counts,
            [row["v5_false_positive_exposure"] for row in technique_rows],
        ),
    }
    labeled_rows = [row for row in technique_rows if row["truth_occurrences"]]
    bias["spearman_procedure_vs_label_recall_at_20"] = spearman_correlation(
        [row["procedure_count"] for row in labeled_rows],
        [row["v5_label_recall_at_20"] for row in labeled_rows],
    )

    _report(progress, "classifying every benchmark true label and attaching V5 evidence")
    full_index = records_by_id(records["full"])
    cases: list[dict[str, Any]] = []
    for cve_id in sorted(truth):
        for technique_id in sorted(truth[cve_id]):
            v1_rank = v1_ranks.get(cve_id, {}).get(technique_id)
            v5_rank = v5_ranks.get(cve_id, {}).get(technique_id)
            cases.append(
                {
                    "cve_id": cve_id,
                    "technique_id": technique_id,
                    "v1_rank": v1_rank,
                    "v5_rank": v5_rank,
                    "status": classify_case(v1_rank, v5_rank),
                    "rank_improvement": (
                        v1_rank - v5_rank
                        if v1_rank is not None and v5_rank is not None
                        else None
                    ),
                    "procedure_count": source_counts.get(technique_id, Counter())["procedure"],
                    "action_evidence": _candidate_evidence(
                        full_index, cve_id, technique_id
                    ),
                }
            )
    case_counts = Counter(row["status"] for row in cases)

    payload = {
        "benchmark": benchmark_dir.name,
        "cohort_cves": len(truth),
        "relevant_labels": sum(len(labels) for labels in truth.values()),
        "run_dirs": {name: str(path) for name, path in run_dirs.items()},
        "metrics": metrics,
        "procedure_bias": bias,
        "case_counts": dict(sorted(case_counts.items())),
    }

    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_jsonl(output_dir / "technique_bias.jsonl", technique_rows)
    _write_jsonl(output_dir / "cases.jsonl", cases)

    labels = {
        "v1": "V1 parent Technique document",
        "parent": "V5 parent descriptions only",
        "subtechnique": "V5 sub-technique descriptions only",
        "descriptions": "V5 parent + sub-technique descriptions",
        "procedure": "V5 procedures only (strict LOO)",
        "full": "V5c all action types (strict LOO)",
    }
    lines = [
        "# Stage-1 V5c final action audit",
        "",
        f"Benchmark: `{benchmark_dir.name}` ({len(truth)} CVEs, "
        f"{sum(len(labels_) for labels_ in truth.values())} parent-Technique labels).",
        "All ablations keep the raw query, ATTACK-BERT, Top-3 rank-RRF, "
        "`rank_constant=60` and the controlled Top-20 boundary fixed.",
        "",
        "## Corpus ablation",
        "",
        "| Corpus | Micro R@5 | Micro R@10 | Micro R@20 | Macro R@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("v1", "parent", "subtechnique", "descriptions", "procedure", "full"):
        point = metrics[name]
        lines.append(
            f"| {labels[name]} | {point['micro_recall_at_5']:.2%} | "
            f"{point['micro_recall_at_10']:.2%} | {point['micro_recall_at_20']:.2%} | "
            f"{point['macro_recall_at_20']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Procedure-count bias diagnostics",
            "",
            "These are descriptive correlations across parent Techniques, not causal estimates.",
            "",
            "| Diagnostic | Spearman rho |",
            "|---|---:|",
            f"| Procedure count vs V5c Top-20 exposure | {bias['spearman_procedure_vs_v5_top20_exposure']:.3f} |",
            f"| Procedure count vs V5c false-positive exposure | {bias['spearman_procedure_vs_v5_false_positive_exposure']:.3f} |",
            f"| Procedure count vs label Recall@20 (labeled Techniques only) | {bias['spearman_procedure_vs_label_recall_at_20']:.3f} |",
            "",
            f"Procedure-count quartiles across {len(technique_rows)} parent Techniques: "
            f"Q1={bias['procedure_count_quantiles']['q25']:.1f}, "
            f"median={bias['procedure_count_quantiles']['q50']:.1f}, "
            f"Q3={bias['procedure_count_quantiles']['q75']:.1f}.",
            "",
            "## True-label case accounting",
            "",
            "| Status | Labels |",
            "|---|---:|",
        ]
    )
    for status in (
        "new_v5_hit",
        "retained_hit",
        "lost_v1_hit",
        "unresolved_rank_21_50",
        "unresolved_beyond_50",
    ):
        lines.append(f"| `{status}` | {case_counts[status]} |")

    for title, status in (
        ("Representative V5c gains", "new_v5_hit"),
        ("V1 hits lost by V5c", "lost_v1_hit"),
        ("Still outside Top-50", "unresolved_beyond_50"),
    ):
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                "| CVE | Technique | V1 rank | V5c rank | Best action type | Evidence |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        selected = [row for row in cases if row["status"] == status][:10]
        if not selected:
            lines.append("| — | — | — | — | — | None |")
        for row in selected:
            evidence = row["action_evidence"][0] if row["action_evidence"] else {}
            lines.append(
                f"| {row['cve_id']} | {row['technique_id']} | {row['v1_rank'] or '—'} | "
                f"{row['v5_rank'] or '—'} | {evidence.get('source_type', '—')} | "
                f"{_escape_table(str(evidence.get('text', ''))[:180])} |"
            )

    lines.extend(
        [
            "",
            "Complete per-Technique statistics are in `technique_bias.jsonl`; complete true-label",
            "ranks and traceable action evidence are in `cases.jsonl`.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _report(progress, f"wrote final audit to {output_dir}")
    return output_dir
