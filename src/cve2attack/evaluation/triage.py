"""TRIAGE-specific reference comparison and disagreement reporting."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence, Set

import yaml

from cve2attack.data.loaders import benchmark_truth, candidate_records, iter_jsonl
from cve2attack.evaluation.ranking import (
    RankingMetrics,
    evaluate_rankings,
    rankings_from_records,
)
from cve2attack.schemas import parent_technique_id


TRIAGE_ALL_BENCHMARK = "triage_2025_test_all"
TRIAGE_NO_SECONDARY_BENCHMARK = "triage_2025_test_no_secondary"
REFERENCE_FILES = {
    TRIAGE_ALL_BENCHMARK: {
        "SMET (paper)": "smet_test_all.json",
        "TRIAGE (paper)": "triage_test_all.json",
    },
    TRIAGE_NO_SECONDARY_BENCHMARK: {
        "SMET (paper)": "smet_test_no_secondary.json",
        "TRIAGE (paper)": "triage_test_no_secondary.json",
    },
}


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _run_name(run_dir: Path) -> str:
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        value = json.loads(manifest.read_text(encoding="utf-8"))
        experiment = str(value.get("experiment") or "legacy")
        run_id = str(value.get("run_id") or run_dir.name)
        return f"{experiment} [{run_id}]"
    return run_dir.name


def load_reference_history(
    path: Path,
    *,
    expected_truth: Mapping[str, Set[str]],
) -> dict[str, list[str]]:
    """Load and validate one official TRIAGE/SMET history file.

    Empty prediction strings are intentionally retained because they occupied a
    rank in the authors' metric implementation.  Technique IDs are rolled up
    to parents in the same way as the published comparison.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a history list: {path}")
    predictions: dict[str, list[str]] = {}
    embedded_truth: dict[str, set[str]] = {}
    for item in value:
        cve_id = str(item["target_cve"])
        if cve_id in predictions:
            raise ValueError(f"Duplicate reference prediction for {cve_id}: {path}")
        ranked: list[str] = []
        for raw in item.get("predictions", []) or []:
            text = "N/A" if raw is None else str(raw).strip()
            ranked.append(parent_technique_id(text) if text.startswith("T") else text)
        predictions[cve_id] = ranked
        embedded_truth[cve_id] = {
            parent_technique_id(str(raw))
            for raw in item.get("true labels", []) or []
            if str(raw).strip()
        }
    if set(predictions) != set(expected_truth):
        raise ValueError(f"Reference cohort does not match benchmark: {path}")
    mismatched = [
        cve_id
        for cve_id in expected_truth
        if embedded_truth.get(cve_id, set()) != set(expected_truth[cve_id])
    ]
    if mismatched:
        raise ValueError(
            f"Reference ground truth differs from benchmark for {len(mismatched)} CVEs: {path}"
        )
    return predictions


def _truth_by_mapping_type(directory: Path) -> dict[str, dict[str, set[str]]]:
    views: dict[str, dict[str, set[str]]] = {
        "exploitation_technique": {},
        "primary_impact": {},
        "secondary_impact": {},
    }
    for path in sorted(directory.glob("CVE-*.jsonl")):
        for record in iter_jsonl(path):
            cve_id = str(record["cve_id"])
            by_type = record.get("labels_by_mapping_type", {})
            for mapping_type in views:
                item = by_type.get(mapping_type, {}) if isinstance(by_type, Mapping) else {}
                techniques = {
                    parent_technique_id(str(raw)) for raw in item.get("techniques", []) or []
                }
                # Mapping-type diagnostics use label-bearing CVEs only.  This
                # avoids treating "no approved label of this semantic type" as
                # a retrieval failure.
                if techniques:
                    views[mapping_type][cve_id] = techniques
    return views


def _reported_metrics(source_dir: Path) -> Mapping[str, Any]:
    path = source_dir / "source.yaml"
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    reported = value.get("reported_metrics", {}) if isinstance(value, Mapping) else {}
    return reported if isinstance(reported, Mapping) else {}


def _validate_paper_metrics(
    metrics: Mapping[str, Mapping[str, RankingMetrics]],
    reported: Mapping[str, Any],
) -> None:
    """Fail early if selected files no longer reproduce the public package."""
    metric_fields = {
        "MAP": "mean_average_precision",
        "Recall@5": "micro_recall_at_5",
        "Recall@10": "micro_recall_at_10",
        "Recall@20": "micro_recall_at_20",
    }
    for view, methods in reported.items():
        if view not in metrics or not isinstance(methods, Mapping):
            continue
        for method, expected in methods.items():
            display_name = f"{method} (paper)"
            if display_name not in metrics[view] or not isinstance(expected, Mapping):
                continue
            actual = metrics[view][display_name]
            for source_name, field_name in metric_fields.items():
                if source_name not in expected:
                    continue
                if abs(getattr(actual, field_name) - float(expected[source_name])) > 1e-9:
                    raise ValueError(
                        f"{display_name} {view} does not reproduce {source_name}"
                    )


def _write_metric_table(
    lines: list[str],
    title: str,
    rows: Mapping[str, RankingMetrics],
) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| Model | CVEs | Labels | Coverage | MAP | Hit@10 | Macro R@10 | Micro R@5 | Micro R@10 | Micro R@20 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in rows.items():
        lines.append(
            f"| {name} | {item.cohort_cves} | {item.relevant_labels} | {_pct(item.coverage)} | "
            f"{item.mean_average_precision:.4f} | {_pct(item.hit_rate_at_10)} | "
            f"{_pct(item.macro_recall_at_10)} | {_pct(item.micro_recall_at_5)} | "
            f"{_pct(item.micro_recall_at_10)} | {_pct(item.micro_recall_at_20)} |"
        )
    lines.append("")


def _disagreement_rows(
    run_rankings: Mapping[str, Sequence[str]],
    triage_rankings: Mapping[str, Sequence[str]],
    truth: Mapping[str, Set[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = {"both_hit": 0, "run_only": 0, "triage_only": 0, "neither": 0}
    overlaps: list[int] = []
    jaccards: list[float] = []
    for cve_id in sorted(truth):
        expected = set(truth[cve_id])
        run_top10 = list(run_rankings.get(cve_id, ()))[:10]
        triage_top10 = list(triage_rankings.get(cve_id, ()))[:10]
        run_hits = sorted(expected.intersection(run_top10))
        triage_hits = sorted(expected.intersection(triage_top10))
        if run_hits and triage_hits:
            category = "both_hit"
        elif run_hits:
            category = "run_only"
        elif triage_hits:
            category = "triage_only"
        else:
            category = "neither"
        counts[category] += 1
        run_set = set(run_top10)
        triage_set = set(triage_top10)
        overlap = len(run_set.intersection(triage_set))
        union = run_set.union(triage_set)
        overlaps.append(overlap)
        jaccards.append(overlap / len(union) if union else 1.0)
        rows.append(
            {
                "cve_id": cve_id,
                "category": category,
                "truth": sorted(expected),
                "run_top10": run_top10,
                "triage_top10": triage_top10,
                "run_hits": run_hits,
                "triage_hits": triage_hits,
                "top10_overlap": overlap,
            }
        )
    summary = {
        **counts,
        "mean_top10_overlap": mean(overlaps) if overlaps else 0.0,
        "mean_top10_jaccard": mean(jaccards) if jaccards else 0.0,
    }
    return rows, summary


def compare_with_triage(
    run_dirs: Sequence[Path],
    *,
    output_dir: Path,
    project_root: Path,
    source_dir: Path,
) -> Path:
    """Compare completed project runs with public TRIAGE and SMET predictions."""
    if output_dir.exists():
        raise FileExistsError(f"Comparison directory already exists: {output_dir}")
    benchmark_root = project_root / "data" / "benchmarks"
    reference_dir = source_dir / "reference_predictions"
    truths = {
        name: benchmark_truth(benchmark_root / name) for name in REFERENCE_FILES
    }
    if any(not truth for truth in truths.values()):
        raise RuntimeError("Run import-triage before compare-triage")

    print("[triage] loading public SMET and TRIAGE reference predictions")
    reference_rankings: dict[str, dict[str, dict[str, list[str]]]] = {}
    for view, files in REFERENCE_FILES.items():
        reference_rankings[view] = {
            name: load_reference_history(reference_dir / filename, expected_truth=truths[view])
            for name, filename in files.items()
        }

    run_rankings: dict[str, dict[str, list[str]]] = {}
    for raw_path in run_dirs:
        run_dir = raw_path if raw_path.is_absolute() else project_root / raw_path
        name = _run_name(run_dir)
        print(f"[triage] evaluating {name}")
        run_rankings[name] = rankings_from_records(candidate_records(run_dir))

    metrics: dict[str, dict[str, RankingMetrics]] = {}
    for view, truth in truths.items():
        rows = {
            name: evaluate_rankings(rankings, truth)
            for name, rankings in run_rankings.items()
        }
        rows.update(
            {
                name: evaluate_rankings(rankings, truth)
                for name, rankings in reference_rankings[view].items()
            }
        )
        metrics[view] = rows
    _validate_paper_metrics(metrics, _reported_metrics(source_dir))

    mapping_truths = _truth_by_mapping_type(benchmark_root / TRIAGE_ALL_BENCHMARK)
    mapping_metrics: dict[str, dict[str, RankingMetrics]] = {}
    all_sources = {
        **run_rankings,
        **reference_rankings[TRIAGE_ALL_BENCHMARK],
    }
    for mapping_type, truth in mapping_truths.items():
        mapping_metrics[mapping_type] = {
            name: evaluate_rankings(rankings, truth)
            for name, rankings in all_sources.items()
        }

    output_dir.mkdir(parents=True)
    payload = {
        "metric_semantics": {
            "macro_recall": "Recall is computed per CVE, then averaged over the fixed cohort.",
            "micro_recall": "TRIAGE paper metric: matched labels divided by all relevant labels.",
            "map": "Mean average precision using every stored candidate rank.",
            "missing_predictions": "Missing CVEs remain in the denominator as misses.",
        },
        "views": {
            view: {name: item.to_dict() for name, item in rows.items()}
            for view, rows in metrics.items()
        },
        "mapping_types": {
            mapping_type: {name: item.to_dict() for name, item in rows.items()}
            for mapping_type, rows in mapping_metrics.items()
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    triage_all = reference_rankings[TRIAGE_ALL_BENCHMARK]["TRIAGE (paper)"]
    disagreement_summaries: dict[str, dict[str, Any]] = {}
    for name, rankings in run_rankings.items():
        detail, summary = _disagreement_rows(rankings, triage_all, truths[TRIAGE_ALL_BENCHMARK])
        disagreement_summaries[name] = summary
        path = output_dir / f"disagreements_{_slug(name)}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in detail:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# TRIAGE public test comparison",
        "",
        "The cohort is the exact 60-CVE public TRIAGE test split. Reference files are taken from the public replication package.",
        "",
        "`Micro R@K` reproduces TRIAGE's pooled-label Recall@K. `Macro R@K` is this project's per-CVE recall. They must not be compared as if they were the same metric.",
        "",
        "## Interpretation boundaries",
        "",
        "- The test labels are a 60-CVE slice of the same 296-CVE CTID KEV mapping snapshot used by this project's KEV benchmark. This is an exact public split and reference-output comparison, not a new independent annotation source.",
        "- TRIAGE uses the other 236 labeled CVEs as in-context demonstrations and adds label-aware mapping components. The project runs are label-free candidate retrieval methods, so the result measures a capability gap under different supervision rather than a controlled same-supervision ablation.",
        "- Project runs normally store Top-20 candidates, while the public histories contain longer rankings. Recall/Hit at 5, 10 and 20 are directly aligned; MAP is still reported but is sensitive to the stored ranking depth.",
        "",
    ]
    _write_metric_table(lines, "All public mapping types", metrics[TRIAGE_ALL_BENCHMARK])
    _write_metric_table(lines, "Secondary impact excluded", metrics[TRIAGE_NO_SECONDARY_BENCHMARK])
    lines.extend(
        [
            "## Mapping-type diagnosis (label-bearing CVEs only)",
            "",
            "| Mapping type | Model | CVEs | Labels | MAP | Hit@10 | Macro R@10 | Micro R@10 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mapping_type, rows in mapping_metrics.items():
        for name, item in rows.items():
            lines.append(
                f"| {mapping_type} | {name} | {item.cohort_cves} | {item.relevant_labels} | "
                f"{item.mean_average_precision:.4f} | {_pct(item.hit_rate_at_10)} | "
                f"{_pct(item.macro_recall_at_10)} | {_pct(item.micro_recall_at_10)} |"
            )
    lines.extend(
        [
            "",
            "## Top-10 disagreement with TRIAGE",
            "",
            "| Run | Both hit | Run only | TRIAGE only | Neither | Mean overlap | Mean Jaccard |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in disagreement_summaries.items():
        lines.append(
            f"| {name} | {item['both_hit']} | {item['run_only']} | {item['triage_only']} | "
            f"{item['neither']} | {item['mean_top10_overlap']:.2f} | "
            f"{item['mean_top10_jaccard']:.3f} |"
        )
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[triage] comparison written to {output_dir}")
    return output_dir
