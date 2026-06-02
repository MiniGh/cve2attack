#!/usr/bin/env python3
"""Evaluate embedding retrieval results by year.

The script compares `output/retrieval/CVE-*.jsonl` against the reference
datasets in `data_result/` and `cve2attack_result/`.

It writes a markdown report and a JSON summary with per-year metrics.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "output" / "retrieval"
DATA_REF_DIR = PROJECT_ROOT / "data_result"
CVE2ATTACK_REF_DIR = PROJECT_ROOT / "cve2attack_result"
OUTPUT_MD = PROJECT_ROOT / "output" / "retrieval" / "embedding_recall_by_year.md"
OUTPUT_JSON = PROJECT_ROOT / "output" / "retrieval" / "embedding_recall_by_year.json"


@dataclass(frozen=True)
class Metrics:
    cves: int
    truth_techniques: int
    predicted_techniques: int
    hit_rate_at_10: float
    hit_rate_at_20: float
    recall_at_10: float
    recall_at_20: float


def load_jsonl_directory(directory: Path) -> Dict[str, Dict[str, List[str]]]:
    """Load yearly JSONL files into a CVE -> techniques mapping."""
    per_year: Dict[str, Dict[str, List[str]]] = {}
    for path in sorted(directory.glob("CVE-*.jsonl")):
        year = path.stem.split("-")[1]
        year_map: Dict[str, List[str]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                cve_id = str(record["cve_id"])
                techniques = [str(t) for t in record.get("techniques", []) if str(t).strip()]
                year_map[cve_id] = techniques
        per_year[year] = year_map
    return per_year


def merge_reference_maps(*maps: Mapping[str, List[str]]) -> Dict[str, Set[str]]:
    merged: Dict[str, Set[str]] = defaultdict(set)
    for mapping in maps:
        for cve_id, techniques in mapping.items():
            merged[cve_id].update(str(t) for t in techniques if str(t).strip())
    return dict(merged)


def compute_metrics(pred: Mapping[str, List[str]], truth: Mapping[str, Set[str]]) -> Metrics:
    total_cves = len(truth)
    total_truth_techniques = sum(len(v) for v in truth.values())
    total_predicted_techniques = sum(len(pred.get(cve_id, [])) for cve_id in truth)

    hit_at_10 = 0.0
    hit_at_20 = 0.0
    recall_at_10_sum = 0.0
    recall_at_20_sum = 0.0

    for cve_id, truth_set in truth.items():
        predicted = pred.get(cve_id, [])
        top10 = predicted[:10]
        top20 = predicted[:20]

        truth_size = len(truth_set)
        if truth_size == 0:
            continue

        top10_set = set(top10)
        top20_set = set(top20)
        top10_hits = len(truth_set & top10_set)
        top20_hits = len(truth_set & top20_set)

        hit_at_10 += 1.0 if top10_hits else 0.0
        hit_at_20 += 1.0 if top20_hits else 0.0
        recall_at_10_sum += top10_hits / truth_size
        recall_at_20_sum += top20_hits / truth_size

    denom = float(total_cves) if total_cves else 1.0
    return Metrics(
        cves=total_cves,
        truth_techniques=total_truth_techniques,
        predicted_techniques=total_predicted_techniques,
        hit_rate_at_10=hit_at_10 / denom,
        hit_rate_at_20=hit_at_20 / denom,
        recall_at_10=recall_at_10_sum / denom,
        recall_at_20=recall_at_20_sum / denom,
    )


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(rows: List[dict], skipped_years: List[str]) -> None:
    lines: List[str] = [
        "# Embedding Retrieval Recall by Year",
        "",
        "Metric definition:",
        "- `hit_rate@k`: fraction of CVEs with at least one reference technique in the top-k retrieved techniques.",
        "- `recall@k`: average per-CVE technique recall, i.e. `|pred ∩ truth| / |truth|`.",
        "- Primary reference set: union of `data_result/` and `cve2attack_result/`.",
        "",
    ]

    if skipped_years:
        lines.append(f"Skipped years without reference data: {', '.join(skipped_years)}")
        lines.append("")

    header = [
        "Year",
        "CVEs",
        "Truth techniques",
        "Pred techniques",
        "Hit@10",
        "Hit@20",
        "Recall@10",
        "Recall@20",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    for row in rows:
        lines.append(
            "| {year} | {cves} | {truth} | {pred} | {hit10} | {hit20} | {rec10} | {rec20} |".format(
                year=row["year"],
                cves=row["metrics_union"].cves,
                truth=row["metrics_union"].truth_techniques,
                pred=row["metrics_union"].predicted_techniques,
                hit10=format_pct(row["metrics_union"].hit_rate_at_10),
                hit20=format_pct(row["metrics_union"].hit_rate_at_20),
                rec10=format_pct(row["metrics_union"].recall_at_10),
                rec20=format_pct(row["metrics_union"].recall_at_20),
            )
        )

    lines.extend([
        "",
        "Source breakdown is included in the JSON file for `data_result/` and `cve2attack_result/` separately.",
    ])

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    pred_by_year = load_jsonl_directory(PRED_DIR)
    data_by_year = load_jsonl_directory(DATA_REF_DIR)
    cve2attack_by_year = load_jsonl_directory(CVE2ATTACK_REF_DIR)

    all_years = sorted(set(data_by_year) | set(cve2attack_by_year))
    rows: List[dict] = []
    skipped_years: List[str] = []

    for year in all_years:
        data_truth = data_by_year.get(year, {})
        cve2attack_truth = cve2attack_by_year.get(year, {})
        union_truth = merge_reference_maps(data_truth, cve2attack_truth)
        if not union_truth:
            skipped_years.append(year)
            continue

        pred = pred_by_year.get(year, {})

        row = {
            "year": year,
            "metrics_data": compute_metrics(pred, merge_reference_maps(data_truth)),
            "metrics_cve2attack": compute_metrics(pred, merge_reference_maps(cve2attack_truth)),
            "metrics_union": compute_metrics(pred, union_truth),
        }
        rows.append(row)

    payload = {
        "primary_reference": "union(data_result, cve2attack_result)",
        "rows": [
            {
                "year": row["year"],
                "data_result": row["metrics_data"].__dict__,
                "cve2attack_result": row["metrics_cve2attack"].__dict__,
                "union": row["metrics_union"].__dict__,
            }
            for row in rows
        ],
        "skipped_years": skipped_years,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(rows, skipped_years)

    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
