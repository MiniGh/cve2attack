"""Markdown reports for single runs and multi-run comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from cve2attack.evaluation.metrics import EvaluationMetrics


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_run_report(
    path: Path,
    *,
    experiment_name: str,
    metrics: Mapping[str, EvaluationMetrics],
) -> None:
    lines = [
        f"# Run report: {experiment_name}",
        "",
        "Missing predictions count as misses. Coverage is reported explicitly.",
        "",
        "| Benchmark | CVEs | Predicted | Coverage | Hit@10 | Hit@20 | Recall@10 | Recall@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in metrics.items():
        lines.append(
            f"| {name} | {item.benchmark_cves} | {item.predicted_cves} | {_pct(item.coverage)} | "
            f"{_pct(item.hit_rate_at_10)} | {_pct(item.hit_rate_at_20)} | "
            f"{_pct(item.recall_at_10)} | {_pct(item.recall_at_20)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(
    path: Path,
    *,
    benchmark_name: str,
    rows: Mapping[str, EvaluationMetrics],
) -> None:
    lines = [
        f"# Run comparison: {benchmark_name}",
        "",
        "All runs use the benchmark's complete fixed cohort.",
        "",
        "| Run | CVEs | Predicted | Coverage | Recall@10 | Recall@20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in rows.items():
        lines.append(
            f"| {name} | {item.benchmark_cves} | {item.predicted_cves} | {_pct(item.coverage)} | "
            f"{_pct(item.recall_at_10)} | {_pct(item.recall_at_20)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
