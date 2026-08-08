"""Paired uncertainty estimates for candidate-ranking comparisons."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Set

import numpy as np

from cve2attack.schemas import CandidateRecord, records_by_id, technique_ids


def _per_cve_recall(
    records: Sequence[CandidateRecord],
    truth: Mapping[str, Set[str]],
    cutoff: int,
) -> np.ndarray:
    """Return recall for every sorted benchmark CVE, including missing runs as zero."""
    predictions = records_by_id(records)
    values: list[float] = []
    for cve_id in sorted(truth):
        expected = set(truth[cve_id])
        record = predictions.get(cve_id)
        ranked = technique_ids(record) if record else []
        matched = len(expected.intersection(ranked[:cutoff]))
        values.append(matched / len(expected) if expected else 0.0)
    return np.asarray(values, dtype=np.float64)


def _bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    iterations: int,
    seed: int,
    batch_size: int = 500,
) -> tuple[float, float]:
    """Compute a percentile CI in batches to bound memory on large cohorts."""
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if differences.size == 0:
        raise ValueError("paired comparison requires a non-empty cohort")
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for start in range(0, iterations, batch_size):
        count = min(batch_size, iterations - start)
        indices = rng.integers(0, differences.size, size=(count, differences.size))
        means.append(differences[indices].mean(axis=1))
    samples = np.concatenate(means)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def paired_recall_comparison(
    left_records: Sequence[CandidateRecord],
    right_records: Sequence[CandidateRecord],
    truth: Mapping[str, Set[str]],
    *,
    left_name: str,
    right_name: str,
    cutoffs: Sequence[int] = (10, 20),
    bootstrap_iterations: int = 10_000,
    seed: int = 20260728,
) -> dict[str, Any]:
    """Compare two runs on identical CVEs with paired bootstrap intervals."""
    result: dict[str, Any] = {
        "left": left_name,
        "right": right_name,
        "cohort_size": len(truth),
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "cutoffs": {},
    }
    tolerance = 1e-12
    for cutoff in cutoffs:
        if cutoff <= 0:
            raise ValueError("cutoffs must be positive")
        left = _per_cve_recall(left_records, truth, cutoff)
        right = _per_cve_recall(right_records, truth, cutoff)
        differences = right - left
        low, high = _bootstrap_mean_ci(
            differences,
            iterations=bootstrap_iterations,
            seed=seed + cutoff,
        )
        result["cutoffs"][str(cutoff)] = {
            "left_recall": float(left.mean()),
            "right_recall": float(right.mean()),
            "delta": float(differences.mean()),
            "ci95_low": low,
            "ci95_high": high,
            "improved_cves": int(np.sum(differences > tolerance)),
            "same_cves": int(np.sum(np.abs(differences) <= tolerance)),
            "worse_cves": int(np.sum(differences < -tolerance)),
        }
    return result
