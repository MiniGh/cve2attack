"""Evaluation metrics with explicit prediction coverage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence, Set

from cve2attack.schemas import CandidateRecord, records_by_id, technique_ids


@dataclass(frozen=True)
class EvaluationMetrics:
    benchmark_cves: int
    predicted_cves: int
    coverage: float
    hit_rate_at_10: float
    hit_rate_at_20: float
    recall_at_10: float
    recall_at_20: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate(
    records: Sequence[CandidateRecord],
    truth: Mapping[str, Set[str]],
) -> EvaluationMetrics:
    """Evaluate on every benchmark CVE; absent predictions are misses."""
    predictions = records_by_id(records)
    total = len(truth)
    predicted = sum(1 for cve_id in truth if cve_id in predictions)
    hit10 = hit20 = recall10 = recall20 = 0.0

    for cve_id, expected in truth.items():
        record = predictions.get(cve_id)
        ranked = technique_ids(record) if record else []
        top10 = set(ranked[:10])
        top20 = set(ranked[:20])
        if not expected:
            continue
        matches10 = len(top10 & expected)
        matches20 = len(top20 & expected)
        hit10 += float(matches10 > 0)
        hit20 += float(matches20 > 0)
        recall10 += matches10 / len(expected)
        recall20 += matches20 / len(expected)

    denominator = float(total) if total else 1.0
    return EvaluationMetrics(
        benchmark_cves=total,
        predicted_cves=predicted,
        coverage=predicted / denominator,
        hit_rate_at_10=hit10 / denominator,
        hit_rate_at_20=hit20 / denominator,
        recall_at_10=recall10 / denominator,
        recall_at_20=recall20 / denominator,
    )
