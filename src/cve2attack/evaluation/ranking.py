"""Ranking metrics used for TRIAGE-compatible candidate evaluation.

The project historically reports per-CVE (macro) recall.  TRIAGE instead
pools all relevant labels before computing recall.  Both views are useful,
but they answer different questions, so this module computes and names them
separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence, Set

from cve2attack.schemas import CandidateRecord, records_by_id, technique_ids


@dataclass(frozen=True)
class RankingMetrics:
    """Metrics for one fixed CVE cohort and one ranked prediction source."""

    cohort_cves: int
    predicted_cves: int
    relevant_labels: int
    coverage: float
    mean_average_precision: float
    hit_rate_at_5: float
    hit_rate_at_10: float
    hit_rate_at_20: float
    macro_recall_at_5: float
    macro_recall_at_10: float
    macro_recall_at_20: float
    micro_recall_at_5: float
    micro_recall_at_10: float
    micro_recall_at_20: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def rankings_from_records(records: Sequence[CandidateRecord]) -> dict[str, list[str]]:
    """Convert canonical candidate records to CVE -> ranked technique IDs."""
    indexed = records_by_id(records)
    return {cve_id: technique_ids(record) for cve_id, record in indexed.items()}


def _average_precision(expected: Set[str], ranked: Sequence[str]) -> float:
    """Reproduce the AP definition used by the TRIAGE replication package."""
    if not expected:
        return 0.0
    relevant_found = 0
    precision_sum = 0.0
    for rank, technique_id in enumerate(ranked, start=1):
        if technique_id in expected:
            relevant_found += 1
            precision_sum += relevant_found / rank
    return precision_sum / len(expected)


def evaluate_rankings(
    predictions: Mapping[str, Sequence[str]],
    truth: Mapping[str, Set[str]],
    *,
    cohort: Sequence[str] | None = None,
) -> RankingMetrics:
    """Evaluate rankings with both project-style macro and TRIAGE-style micro recall.

    Missing predictions remain in every denominator.  Empty ground-truth sets
    contribute zero to macro metrics and do not add labels to the micro
    denominator.  The frozen TRIAGE all/no-secondary views contain at least one
    approved technique for every CVE.
    """
    identifiers = list(cohort) if cohort is not None else sorted(truth)
    total_cves = len(identifiers)
    denominator_cves = float(total_cves) if total_cves else 1.0
    predicted_cves = sum(cve_id in predictions for cve_id in identifiers)

    average_precision = 0.0
    hits = {5: 0.0, 10: 0.0, 20: 0.0}
    macro_recall = {5: 0.0, 10: 0.0, 20: 0.0}
    micro_matches = {5: 0, 10: 0, 20: 0}
    relevant_labels = 0

    for cve_id in identifiers:
        expected = set(truth.get(cve_id, set()))
        ranked = list(predictions.get(cve_id, ()))
        average_precision += _average_precision(expected, ranked)
        relevant_labels += len(expected)
        if not expected:
            continue
        for cutoff in (5, 10, 20):
            matches = len(expected.intersection(set(ranked[:cutoff])))
            hits[cutoff] += float(matches > 0)
            macro_recall[cutoff] += matches / len(expected)
            micro_matches[cutoff] += matches

    denominator_labels = float(relevant_labels) if relevant_labels else 1.0
    return RankingMetrics(
        cohort_cves=total_cves,
        predicted_cves=predicted_cves,
        relevant_labels=relevant_labels,
        coverage=predicted_cves / denominator_cves,
        mean_average_precision=average_precision / denominator_cves,
        hit_rate_at_5=hits[5] / denominator_cves,
        hit_rate_at_10=hits[10] / denominator_cves,
        hit_rate_at_20=hits[20] / denominator_cves,
        macro_recall_at_5=macro_recall[5] / denominator_cves,
        macro_recall_at_10=macro_recall[10] / denominator_cves,
        macro_recall_at_20=macro_recall[20] / denominator_cves,
        micro_recall_at_5=micro_matches[5] / denominator_labels,
        micro_recall_at_10=micro_matches[10] / denominator_labels,
        micro_recall_at_20=micro_matches[20] / denominator_labels,
    )
