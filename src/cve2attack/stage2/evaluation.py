"""Before/after ranking metrics for the stage-2 closed loop."""

from __future__ import annotations

from statistics import mean
from typing import Any, Iterable, Mapping


def _candidate_ids(record: Mapping[str, Any]) -> list[str]:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Stage-2 record must contain a candidates list")
    return [str(candidate.get("technique_id") or "") for candidate in candidates]


def _best_rank(candidate_ids: list[str], labels: set[str]) -> int | None:
    ranks = [rank for rank, technique_id in enumerate(candidate_ids, start=1) if technique_id in labels]
    return min(ranks) if ranks else None


def _ranking_summary(ranks: list[int | None]) -> dict[str, float]:
    count = len(ranks)
    if count == 0:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0}
    return {
        "top1": sum(rank == 1 for rank in ranks) / count,
        "top3": sum(rank is not None and rank <= 3 for rank in ranks) / count,
        "top5": sum(rank is not None and rank <= 5 for rank in ranks) / count,
        "mrr": mean(0.0 if rank is None else 1.0 / rank for rank in ranks),
    }


def evaluate_reranking(
    joined_records: Iterable[Mapping[str, Any]],
    reranked_records: Iterable[Mapping[str, Any]],
    truth: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Compare original and reranked order on exactly the same candidates."""
    original_by_cve = {str(record.get("cve_id")): record for record in joined_records}
    reranked_by_cve = {str(record.get("cve_id")): record for record in reranked_records}
    if set(original_by_cve) != set(reranked_by_cve):
        raise ValueError("Original and reranked records cover different CVE IDs")

    cases: list[dict[str, Any]] = []
    original_ranks: list[int | None] = []
    reranked_ranks: list[int | None] = []
    wins = ties = losses = unrecoverable = 0

    for cve_id in sorted(original_by_cve):
        labels = set(truth.get(cve_id, set()))
        if not labels:
            continue
        original_ids = _candidate_ids(original_by_cve[cve_id])
        reranked_ids = _candidate_ids(reranked_by_cve[cve_id])
        if set(original_ids) != set(reranked_ids) or len(original_ids) != len(reranked_ids):
            raise ValueError(f"Candidate set changed during reranking: {cve_id}")

        original_rank = _best_rank(original_ids, labels)
        reranked_rank = _best_rank(reranked_ids, labels)
        original_ranks.append(original_rank)
        reranked_ranks.append(reranked_rank)
        if original_rank is None:
            unrecoverable += 1
            outcome = "unrecoverable"
        elif reranked_rank is not None and reranked_rank < original_rank:
            wins += 1
            outcome = "improved"
        elif reranked_rank is not None and reranked_rank > original_rank:
            losses += 1
            outcome = "degraded"
        else:
            ties += 1
            outcome = "unchanged"

        cases.append(
            {
                "cve_id": cve_id,
                "labels": sorted(labels),
                "candidate_count": len(original_ids),
                "candidate_set_preserved": True,
                "best_original_rank": original_rank,
                "best_reranked_rank": reranked_rank,
                "rank_gain": None
                if original_rank is None or reranked_rank is None
                else original_rank - reranked_rank,
                "original_top1": original_ids[0] if original_ids else None,
                "reranked_top1": reranked_ids[0] if reranked_ids else None,
                "outcome": outcome,
            }
        )

    return {
        "evaluated_cves": len(cases),
        "candidate_sets_preserved": all(case["candidate_set_preserved"] for case in cases),
        "original": _ranking_summary(original_ranks),
        "reranked": _ranking_summary(reranked_ranks),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "unrecoverable": unrecoverable,
        "cases": cases,
    }
