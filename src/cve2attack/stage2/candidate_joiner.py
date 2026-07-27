"""Join stage-1 candidate records with extracted attack-graph contexts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from cve2attack.schemas import CandidateRecord
from cve2attack.stage2.context_extractor import normalize_cve_id


@dataclass(frozen=True)
class JoinStats:
    """Coverage statistics that prevent silent loss at the stage boundary."""

    context_records: int
    candidate_records: int
    matched: int
    missing_candidates: tuple[str, ...]
    candidates_without_context: tuple[str, ...]
    unresolved_context_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_index(records: Iterable[CandidateRecord]) -> dict[str, CandidateRecord]:
    """Index records by normalized CVE ID and reject ambiguous duplicates."""
    result: dict[str, CandidateRecord] = {}
    for record in records:
        cve_id = normalize_cve_id(record.cve_id)
        if cve_id in result:
            raise ValueError(f"Duplicate stage-1 candidate record after normalization: {cve_id}")
        result[cve_id] = record
    return result


def join_contexts_with_candidates(
    context_document: Mapping[str, Any],
    candidate_records: Iterable[CandidateRecord],
) -> tuple[list[dict[str, Any]], JoinStats]:
    """Return matched stage-2 records without modifying either input.

    Context records with non-CVE identifiers, such as a generic ``vulID``, are
    reported as unresolved.  Valid CVEs that have no stage-1 candidates are
    reported separately.  Both cases remain visible in ``JoinStats`` instead
    of disappearing from the pipeline.
    """
    raw_contexts = context_document.get("contexts")
    if not isinstance(raw_contexts, list):
        raise ValueError("Context document must contain a list field named 'contexts'")

    candidates_by_cve = _candidate_index(candidate_records)
    contexts_by_cve: dict[str, Mapping[str, Any]] = {}
    unresolved: list[str] = []

    for raw_context in raw_contexts:
        if not isinstance(raw_context, Mapping):
            raise ValueError("Each context record must be a JSON object")
        raw_id = str(raw_context.get("cve_id") or "").strip()
        cve_id = normalize_cve_id(raw_id)
        if not cve_id.upper().startswith("CVE-"):
            unresolved.append(raw_id or "<missing>")
            continue
        if cve_id in contexts_by_cve:
            raise ValueError(f"Duplicate graph context after normalization: {cve_id}")
        contexts_by_cve[cve_id] = raw_context

    joined: list[dict[str, Any]] = []
    missing_candidates: list[str] = []
    for cve_id, raw_context in sorted(contexts_by_cve.items()):
        stage1_record = candidates_by_cve.get(cve_id)
        if stage1_record is None:
            missing_candidates.append(cve_id)
            continue

        record = deepcopy(dict(raw_context))
        record["cve_id"] = cve_id
        record["candidates"] = [candidate.to_dict() for candidate in stage1_record.candidates]
        record["stage1"] = {
            "domain": stage1_record.domain,
            "metadata": dict(stage1_record.metadata),
            "candidate_count": len(stage1_record.candidates),
        }
        joined.append(record)

    stats = JoinStats(
        context_records=len(raw_contexts),
        candidate_records=len(candidates_by_cve),
        matched=len(joined),
        missing_candidates=tuple(sorted(missing_candidates)),
        candidates_without_context=tuple(
            sorted(set(candidates_by_cve).difference(contexts_by_cve))
        ),
        unresolved_context_ids=tuple(sorted(unresolved)),
    )
    return joined, stats
