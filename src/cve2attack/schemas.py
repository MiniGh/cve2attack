"""Canonical candidate schema plus adapters for historical output formats."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0"


def parent_technique_id(value: str) -> str:
    """Normalize an ATT&CK technique ID and roll sub-techniques up to parents."""
    tech_id = str(value).strip().upper()
    if tech_id and not tech_id.startswith("T"):
        tech_id = f"T{tech_id}"
    return tech_id.split(".", 1)[0]


@dataclass(frozen=True)
class TechniqueCandidate:
    technique_id: str
    score: float | None = None
    sources: tuple[str, ...] = ("embedding",)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "technique_id": self.technique_id,
            "sources": list(self.sources),
        }
        if self.score is not None:
            payload["score"] = round(float(self.score), 6)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class CandidateRecord:
    cve_id: str
    candidates: tuple[TechniqueCandidate, ...]
    domain: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "cve_id": self.cve_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
        if self.domain:
            payload["domain"] = self.domain
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "CandidateRecord":
        """Read canonical records and both historical `techniques` variants."""
        raw_candidates: Sequence[Any]
        if isinstance(record.get("candidates"), list):
            raw_candidates = record["candidates"]
        elif isinstance(record.get("techniques"), list):
            raw_candidates = record["techniques"]
        else:
            raw_candidates = []

        candidates: list[TechniqueCandidate] = []
        seen: set[str] = set()
        for raw in raw_candidates:
            candidate = parse_candidate(raw)
            if candidate is None or candidate.technique_id in seen:
                continue
            candidates.append(candidate)
            seen.add(candidate.technique_id)

        return cls(
            cve_id=str(record["cve_id"]),
            domain=str(record["domain"]) if record.get("domain") else None,
            candidates=tuple(candidates),
            metadata=record.get("metadata", {}) if isinstance(record.get("metadata"), Mapping) else {},
        )


def parse_candidate(raw: Any) -> TechniqueCandidate | None:
    if isinstance(raw, str):
        technique_id = parent_technique_id(raw)
        return TechniqueCandidate(technique_id, score=None, sources=("legacy",)) if technique_id else None
    if not isinstance(raw, Mapping):
        return None

    identifier = raw.get("technique_id") or raw.get("tech_id") or raw.get("id")
    if not identifier:
        return None
    technique_id = parent_technique_id(str(identifier))
    raw_score = raw.get("score")
    score = float(raw_score) if raw_score is not None else None
    raw_sources = raw.get("sources")
    sources = tuple(str(item) for item in raw_sources) if isinstance(raw_sources, list) else ("legacy",)
    metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata"), Mapping) else {}
    return TechniqueCandidate(technique_id, score=score, sources=sources, metadata=metadata)


def technique_ids(record: CandidateRecord, limit: int | None = None) -> list[str]:
    values = [candidate.technique_id for candidate in record.candidates]
    return values if limit is None else values[:limit]


def records_by_id(records: Iterable[CandidateRecord]) -> dict[str, CandidateRecord]:
    return {record.cve_id: record for record in records}
