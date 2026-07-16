"""Historical CWE → CAPEC → ATT&CK fusion, adapted to the canonical schema."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cve2attack.schemas import CandidateRecord, TechniqueCandidate, parent_technique_id


def load_cwe_abstractions(path: Path) -> set[str]:
    namespace = {"cwe": "http://cwe.mitre.org/cwe-7"}
    allowed = {"Base", "Variant"}
    root = ET.parse(path).getroot()
    return {
        str(weakness.get("ID"))
        for weakness in root.iterfind("cwe:Weaknesses/cwe:Weakness", namespace)
        if weakness.get("ID") and weakness.get("Abstraction") in allowed
    }


def load_chain(path: Path) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            result[str(record["cve_id"])] = {
                "cwes": [str(item) for item in record.get("cwes", [])],
                "capecs": [str(item) for item in record.get("capecs", [])],
                "techniques": [parent_technique_id(str(item)) for item in record.get("techniques", [])],
            }
    return result


def capec_fanout(chain: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, int]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for record in chain.values():
        for capec in record.get("capecs", []):
            mapping[str(capec)].update(parent_technique_id(item) for item in record.get("techniques", []))
    return {capec: len(techniques) for capec, techniques in mapping.items()}


def chain_scores(
    cve_id: str,
    *,
    chain: Mapping[str, Mapping[str, Sequence[str]]],
    valid_cwes: set[str],
    capec_counts: Mapping[str, int],
) -> tuple[dict[str, float], int]:
    record = chain.get(cve_id)
    if not record or not any(cwe in valid_cwes for cwe in record.get("cwes", [])):
        return {}, 0

    techniques = {parent_technique_id(item) for item in record.get("techniques", []) if item}
    capecs = list(record.get("capecs", []))
    cwe_fanout = len(capecs) or 1
    average_capec_fanout = (
        sum(capec_counts.get(capec, 1) for capec in capecs) / len(capecs)
        if capecs
        else 1.0
    )
    score = 1.0 / (cwe_fanout * average_capec_fanout)
    return {technique_id: score for technique_id in techniques}, len(techniques)


def fuse_records(
    records: Sequence[CandidateRecord],
    *,
    chain_file: Path,
    cwe_xml: Path,
    alpha: float,
    fanout_threshold: int,
    top_k: int,
) -> list[CandidateRecord]:
    chain = load_chain(chain_file)
    valid_cwes = load_cwe_abstractions(cwe_xml)
    capec_counts = capec_fanout(chain)
    fused_records: list[CandidateRecord] = []

    for record in records:
        contributions, fanout = chain_scores(
            record.cve_id,
            chain=chain,
            valid_cwes=valid_cwes,
            capec_counts=capec_counts,
        )
        retrieval = {candidate.technique_id: candidate for candidate in record.candidates}
        scores: dict[str, float] = {
            technique_id: float(candidate.score or 0.0)
            for technique_id, candidate in retrieval.items()
        }
        lowest = min(scores.values()) if scores else 0.0
        beta = lowest / 2.0

        for technique_id, contribution in contributions.items():
            if technique_id in scores:
                scores[technique_id] += alpha * contribution
            elif fanout <= fanout_threshold:
                scores[technique_id] = beta + alpha * contribution

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        candidates: list[TechniqueCandidate] = []
        for technique_id, score in ordered:
            existing = retrieval.get(technique_id)
            sources = ["embedding"] if existing else []
            if technique_id in contributions:
                sources.append("structured_chain")
            metadata = dict(existing.metadata) if existing else {}
            if technique_id in contributions:
                metadata["chain_score"] = contributions[technique_id]
            candidates.append(
                TechniqueCandidate(
                    technique_id=technique_id,
                    score=score,
                    sources=tuple(sources),
                    metadata=metadata,
                )
            )
        fused_records.append(
            CandidateRecord(
                cve_id=record.cve_id,
                domain=record.domain,
                candidates=tuple(candidates),
                metadata={**dict(record.metadata), "fusion": "structured_chain"},
            )
        )
    return fused_records
