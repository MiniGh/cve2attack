"""Reusable candidate retrieval and technique-embedding cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from cve2attack.retrieval.embedder import Embedder, l2_normalize
from cve2attack.retrieval.technique_kb import TechniqueDocument
from cve2attack.schemas import CandidateRecord, TechniqueCandidate


def cache_key(
    *,
    model_name: str,
    attack_bundle: Path,
    include_procedures: bool,
    procedure_char_limit: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(str(attack_bundle.resolve()).encode("utf-8"))
    digest.update(str(attack_bundle.stat().st_size).encode("ascii"))
    digest.update(str(attack_bundle.stat().st_mtime_ns).encode("ascii"))
    digest.update(str(include_procedures).encode("ascii"))
    digest.update(str(procedure_char_limit).encode("ascii"))
    return digest.hexdigest()[:16]


def load_or_create_technique_embeddings(
    *,
    embedder: Embedder,
    techniques: Sequence[TechniqueDocument],
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    expected_ids = [item.technique_id for item in techniques]
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=False)
        cached_ids = [str(item) for item in loaded["technique_ids"].tolist()]
        if cached_ids == expected_ids:
            return np.asarray(loaded["embeddings"], dtype=np.float32)

    embeddings = l2_normalize(embedder.encode([item.text for item in techniques], batch_size))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings.astype(np.float32),
        technique_ids=np.asarray(expected_ids, dtype=str),
    )
    return embeddings


def retrieve_candidates(
    *,
    queries: Mapping[str, str],
    techniques: Sequence[TechniqueDocument],
    technique_embeddings: np.ndarray,
    embedder: Embedder,
    top_k: int,
    batch_size: int,
    domain: str = "Enterprise",
) -> list[CandidateRecord]:
    identifiers = sorted(queries)
    if not identifiers:
        return []

    records: list[CandidateRecord] = []
    for start in range(0, len(identifiers), batch_size):
        batch_ids = identifiers[start : start + batch_size]
        vectors = l2_normalize(embedder.encode([queries[cve_id] for cve_id in batch_ids], batch_size))
        scores = vectors @ technique_embeddings.T

        for row_index, cve_id in enumerate(batch_ids):
            row = scores[row_index]
            k = min(max(0, top_k), len(row))
            if k == 0:
                candidates: tuple[TechniqueCandidate, ...] = ()
            else:
                indices = np.argpartition(row, -k)[-k:]
                indices = indices[np.argsort(row[indices])[::-1]]
                candidates = tuple(
                    TechniqueCandidate(
                        technique_id=techniques[int(index)].technique_id,
                        score=float(row[index]),
                        sources=("embedding",),
                        metadata={
                            "name": techniques[int(index)].name,
                            "tactics": list(techniques[int(index)].tactics),
                        },
                    )
                    for index in indices
                )
            records.append(CandidateRecord(cve_id=cve_id, domain=domain, candidates=candidates))
    return records
