"""Reusable candidate retrieval and technique-embedding cache."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from cve2attack.retrieval.embedder import Embedder, l2_normalize
from cve2attack.retrieval.technique_kb import TechniqueDocument
from cve2attack.schemas import CandidateRecord, TechniqueCandidate


ProgressReporter = Callable[[str], None]


def _report(progress: ProgressReporter | None, message: str) -> None:
    """Send progress to a caller, or print a useful default terminal message."""
    if progress is None:
        print(f"[retrieval] {message}", flush=True)
    else:
        progress(message)


def _format_duration(seconds: float) -> str:
    """Render a duration without introducing a logging dependency."""
    whole_seconds = max(0, int(round(seconds)))
    minutes, seconds_part = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds_part:02d}s"
    return f"{seconds_part}s"


def cache_key(
    *,
    model_name: str,
    attack_bundle: Path,
    include_procedures: bool,
    procedure_char_limit: int,
) -> str:
    """Return a corpus-aware key so incompatible technique vectors are never reused."""
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
    progress: ProgressReporter | None = None,
) -> np.ndarray:
    """Load a compatible technique cache or encode and atomically create one."""
    expected_ids = [item.technique_id for item in techniques]
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as loaded:
            cached_ids = [str(item) for item in loaded["technique_ids"].tolist()]
            if cached_ids == expected_ids:
                embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
                _report(
                    progress,
                    f"technique cache hit; techniques={len(expected_ids)}; path={cache_path}",
                )
                return embeddings
        _report(progress, f"technique cache is incompatible; rebuilding path={cache_path}")
    else:
        _report(progress, f"technique cache miss; encoding techniques={len(expected_ids)}")

    embeddings = l2_normalize(embedder.encode([item.text for item in techniques], batch_size))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings.astype(np.float32),
        technique_ids=np.asarray(expected_ids, dtype=str),
    )
    _report(progress, f"technique cache saved; path={cache_path}")
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
    progress: ProgressReporter | None = None,
) -> list[CandidateRecord]:
    """Rank ATT&CK techniques for every query and report per-batch completion."""
    identifiers = sorted(queries)
    if not identifiers:
        return []

    records: list[CandidateRecord] = []
    total_batches = (len(identifiers) + batch_size - 1) // batch_size
    started_at = time.perf_counter()
    _report(
        progress,
        f"retrieving candidates; queries={len(identifiers)}; top_k={top_k}; batches={total_batches}",
    )
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
        completed = min(start + len(batch_ids), len(identifiers))
        elapsed = time.perf_counter() - started_at
        rate = completed / elapsed if elapsed else 0.0
        remaining = len(identifiers) - completed
        eta = remaining / rate if rate else 0.0
        _report(
            progress,
            f"candidate progress={completed}/{len(identifiers)}; "
            f"batch={start // batch_size + 1}/{total_batches}; "
            f"elapsed={_format_duration(elapsed)}; eta={_format_duration(eta)}",
        )
    return records
