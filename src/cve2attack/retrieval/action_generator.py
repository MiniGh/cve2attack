"""Embed ATT&CK action units and aggregate their hits into Technique candidates."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from cve2attack.retrieval.action_kb import ACTION_CORPUS_VERSION, ActionDocument
from cve2attack.retrieval.embedder import Embedder, l2_normalize
from cve2attack.retrieval.generator import ProgressReporter, _format_duration, _report
from cve2attack.schemas import CandidateRecord, TechniqueCandidate


def action_cache_key(
    *,
    model_name: str,
    attack_bundle: Path,
    include_descriptions: bool,
    include_procedures: bool,
    source_types: Sequence[str] | None = None,
    min_chars: int,
    max_chars: int,
) -> str:
    """Return a versioned key that invalidates vectors when action rules change."""
    digest = hashlib.sha256()
    for value in (
        ACTION_CORPUS_VERSION,
        model_name,
        str(attack_bundle.resolve()),
        str(attack_bundle.stat().st_size),
        str(attack_bundle.stat().st_mtime_ns),
        str(include_descriptions),
        str(include_procedures),
        str(min_chars),
        str(max_chars),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    # Keep the historical cache key unchanged when no fine-grained filter is
    # requested, so the frozen full V5c corpus reuses its existing embeddings.
    if source_types is not None:
        digest.update(",".join(sorted(source_types)).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def load_or_create_action_embeddings(
    *,
    embedder: Embedder,
    actions: Sequence[ActionDocument],
    cache_path: Path,
    batch_size: int,
    superset_cache_paths: Sequence[Path] = (),
    progress: ProgressReporter | None = None,
) -> np.ndarray:
    """Load a compatible action cache or encode and atomically create one."""
    expected_ids = [item.action_id for item in actions]
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as loaded:
            cached_ids = [str(item) for item in loaded["action_ids"].tolist()]
            if cached_ids == expected_ids:
                embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
                _report(progress, f"action cache hit; actions={len(actions)}; path={cache_path}")
                return embeddings
        _report(progress, f"action cache is incompatible; rebuilding path={cache_path}")
    else:
        _report(progress, f"action cache miss; resolving vectors for actions={len(actions)}")

    # Fine-grained corpus ablations are exact subsets of the frozen full V5c
    # corpus. Their vectors can therefore be selected by action ID from the
    # model- and corpus-keyed full cache without re-encoding thousands of
    # unchanged texts. The exact subset is persisted under its own cache key.
    for superset_path in superset_cache_paths:
        if not superset_path.exists() or superset_path == cache_path:
            continue
        with np.load(superset_path, allow_pickle=False) as loaded:
            superset_ids = [str(item) for item in loaded["action_ids"].tolist()]
            superset_embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
        indices = {action_id: index for index, action_id in enumerate(superset_ids)}
        if not all(action_id in indices for action_id in expected_ids):
            continue
        embeddings = np.vstack(
            [superset_embeddings[indices[action_id]] for action_id in expected_ids]
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                embeddings=embeddings.astype(np.float32),
                action_ids=np.asarray(expected_ids, dtype=str),
            )
        temporary.replace(cache_path)
        _report(
            progress,
            f"action cache sliced from compatible full corpus; actions={len(actions)}; "
            f"source={superset_path}; path={cache_path}",
        )
        return embeddings

    _report(progress, f"no compatible action cache; encoding actions={len(actions)}")

    started_at = time.perf_counter()
    vectors: list[np.ndarray] = []
    total_batches = (len(actions) + batch_size - 1) // batch_size
    for start in range(0, len(actions), batch_size):
        batch = actions[start : start + batch_size]
        vectors.append(embedder.encode([item.text for item in batch], batch_size))
        completed = min(start + len(batch), len(actions))
        elapsed = time.perf_counter() - started_at
        rate = completed / elapsed if elapsed else 0.0
        remaining = len(actions) - completed
        eta = remaining / rate if rate else 0.0
        _report(
            progress,
            f"action embedding progress={completed}/{len(actions)}; "
            f"batch={start // batch_size + 1}/{total_batches}; "
            f"elapsed={_format_duration(elapsed)}; eta={_format_duration(eta)}",
        )
    embeddings = l2_normalize(np.vstack(vectors)) if vectors else np.empty((0, 0), dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            embeddings=embeddings.astype(np.float32),
            action_ids=np.asarray(expected_ids, dtype=str),
        )
    temporary.replace(cache_path)
    _report(progress, f"action cache saved; elapsed={_format_duration(time.perf_counter() - started_at)}; path={cache_path}")
    return embeddings


def aggregate_action_scores(
    *,
    scores: np.ndarray,
    actions: Sequence[ActionDocument],
    aggregation: str,
    aggregation_top_m: int,
    rank_constant: float,
    top_k: int,
    evidence_limit: int,
    evidence_text_limit: int,
    excluded_action_indices: set[int] | None = None,
) -> tuple[TechniqueCandidate, ...]:
    """Aggregate one query's action similarities into deterministic parent ranks."""
    values = np.asarray(scores, dtype=np.float32)
    if len(values) != len(actions):
        raise ValueError("Action score count does not match the action corpus")
    if aggregation not in {"max", "rank_rrf"}:
        raise ValueError(f"Unsupported action aggregation: {aggregation}")
    if aggregation_top_m <= 0:
        raise ValueError("aggregation_top_m must be positive")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    excluded = excluded_action_indices or set()
    action_indices: dict[str, list[int]] = {}
    for index, action in enumerate(actions):
        if index in excluded:
            continue
        action_indices.setdefault(action.technique_id, []).append(index)

    aggregated: dict[str, float] = {}
    if aggregation == "max":
        for technique_id, indices in action_indices.items():
            aggregated[technique_id] = max(float(values[index]) for index in indices)
    else:
        # Actions are pre-sorted by stable action_id, so mergesort gives a
        # deterministic action order for equal similarities.
        ranked_indices = np.argsort(-values, kind="mergesort")
        used: dict[str, int] = {}
        effective_rank = 0
        for raw_index in ranked_indices:
            if int(raw_index) in excluded:
                continue
            effective_rank += 1
            action = actions[int(raw_index)]
            count = used.get(action.technique_id, 0)
            if count >= aggregation_top_m:
                continue
            aggregated[action.technique_id] = aggregated.get(action.technique_id, 0.0) + (
                1.0 / (rank_constant + effective_rank)
            )
            used[action.technique_id] = count + 1

    best_similarity = {
        technique_id: max(float(values[index]) for index in indices)
        for technique_id, indices in action_indices.items()
    }
    ranked_techniques = sorted(
        aggregated,
        key=lambda technique_id: (
            -aggregated[technique_id],
            -best_similarity[technique_id],
            technique_id,
        ),
    )[: min(max(0, top_k), len(aggregated))]

    candidates: list[TechniqueCandidate] = []
    for technique_id in ranked_techniques:
        indices = sorted(
            action_indices[technique_id],
            key=lambda index: (-float(values[index]), actions[index].action_id),
        )
        evidence = []
        for index in indices[: max(0, evidence_limit)]:
            action = actions[index]
            text = action.text
            if evidence_text_limit > 0 and len(text) > evidence_text_limit:
                text = text[:evidence_text_limit].rstrip()
            evidence.append(
                {
                    "action_id": action.action_id,
                    "source_type": action.source_type,
                    "similarity": round(float(values[index]), 6),
                    "text": text,
                }
            )
        representative = actions[indices[0]]
        candidates.append(
            TechniqueCandidate(
                technique_id=technique_id,
                score=aggregated[technique_id],
                sources=("action_embedding",),
                metadata={
                    "name": representative.technique_name,
                    "tactics": list(representative.tactics),
                    "retrieval_corpus": "action",
                    "aggregation": aggregation,
                    "best_action_similarity": round(best_similarity[technique_id], 6),
                    "corpus_action_count": len(indices),
                    "action_evidence": evidence,
                },
            )
        )
    return tuple(candidates)


def retrieve_action_candidates(
    *,
    queries: Mapping[str, str],
    actions: Sequence[ActionDocument],
    action_embeddings: np.ndarray,
    embedder: Embedder,
    top_k: int,
    batch_size: int,
    aggregation: str,
    aggregation_top_m: int = 3,
    rank_constant: float = 60.0,
    evidence_limit: int = 3,
    evidence_text_limit: int = 400,
    exclude_query_cve_actions: bool = True,
    domain: str = "Enterprise",
    progress: ProgressReporter | None = None,
) -> list[CandidateRecord]:
    """Retrieve action units, aggregate them, and report per-batch progress."""
    identifiers = sorted(queries)
    if not identifiers:
        return []
    if not actions:
        raise ValueError("Action corpus is empty")
    if action_embeddings.shape[0] != len(actions):
        raise ValueError("Action embedding count does not match the action corpus")

    records: list[CandidateRecord] = []
    excluded_by_cve: dict[str, set[int]] = {}
    if exclude_query_cve_actions:
        # Build this inverted index once. Re-scanning the full action corpus
        # for every query is unnecessary and becomes expensive on large KEV
        # benchmark views.
        for index, action in enumerate(actions):
            for cve_id in action.vulnerability_ids:
                excluded_by_cve.setdefault(cve_id, set()).add(index)
    total_batches = (len(identifiers) + batch_size - 1) // batch_size
    started_at = time.perf_counter()
    _report(
        progress,
        f"retrieving action candidates; queries={len(identifiers)}; actions={len(actions)}; "
        f"aggregation={aggregation}; exclude_query_cve_actions={exclude_query_cve_actions}; "
        f"top_k={top_k}; batches={total_batches}",
    )
    for start in range(0, len(identifiers), batch_size):
        batch_ids = identifiers[start : start + batch_size]
        vectors = l2_normalize(embedder.encode([queries[cve_id] for cve_id in batch_ids], batch_size))
        batch_scores = vectors @ action_embeddings.T
        for row_index, cve_id in enumerate(batch_ids):
            excluded_indices = excluded_by_cve.get(cve_id.upper(), set())
            candidates = aggregate_action_scores(
                scores=batch_scores[row_index],
                actions=actions,
                aggregation=aggregation,
                aggregation_top_m=aggregation_top_m,
                rank_constant=rank_constant,
                top_k=top_k,
                evidence_limit=evidence_limit,
                evidence_text_limit=evidence_text_limit,
                excluded_action_indices=excluded_indices,
            )
            records.append(
                CandidateRecord(
                    cve_id=cve_id,
                    domain=domain,
                    candidates=candidates,
                    metadata={"excluded_query_cve_actions": len(excluded_indices)},
                )
            )
        completed = min(start + len(batch_ids), len(identifiers))
        elapsed = time.perf_counter() - started_at
        rate = completed / elapsed if elapsed else 0.0
        remaining = len(identifiers) - completed
        eta = remaining / rate if rate else 0.0
        _report(
            progress,
            f"action candidate progress={completed}/{len(identifiers)}; "
            f"batch={start // batch_size + 1}/{total_batches}; "
            f"elapsed={_format_duration(elapsed)}; eta={_format_duration(eta)}",
        )
    return records
