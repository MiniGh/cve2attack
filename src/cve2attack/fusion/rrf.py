"""Reciprocal Rank Fusion for completed candidate-generation runs.

RRF combines rank positions rather than raw similarity scores, so sources with
different score scales can be fused without supervised calibration.  This
module writes a normal auditable run and deliberately separates the internal
source depth from the final controlled candidate budget.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cve2attack.config import PROJECT_ROOT
from cve2attack.data.loaders import benchmark_truth, candidate_records, write_candidate_records
from cve2attack.evaluation.metrics import evaluate
from cve2attack.evaluation.report import write_run_report
from cve2attack.schemas import CandidateRecord, TechniqueCandidate, records_by_id


ProgressCallback = Callable[[str], None]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_").lower()


def _git_commit(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _source_name(run_dir: Path) -> tuple[str, dict[str, Any]]:
    """Return a stable source name and the source manifest when available."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return run_dir.name, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"RRF input run is not complete: {run_dir}")
    name = str(manifest.get("experiment") or manifest.get("run_id") or run_dir.name)
    return name, manifest


def fuse_rrf_records(
    source_records: Mapping[str, Sequence[CandidateRecord]],
    *,
    cohort: Sequence[str],
    top_k: int,
    source_depth: int,
    rank_constant: float,
    weights: Mapping[str, float] | None = None,
    progress: ProgressCallback | None = None,
) -> list[CandidateRecord]:
    """Fuse source rankings and return at most ``top_k`` candidates per CVE.

    Only the first ``source_depth`` candidates from each source can contribute.
    Ties are resolved without labels: better best-source rank, then more source
    votes, then Technique ID.  Every output candidate records its contributing
    source ranks and individual RRF contributions for later inspection.
    """
    if not source_records:
        raise ValueError("RRF requires at least one source")
    if top_k <= 0:
        raise ValueError("RRF top_k must be positive")
    if source_depth <= 0:
        raise ValueError("RRF source_depth must be positive")
    if rank_constant <= 0:
        raise ValueError("RRF rank_constant must be positive")

    source_names = list(source_records)
    resolved_weights = {
        name: float(weights[name]) if weights and name in weights else 1.0
        for name in source_names
    }
    if any(value <= 0 for value in resolved_weights.values()):
        raise ValueError("Every RRF source weight must be positive")
    indexed = {name: records_by_id(records) for name, records in source_records.items()}

    fused: list[CandidateRecord] = []
    total = len(cohort)
    progress_step = max(1, min(10, total // 5 if total else 1))
    started_at = time.perf_counter()
    for index, cve_id in enumerate(cohort, start=1):
        aggregate: dict[str, dict[str, Any]] = {}
        domain: str | None = None
        for source_name in source_names:
            record = indexed[source_name][cve_id]
            domain = domain or record.domain
            weight = resolved_weights[source_name]
            for rank, candidate in enumerate(record.candidates[:source_depth], start=1):
                contribution = weight / (rank_constant + rank)
                item = aggregate.setdefault(
                    candidate.technique_id,
                    {
                        "score": 0.0,
                        "source_ranks": {},
                        "contributions": {},
                    },
                )
                item["score"] += contribution
                item["source_ranks"][source_name] = rank
                item["contributions"][source_name] = contribution

        ordered = sorted(
            aggregate.items(),
            key=lambda pair: (
                -float(pair[1]["score"]),
                min(pair[1]["source_ranks"].values()),
                -len(pair[1]["source_ranks"]),
                pair[0],
            ),
        )
        candidates = tuple(
            TechniqueCandidate(
                technique_id=technique_id,
                score=float(item["score"]),
                sources=("rrf",),
                metadata={
                    "source_ranks": dict(item["source_ranks"]),
                    "rrf_contributions": {
                        name: round(float(value), 10)
                        for name, value in item["contributions"].items()
                    },
                },
            )
            for technique_id, item in ordered[:top_k]
        )
        fused.append(
            CandidateRecord(
                cve_id=cve_id,
                candidates=candidates,
                domain=domain,
                metadata={
                    "fusion": "rrf",
                    "rank_constant": rank_constant,
                    "source_depth": source_depth,
                    "source_count": len(source_names),
                },
            )
        )
        if progress and (index == total or index % progress_step == 0):
            elapsed = time.perf_counter() - started_at
            progress(f"fusion progress={index}/{total}; elapsed={elapsed:.1f}s")
    return fused


def run_rrf_fusion(
    run_dirs: Sequence[Path],
    *,
    run_id: str,
    benchmark_name: str,
    top_k: int = 20,
    source_depth: int = 50,
    rank_constant: float = 60.0,
    weights: Sequence[float] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Fuse completed runs and persist a standard run directory and manifest."""
    if not run_dirs:
        raise ValueError("RRF requires at least one input run")
    if weights is not None and len(weights) != len(run_dirs):
        raise ValueError("--weights must provide exactly one value per input run")

    run_dir = project_root / "runs" / _slug(run_id)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    benchmark_dir = project_root / "data" / "benchmarks" / benchmark_name
    truth = benchmark_truth(benchmark_dir)
    if not truth:
        raise RuntimeError(f"Benchmark has no records: {benchmark_name}")

    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    started_at = time.perf_counter()

    def report(message: str) -> None:
        print(f"[rrf] {message}", flush=True)

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "experiment": "rrf_fusion",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": _git_commit(project_root),
        "status": "running",
        "benchmark": benchmark_name,
        "fusion": {
            "strategy": "reciprocal_rank_fusion",
            "top_k": top_k,
            "source_depth": source_depth,
            "rank_constant": rank_constant,
            "weights": list(weights) if weights is not None else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        report(
            f"starting run_id={run_dir.name}; benchmark={benchmark_name}; "
            f"sources={len(run_dirs)}; source_depth={source_depth}; top_k={top_k}; "
            f"rank_constant={rank_constant:g}"
        )
        source_records: dict[str, list[CandidateRecord]] = {}
        source_details: list[dict[str, Any]] = []
        weight_map: dict[str, float] = {}
        cohort = sorted(truth)
        for index, raw_path in enumerate(run_dirs, start=1):
            source_dir = raw_path if raw_path.is_absolute() else project_root / raw_path
            source_name, source_manifest = _source_name(source_dir)
            if source_name in source_records:
                raise ValueError(f"Duplicate RRF source name: {source_name}")
            records = candidate_records(source_dir)
            indexed = records_by_id(records)
            missing = [cve_id for cve_id in cohort if cve_id not in indexed]
            shallow = [
                cve_id
                for cve_id in cohort
                if cve_id in indexed and len(indexed[cve_id].candidates) < source_depth
            ]
            if missing or shallow:
                raise ValueError(
                    f"RRF source {source_name} is not comparable: "
                    f"missing_cves={len(missing)}, rankings_shorter_than_{source_depth}={len(shallow)}"
                )
            source_records[source_name] = [indexed[cve_id] for cve_id in cohort]
            weight = float(weights[index - 1]) if weights is not None else 1.0
            weight_map[source_name] = weight
            try:
                stored_path = str(source_dir.relative_to(project_root))
            except ValueError:
                stored_path = str(source_dir)
            source_details.append(
                {
                    "name": source_name,
                    "path": stored_path,
                    "weight": weight,
                    "run_id": source_manifest.get("run_id"),
                    "experiment": source_manifest.get("experiment"),
                    "git_commit": source_manifest.get("git_commit"),
                    "resolved_config": source_manifest.get("resolved_config"),
                }
            )
            report(
                f"source {index}/{len(run_dirs)} loaded: {source_name}; "
                f"coverage={len(cohort)}/{len(cohort)}; weight={weight:g}"
            )

        records = fuse_rrf_records(
            source_records,
            cohort=cohort,
            top_k=top_k,
            source_depth=source_depth,
            rank_constant=rank_constant,
            weights=weight_map,
            progress=report,
        )
        candidate_paths = write_candidate_records(records, run_dir / "candidates")
        metrics = evaluate(records, truth)
        metrics_payload = {benchmark_name: metrics.to_dict()}
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_run_report(
            run_dir / "report.md",
            experiment_name="rrf_fusion",
            metrics={benchmark_name: metrics},
        )
        manifest.update(
            {
                "status": "complete",
                "input_runs": source_details,
                "benchmark_cves": len(cohort),
                "candidate_records": len(records),
                "candidate_files": [path.name for path in candidate_paths],
                "metrics": metrics_payload,
            }
        )
        report(f"metrics={json.dumps(metrics_payload, ensure_ascii=False)}")
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        report(f"failed; error={type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report(f"complete; elapsed={manifest['elapsed_seconds']}s; output={run_dir}")
    return run_dir
