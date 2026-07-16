"""Config-driven end-to-end stage-1 candidate generation."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from cve2attack.config import PROJECT_ROOT, load_experiment, project_path
from cve2attack.data.loaders import (
    CVERepository,
    benchmark_truth,
    enterprise_cve_ids,
    load_json_mapping,
    write_candidate_records,
)
from cve2attack.evaluation.metrics import EvaluationMetrics, evaluate
from cve2attack.evaluation.report import write_run_report
from cve2attack.fusion.structured_chain import fuse_records
from cve2attack.retrieval.embedder import SentenceTransformerEmbedder
from cve2attack.retrieval.generator import (
    cache_key,
    load_or_create_technique_embeddings,
    retrieve_candidates,
)
from cve2attack.retrieval.technique_kb import load_technique_documents


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def make_run_id(experiment_name: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(experiment_name)}"


def select_input_ids(config: Mapping[str, Any], project_root: Path) -> list[str]:
    input_config = config["input"]
    if input_config["mode"] == "benchmark":
        benchmark = str(input_config["benchmark"])
        truth = benchmark_truth(project_root / "data" / "benchmarks" / benchmark)
        return sorted(truth)
    return enterprise_cve_ids(project_root / "data" / "derived" / "domain_mapping")


def build_queries(
    *,
    config: Mapping[str, Any],
    cve_ids: list[str],
    repository: CVERepository,
    project_root: Path,
) -> tuple[dict[str, str], dict[str, int]]:
    strategy = config["query"]["strategy"]
    missing_description = missing_rewrite = 0
    queries: dict[str, str] = {}
    rewrites: dict[str, str] = {}
    if strategy == "rewrite_cache":
        benchmark_name = str(config["input"].get("benchmark", "full_enterprise"))
        cache_value = str(config["query"]["cache"]).format(benchmark=benchmark_name)
        rewrites = load_json_mapping(project_path(cache_value, project_root))

    for cve_id in cve_ids:
        if strategy == "rewrite_cache":
            text = rewrites.get(cve_id)
            if not text:
                missing_rewrite += 1
                continue
        else:
            text = repository.description(cve_id)
            if not text:
                missing_description += 1
                continue
        queries[cve_id] = text
    return queries, {
        "selected_cves": len(cve_ids),
        "query_cves": len(queries),
        "missing_description": missing_description,
        "missing_rewrite": missing_rewrite,
    }


def run_experiment(
    config_path: Path,
    *,
    run_id: str | None = None,
    max_cves: int | None = None,
    benchmark: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    config = load_experiment(config_path)
    if benchmark is not None:
        config["input"] = {**config["input"], "mode": "benchmark", "benchmark": benchmark}
    identifier = run_id or make_run_id(str(config["name"]))
    run_dir = project_root / "runs" / _slug(identifier)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    candidate_dir = run_dir / "candidates"
    run_dir.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "experiment": config["name"],
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": _git_commit(project_root),
        "config_file": str(config_path),
        "resolved_config": config,
        "status": "running",
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        cve_ids = select_input_ids(config, project_root)
        if max_cves is not None:
            cve_ids = cve_ids[: max(0, max_cves)]
        repository = CVERepository(project_root / "data" / "raw" / "cve")
        queries, coverage = build_queries(
            config=config,
            cve_ids=cve_ids,
            repository=repository,
            project_root=project_root,
        )
        if not queries:
            raise RuntimeError("No query text is available for the selected CVEs")

        document_config = config["technique_document"]
        attack_bundle = project_root / "data" / "knowledge" / "enterprise-attack.json"
        techniques = load_technique_documents(
            attack_bundle,
            include_procedures=bool(document_config["include_procedures"]),
            procedure_char_limit=int(document_config["procedure_char_limit"]),
        )

        retrieval_config = config["retrieval"]
        embedder = SentenceTransformerEmbedder(str(retrieval_config["model"]))
        key = cache_key(
            model_name=embedder.model_name,
            attack_bundle=attack_bundle,
            include_procedures=bool(document_config["include_procedures"]),
            procedure_char_limit=int(document_config["procedure_char_limit"]),
        )
        cache_path = project_root / "data" / "derived" / "embedding_cache" / f"techniques_{key}.npz"
        embeddings = load_or_create_technique_embeddings(
            embedder=embedder,
            techniques=techniques,
            cache_path=cache_path,
            batch_size=int(retrieval_config["batch_size"]),
        )
        records = retrieve_candidates(
            queries=queries,
            techniques=techniques,
            technique_embeddings=embeddings,
            embedder=embedder,
            top_k=int(retrieval_config["top_k"]),
            batch_size=int(retrieval_config["batch_size"]),
        )

        fusion_config = config["fusion"]
        if fusion_config["strategy"] == "structured_chain":
            records = fuse_records(
                records,
                chain_file=project_path(fusion_config["chain_file"], project_root),
                cwe_xml=project_path(fusion_config["cwe_xml"], project_root),
                alpha=float(fusion_config["alpha"]),
                fanout_threshold=int(fusion_config["fanout_threshold"]),
                top_k=int(retrieval_config["top_k"]),
            )

        written = write_candidate_records(records, candidate_dir)
        benchmark_metrics: dict[str, EvaluationMetrics] = {}
        benchmark_names = list(config["evaluation"].get("benchmarks", []))
        benchmark_names = [
            config["input"].get("benchmark") if name == "input" else name
            for name in benchmark_names
        ]
        for benchmark_name in benchmark_names:
            if not benchmark_name:
                continue
            benchmark_dir = project_root / "data" / "benchmarks" / str(benchmark_name)
            if benchmark_dir.is_dir():
                benchmark_metrics[str(benchmark_name)] = evaluate(records, benchmark_truth(benchmark_dir))

        metrics_payload = {name: value.to_dict() for name, value in benchmark_metrics.items()}
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_run_report(
            run_dir / "report.md",
            experiment_name=str(config["name"]),
            metrics=benchmark_metrics,
        )
        manifest.update(
            {
                "status": "complete",
                "input_coverage": coverage,
                "technique_count": len(techniques),
                "candidate_records": len(records),
                "candidate_files": [path.name for path in written],
                "embedding_cache": str(cache_path.relative_to(project_root)),
            }
        )
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir
