"""Config-driven end-to-end stage-1 candidate generation."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

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
    """Create a timestamped directory name when the caller does not provide one."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(experiment_name)}"


def select_input_ids(config: Mapping[str, Any], project_root: Path) -> list[str]:
    """Select the fixed evaluation cohort or all enterprise CVEs for one experiment."""
    input_config = config["input"]
    if input_config["mode"] == "benchmark":
        benchmark = str(input_config["benchmark"])
        truth = benchmark_truth(project_root / "data" / "benchmarks" / benchmark)
        return sorted(truth)
    return enterprise_cve_ids(project_root / "data" / "derived" / "domain_mapping")


def resolve_attack_bundle(config: Mapping[str, Any], project_root: Path) -> Path:
    """Select the versioned ATT&CK corpus required by an experiment input.

    A benchmark may declare ``technique_corpus.path`` in its ``dataset.yaml``.
    This keeps a frozen benchmark's gold labels and retrieval corpus on the
    same ATT&CK release.  An explicit experiment-level path takes precedence
    for deliberately version-migration experiments.
    """
    technique_config = config.get("technique_document", {})
    explicit_path = technique_config.get("attack_bundle")
    if explicit_path:
        bundle = project_path(str(explicit_path), project_root)
    elif config["input"]["mode"] == "benchmark":
        benchmark = str(config["input"]["benchmark"])
        metadata_path = project_root / "data" / "benchmarks" / benchmark / "dataset.yaml"
        corpus_path: str | None = None
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = yaml.safe_load(handle) or {}
            if not isinstance(metadata, Mapping):
                raise ValueError(f"Benchmark metadata must be a mapping: {metadata_path}")
            corpus = metadata.get("technique_corpus")
            if isinstance(corpus, Mapping):
                value = corpus.get("path")
                corpus_path = str(value) if value else None
            elif corpus:
                corpus_path = str(corpus)
        bundle = (
            project_path(corpus_path, project_root)
            if corpus_path
            else project_root / "data" / "knowledge" / "enterprise-attack.json"
        )
    else:
        bundle = project_root / "data" / "knowledge" / "enterprise-attack.json"
    if not bundle.is_file():
        raise FileNotFoundError(f"ATT&CK technique corpus does not exist: {bundle}")
    return bundle


def build_queries(
    *,
    config: Mapping[str, Any],
    cve_ids: list[str],
    repository: CVERepository,
    project_root: Path,
) -> tuple[dict[str, str], dict[str, int]]:
    """Resolve raw descriptions or cached rewrites and account for every omission."""
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
    """Execute candidate generation and persist a self-contained, auditable run."""
    started_at = time.perf_counter()
    config = load_experiment(config_path)
    if benchmark is not None:
        config["input"] = {**config["input"], "mode": "benchmark", "benchmark": benchmark}
    identifier = run_id or make_run_id(str(config["name"]))
    run_dir = project_root / "runs" / _slug(identifier)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    candidate_dir = run_dir / "candidates"
    run_dir.mkdir(parents=True)

    def report(message: str) -> None:
        """Emit a flushed status line suitable for an interactive SSH terminal."""
        print(f"[run] {message}", flush=True)

    report(
        f"starting experiment={config['name']}; run_id={run_dir.name}; "
        f"input={config['input'].get('benchmark', config['input'].get('mode'))}"
    )

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
        report(f"selected_cves={len(cve_ids)}")
        repository = CVERepository(project_root / "data" / "raw" / "cve")
        queries, coverage = build_queries(
            config=config,
            cve_ids=cve_ids,
            repository=repository,
            project_root=project_root,
        )
        if not queries:
            raise RuntimeError("No query text is available for the selected CVEs")
        report(
            f"query coverage: available={coverage['query_cves']}/{coverage['selected_cves']}; "
            f"missing_description={coverage['missing_description']}; "
            f"missing_rewrite={coverage['missing_rewrite']}"
        )

        document_config = config["technique_document"]
        attack_bundle = resolve_attack_bundle(config, project_root)
        techniques = load_technique_documents(
            attack_bundle,
            include_procedures=bool(document_config["include_procedures"]),
            procedure_char_limit=int(document_config["procedure_char_limit"]),
        )
        report(
            f"ATT&CK corpus={attack_bundle}; techniques={len(techniques)}; "
            f"include_procedures={bool(document_config['include_procedures'])}"
        )

        retrieval_config = config["retrieval"]
        report(
            f"retrieval model={retrieval_config['model']}; top_k={retrieval_config['top_k']}; "
            f"batch_size={retrieval_config['batch_size']}"
        )
        embedder = SentenceTransformerEmbedder(
            str(retrieval_config["model"]),
            local_files_only=bool(retrieval_config.get("local_files_only", True)),
        )
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
            progress=report,
        )
        records = retrieve_candidates(
            queries=queries,
            techniques=techniques,
            technique_embeddings=embeddings,
            embedder=embedder,
            top_k=int(retrieval_config["top_k"]),
            batch_size=int(retrieval_config["batch_size"]),
            progress=report,
        )

        fusion_config = config["fusion"]
        if fusion_config["strategy"] == "structured_chain":
            report("applying structured-chain score fusion")
            records = fuse_records(
                records,
                chain_file=project_path(fusion_config["chain_file"], project_root),
                cwe_xml=project_path(fusion_config["cwe_xml"], project_root),
                alpha=float(fusion_config["alpha"]),
                fanout_threshold=int(fusion_config["fanout_threshold"]),
                top_k=int(retrieval_config["top_k"]),
            )

        written = write_candidate_records(records, candidate_dir)
        report(f"candidate files written={len(written)}; path={candidate_dir}")
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
        report(f"metrics={json.dumps(metrics_payload, ensure_ascii=False)}")
        write_run_report(
            run_dir / "report.md",
            experiment_name=str(config["name"]),
            metrics=benchmark_metrics,
        )
        try:
            technique_corpus = str(attack_bundle.relative_to(project_root))
        except ValueError:
            technique_corpus = str(attack_bundle)
        manifest.update(
            {
                "status": "complete",
                "input_coverage": coverage,
                "technique_count": len(techniques),
                "candidate_records": len(records),
                "candidate_files": [path.name for path in written],
                "technique_corpus": technique_corpus,
                "embedding_cache": str(cache_path.relative_to(project_root)),
            }
        )
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        report(f"failed after {int(time.perf_counter() - started_at)}s; error={type(exc).__name__}: {exc}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report(f"complete; elapsed={int(time.perf_counter() - started_at)}s; output={run_dir}")
    return run_dir
