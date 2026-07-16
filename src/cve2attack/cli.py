"""Command-line entry points for running, rewriting and comparing experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from cve2attack.config import PROJECT_ROOT, load_experiment, project_path
from cve2attack.data.loaders import CVERepository, benchmark_truth, candidate_records
from cve2attack.domain.classifier import classify_directory
from cve2attack.evaluation.metrics import evaluate
from cve2attack.evaluation.report import write_comparison_report
from cve2attack.pipeline import build_queries, run_experiment, select_input_ids
from cve2attack.retrieval.technique_kb import load_technique_documents
from cve2attack.rewrite.ollama import OllamaClient
from cve2attack.rewrite.pipeline import generate_rewrite_cache


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CVE to ATT&CK stage-1 candidate generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate an experiment definition")
    validate.add_argument("experiment", type=Path)

    inspect = subparsers.add_parser("inspect", help="Check paths and input coverage without loading a model")
    inspect.add_argument("experiment", type=Path)
    inspect.add_argument("--max-cves", type=int)
    inspect.add_argument("--benchmark")

    subparsers.add_parser("classify-domain", help="Rebuild yearly ATT&CK domain mappings")

    run = subparsers.add_parser("run", help="Run one experiment")
    run.add_argument("experiment", type=Path)
    run.add_argument("--run-id")
    run.add_argument("--max-cves", type=int)
    run.add_argument("--benchmark")

    rewrite = subparsers.add_parser("rewrite", help="Build the rewrite cache used by an experiment")
    rewrite.add_argument("experiment", type=Path)
    rewrite.add_argument("--workers", type=int, default=4)
    rewrite.add_argument("--max-cves", type=int)
    rewrite.add_argument("--no-cache", action="store_true")
    rewrite.add_argument("--benchmark")

    compare = subparsers.add_parser("compare", help="Compare completed runs on one benchmark")
    compare.add_argument("runs", nargs="+", type=Path)
    compare.add_argument("--benchmark", required=True)
    compare.add_argument("--comparison-id")
    return parser


def _run_name(run_dir: Path) -> str:
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        value = json.loads(manifest.read_text(encoding="utf-8"))
        experiment = str(value.get("experiment") or "legacy")
        run_id = str(value.get("run_id") or run_dir.name)
        return f"{experiment} [{run_id}]"
    return run_dir.name


def compare_runs(
    run_dirs: Sequence[Path],
    *,
    benchmark_name: str,
    comparison_id: str | None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    truth = benchmark_truth(project_root / "data" / "benchmarks" / benchmark_name)
    if not truth:
        raise RuntimeError(f"Benchmark has no records: {benchmark_name}")

    rows = {}
    for raw_path in run_dirs:
        run_dir = raw_path if raw_path.is_absolute() else project_root / raw_path
        rows[_run_name(run_dir)] = evaluate(candidate_records(run_dir), truth)

    identifier = comparison_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{benchmark_name}"
    output_dir = project_root / "comparisons" / identifier
    if output_dir.exists():
        raise FileExistsError(f"Comparison directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    payload = {
        "benchmark": benchmark_name,
        "cohort_size": len(truth),
        "runs": {name: metrics.to_dict() for name, metrics in rows.items()},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "cohort.json").write_text(
        json.dumps(sorted(truth), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_comparison_report(
        output_dir / "report.md", benchmark_name=benchmark_name, rows=rows
    )
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)

    if args.command == "validate":
        config = load_experiment(args.experiment)
        print(f"Valid experiment: {config['name']}")
        return

    if args.command == "inspect":
        config = load_experiment(args.experiment)
        if args.benchmark:
            config["input"] = {**config["input"], "mode": "benchmark", "benchmark": args.benchmark}
        identifiers = select_input_ids(config, PROJECT_ROOT)
        if args.max_cves is not None:
            identifiers = identifiers[: max(0, args.max_cves)]
        queries, coverage = build_queries(
            config=config,
            cve_ids=identifiers,
            repository=CVERepository(PROJECT_ROOT / "data" / "raw" / "cve"),
            project_root=PROJECT_ROOT,
        )
        document_config = config["technique_document"]
        techniques = load_technique_documents(
            PROJECT_ROOT / "data" / "knowledge" / "enterprise-attack.json",
            include_procedures=bool(document_config["include_procedures"]),
            procedure_char_limit=int(document_config["procedure_char_limit"]),
        )
        print(json.dumps({**coverage, "technique_count": len(techniques)}, indent=2))
        return

    if args.command == "classify-domain":
        stats = classify_directory(
            PROJECT_ROOT / "data" / "raw" / "cve",
            PROJECT_ROOT / "data" / "derived" / "domain_mapping",
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        path = run_experiment(
            args.experiment,
            run_id=args.run_id,
            max_cves=args.max_cves,
            benchmark=args.benchmark,
        )
        print(path)
        return

    if args.command == "rewrite":
        config = load_experiment(args.experiment)
        if args.benchmark:
            config["input"] = {**config["input"], "mode": "benchmark", "benchmark": args.benchmark}
        if config["query"]["strategy"] != "rewrite_cache":
            raise SystemExit("This experiment does not use query.strategy=rewrite_cache")
        llm = config["query"].get("llm")
        if not isinstance(llm, dict):
            raise SystemExit("The experiment must define query.llm to generate rewrites")
        identifiers = select_input_ids(config, PROJECT_ROOT)
        if args.max_cves is not None:
            identifiers = identifiers[: max(0, args.max_cves)]
        client = OllamaClient(
            base_url=str(llm["base_url"]),
            model=str(llm["model"]),
            timeout_seconds=int(llm.get("timeout_seconds", 120)),
            max_retries=int(llm.get("max_retries", 3)),
        )
        benchmark_name = str(config["input"].get("benchmark", "full_enterprise"))
        cache_value = str(config["query"]["cache"]).format(benchmark=benchmark_name)
        stats = generate_rewrite_cache(
            cve_ids=identifiers,
            repository=CVERepository(PROJECT_ROOT / "data" / "raw" / "cve"),
            cwe_xml=PROJECT_ROOT / "data" / "knowledge" / "cwe.xml",
            client=client,
            output_path=project_path(cache_value),
            workers=args.workers,
            ignore_existing=args.no_cache,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.command == "compare":
        path = compare_runs(
            args.runs,
            benchmark_name=args.benchmark,
            comparison_id=args.comparison_id,
        )
        print(path)


if __name__ == "__main__":
    main()
