"""Command-line entry points for running, rewriting and comparing experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from cve2attack.config import PROJECT_ROOT, load_experiment, project_path
from cve2attack.data.kev import import_kev_benchmarks
from cve2attack.data.loaders import CVERepository, benchmark_truth, candidate_records
from cve2attack.data.triage import import_triage_benchmarks
from cve2attack.domain.classifier import classify_directory
from cve2attack.evaluation.metrics import evaluate
from cve2attack.evaluation.report import write_comparison_report
from cve2attack.evaluation.diagnostics import diagnose_triage_candidates
from cve2attack.evaluation.triage import compare_with_triage
from cve2attack.fusion.rrf import run_rrf_fusion
from cve2attack.pipeline import (
    build_queries,
    resolve_attack_bundle,
    run_experiment,
    select_input_ids,
)
from cve2attack.retrieval.technique_kb import load_technique_documents
from cve2attack.rewrite.ollama import OllamaClient
from cve2attack.rewrite.pipeline import generate_rewrite_cache
from cve2attack.stage2.pipeline import run_context_extraction, run_stage2_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CVE to ATT&CK mapping research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate an experiment definition")
    validate.add_argument("experiment", type=Path)

    inspect = subparsers.add_parser("inspect", help="Check paths and input coverage without loading a model")
    inspect.add_argument("experiment", type=Path)
    inspect.add_argument("--max-cves", type=int)
    inspect.add_argument("--benchmark")

    kev = subparsers.add_parser(
        "import-kev",
        help="Build frozen CTID KEV benchmark views from the published CSV snapshot",
    )
    kev.add_argument(
        "--source",
        default="data/raw/kev/kev-02.13.2025_attack-15.1-enterprise.csv",
        help="Path to the fixed CTID KEV CSV snapshot",
    )
    kev.add_argument(
        "--benchmark-root",
        default="data/benchmarks",
        help="Directory in which the three generated benchmark views are created",
    )
    kev.add_argument(
        "--cve2attack-benchmark",
        default="data/benchmarks/cve2attack_result",
        help="Existing CVE2ATT&CK directory used to create the strict non-overlap view",
    )

    triage_import = subparsers.add_parser(
        "import-triage",
        help="Build the exact public TRIAGE test views from selected replication files",
    )
    triage_import.add_argument(
        "--source-dir",
        default="data/raw/triage/triage_2025",
        help="Directory containing the frozen TRIAGE split, labels and source metadata",
    )
    triage_import.add_argument(
        "--benchmark-root",
        default="data/benchmarks",
        help="Directory in which the two generated TRIAGE test views are created",
    )

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

    triage_compare = subparsers.add_parser(
        "compare-triage",
        help="Compare completed runs with public TRIAGE and SMET predictions",
    )
    triage_compare.add_argument("runs", nargs="+", type=Path)
    triage_compare.add_argument("--comparison-id")
    triage_compare.add_argument(
        "--source-dir",
        default="data/raw/triage/triage_2025",
        help="Directory containing the frozen public reference predictions",
    )

    triage_diagnose = subparsers.add_parser(
        "diagnose-triage",
        help="Diagnose candidate complementarity on the exact public TRIAGE test split",
    )
    triage_diagnose.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Full-ranking V1/V2/V3 run directories to diagnose",
    )
    triage_diagnose.add_argument("--comparison-id")
    triage_diagnose.add_argument(
        "--source-dir",
        default="data/raw/triage/triage_2025",
        help="Directory containing the frozen split, labels and reference predictions",
    )

    rrf = subparsers.add_parser(
        "fuse-rrf",
        help="Fuse completed candidate runs with label-free Reciprocal Rank Fusion",
    )
    rrf.add_argument("runs", nargs="+", type=Path, help="Completed source run directories")
    rrf.add_argument("--run-id", required=True, help="Unique output directory name under runs/")
    rrf.add_argument("--benchmark", required=True, help="Fixed benchmark cohort to fuse and evaluate")
    rrf.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Final controlled number of candidates per CVE (default: 20)",
    )
    rrf.add_argument(
        "--source-depth",
        type=int,
        default=50,
        help="Maximum rank read from each source before fusion (default: 50)",
    )
    rrf.add_argument(
        "--rank-constant",
        type=float,
        default=60.0,
        help="Positive RRF smoothing constant (default: 60)",
    )
    rrf.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Optional positive source weights in the same order as the run paths",
    )

    graph_context = subparsers.add_parser(
        "extract-graph-context",
        help="Extract versioned local and upstream context from a MulVAL AttackGraph.xml",
    )
    graph_context.add_argument(
        "--attack-graph",
        required=True,
        type=Path,
        help="MulVAL AttackGraph.xml path; relative paths are resolved from the project root",
    )
    graph_context.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON path; relative paths are resolved from the project root",
    )
    graph_context.add_argument(
        "--max-graph-depth",
        type=int,
        default=2,
        help="Maximum upstream evidence expansion depth (default: 2)",
    )
    graph_context.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file instead of failing",
    )

    stage2_run = subparsers.add_parser(
        "run-stage2",
        help="Join stage-1 candidates with one attack graph and run topology-only reranking",
    )
    stage2_run.add_argument(
        "--stage1-run",
        required=True,
        type=Path,
        help="Completed stage-1 run directory containing candidate JSONL files",
    )
    stage2_run.add_argument(
        "--attack-graph",
        required=True,
        type=Path,
        help="MulVAL AttackGraph.xml path",
    )
    stage2_run.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark directory name under data/benchmarks used only for after-the-fact evaluation",
    )
    stage2_run.add_argument("--run-id", required=True, help="Unique directory name under output-root")
    stage2_run.add_argument(
        "--output-root",
        type=Path,
        default=Path("stage2_runs"),
        help="Stage-2 output root (default: stage2_runs)",
    )
    stage2_run.add_argument(
        "--scenario-kind",
        default="synthetic_topology_smoke",
        help="Provenance label written to the manifest and report",
    )
    stage2_run.add_argument(
        "--max-graph-depth",
        type=int,
        default=2,
        help="Maximum upstream evidence expansion depth (default: 2)",
    )
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
            resolve_attack_bundle(config, PROJECT_ROOT),
            include_procedures=bool(document_config["include_procedures"]),
            procedure_char_limit=int(document_config["procedure_char_limit"]),
        )
        print(json.dumps({**coverage, "technique_count": len(techniques)}, indent=2))
        return

    if args.command == "import-kev":
        stats = import_kev_benchmarks(
            source=project_path(args.source),
            benchmark_root=project_path(args.benchmark_root),
            cve2attack_benchmark=project_path(args.cve2attack_benchmark),
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.command == "import-triage":
        stats = import_triage_benchmarks(
            source_dir=project_path(args.source_dir),
            benchmark_root=project_path(args.benchmark_root),
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
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
        return

    if args.command == "compare-triage":
        identifier = args.comparison_id or (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_triage_public_test"
        )
        path = compare_with_triage(
            args.runs,
            output_dir=PROJECT_ROOT / "comparisons" / identifier,
            project_root=PROJECT_ROOT,
            source_dir=project_path(args.source_dir),
        )
        print(path)
        return

    if args.command == "diagnose-triage":
        identifier = args.comparison_id or (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_triage_candidate_diagnostics"
        )
        path = diagnose_triage_candidates(
            args.runs,
            output_dir=PROJECT_ROOT / "comparisons" / identifier,
            project_root=PROJECT_ROOT,
            source_dir=project_path(args.source_dir),
        )
        print(path)
        return

    if args.command == "fuse-rrf":
        path = run_rrf_fusion(
            args.runs,
            run_id=args.run_id,
            benchmark_name=args.benchmark,
            top_k=args.top_k,
            source_depth=args.source_depth,
            rank_constant=args.rank_constant,
            weights=args.weights,
        )
        print(path)
        return

    if args.command == "extract-graph-context":
        attack_graph = (
            args.attack_graph
            if args.attack_graph.is_absolute()
            else PROJECT_ROOT / args.attack_graph
        )
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        path = run_context_extraction(
            attack_graph,
            output,
            max_graph_depth=args.max_graph_depth,
            overwrite=args.force,
        )
        print(path)
        return

    if args.command == "run-stage2":
        stage1_run = (
            args.stage1_run if args.stage1_run.is_absolute() else PROJECT_ROOT / args.stage1_run
        )
        attack_graph = (
            args.attack_graph
            if args.attack_graph.is_absolute()
            else PROJECT_ROOT / args.attack_graph
        )
        output_root = (
            args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
        )
        path = run_stage2_experiment(
            stage1_run=stage1_run,
            attack_graph_path=attack_graph,
            benchmark_dir=PROJECT_ROOT / "data" / "benchmarks" / args.benchmark,
            output_root=output_root,
            run_id=args.run_id,
            project_root=PROJECT_ROOT,
            scenario_kind=args.scenario_kind,
            max_graph_depth=args.max_graph_depth,
        )
        print(path)
        return


if __name__ == "__main__":
    main()
