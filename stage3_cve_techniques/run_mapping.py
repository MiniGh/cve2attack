"""Batch runner for Stage 3 CVE to ATT&CK techniques mapping using LLM."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage3_cve_techniques.extract_techniques import extract_and_save_all_techniques
from stage3_cve_techniques.llm_client import LLMClient, LLMClientConfig
from stage3_cve_techniques.parser import parse_techniques_from_llm_response
from stage3_cve_techniques.prompt_builder import build_technique_mapping_prompt
from stage3_cve_techniques.utils import (
    iter_cve_records,
    iter_cve_year_files,
    load_json,
    save_json,
    setup_logging,
    tqdm,
)

LOGGER = logging.getLogger("stage3.run_mapping")


def _normalize_domain(domain: str) -> str:
    """Normalize domain label to one of enterprise/ics/mobile."""
    normalized = str(domain).strip().lower()
    if normalized in {"enterprise", "ics", "mobile"}:
        return normalized
    return "enterprise"


def _extract_year_from_file(year_file: Path) -> int | None:
    """Extract numeric year from a file name like CVE-2021.json."""
    try:
        return int(year_file.stem.split("-")[1])
    except (IndexError, ValueError):
        return None


def _load_tactic_to_techniques_by_domain(
    project_root: Path,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load domain -> tactic -> techniques index."""
    data_dir = project_root / "stage3_cve_techniques" / "data"
    file_map = {
        "enterprise": data_dir / "enterprise_techniques.json",
        "ics": data_dir / "ics_techniques.json",
        "mobile": data_dir / "mobile_techniques.json",
    }

    index_by_domain: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for domain, file_path in file_map.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing techniques file: {file_path}")

        data = load_json(file_path)
        domain_index: Dict[str, List[Dict[str, Any]]] = {}

        if isinstance(data, dict) and isinstance(data.get("tactics"), list):
            for tactic in data.get("tactics", []):
                if not isinstance(tactic, dict):
                    continue
                tactic_id = str(tactic.get("id", "")).strip().upper()
                if not tactic_id:
                    continue

                techniques = tactic.get("techniques", [])
                if not isinstance(techniques, list):
                    techniques = []

                normalized_techniques: List[Dict[str, Any]] = []
                seen: Set[str] = set()
                for technique in techniques:
                    if not isinstance(technique, dict):
                        continue
                    technique_id = str(technique.get("id", "")).strip().upper()
                    if not technique_id or technique_id in seen:
                        continue
                    seen.add(technique_id)
                    normalized_techniques.append(
                        {
                            "id": technique_id,
                            "name": str(technique.get("name", "")).strip(),
                            "description": str(technique.get("description", "")).strip(),
                        }
                    )

                domain_index[tactic_id] = normalized_techniques
        else:
            raise ValueError(f"Invalid techniques file format: {file_path}")

        index_by_domain[domain] = domain_index

    return index_by_domain


def _load_stage2_predictions_by_cve(stage2_result_dir: Path) -> Dict[str, List[str]]:
    """Load stage-2 predictions from yearly files into a CVE-keyed mapping."""
    predictions: Dict[str, List[str]] = {}
    for year_file in sorted(stage2_result_dir.glob("CVE-*.json")):
        data = load_json(year_file)
        if not isinstance(data, dict):
            continue

        for cve_id, tactics in data.items():
            if not isinstance(cve_id, str):
                continue
            if not isinstance(tactics, list):
                tactics = []
            normalized = [str(x).strip().upper() for x in tactics if str(x).strip()]
            predictions[cve_id] = list(dict.fromkeys(normalized))

    return predictions


def _collect_candidate_techniques(
    domain: str,
    stage2_tactics: List[str],
    tactic_to_techniques: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """Collect unique candidate techniques from mapped stage-2 tactics."""
    candidates: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    domain_index = tactic_to_techniques.get(domain, {})
    for tactic_id in stage2_tactics:
        for technique in domain_index.get(tactic_id, []):
            technique_id = str(technique.get("id", "")).strip().upper()
            if not technique_id or technique_id in seen:
                continue
            seen.add(technique_id)
            candidates.append(technique)

    return candidates


def _map_single_cve(
    cve_id: str,
    record: Dict[str, Any],
    domain_mapping: Dict[str, str],
    stage2_predictions: Dict[str, List[str]],
    tactic_to_techniques: Dict[str, Dict[str, List[Dict[str, Any]]]],
    client: LLMClient,
) -> tuple[List[str], Dict[str, Any]]:
    """Map one CVE to technique IDs through LLM inference and collect trace."""
    stage2_tactics = stage2_predictions.get(cve_id, [])
    domain = _normalize_domain(domain_mapping.get(cve_id, "enterprise"))

    trace: Dict[str, Any] = {
        "cve_id": cve_id,
        "domain": domain,
        "stage2_tactics": stage2_tactics,
        "candidate_technique_ids": [],
        "candidate_count": 0,
        "prompt": "",
        "llm_raw_response": "",
        "parsed_techniques": [],
        "filtered_techniques": [],
        "empty_reason": "",
    }

    if not stage2_tactics:
        trace["empty_reason"] = "stage2_empty"
        return [], trace

    description = str(record.get("description", "")).strip()
    if not description:
        LOGGER.warning("Empty description, skip CVE: %s", cve_id)
        trace["empty_reason"] = "empty_description"
        return [], trace

    candidate_techniques = _collect_candidate_techniques(
        domain=domain,
        stage2_tactics=stage2_tactics,
        tactic_to_techniques=tactic_to_techniques,
    )
    trace["candidate_technique_ids"] = [str(t.get("id", "")).strip().upper() for t in candidate_techniques]
    trace["candidate_count"] = len(candidate_techniques)

    if not candidate_techniques:
        LOGGER.warning("No candidate techniques from stage-2 tactics for %s", cve_id)
        trace["empty_reason"] = "no_candidates"
        return [], trace

    prompt = build_technique_mapping_prompt(description, candidate_techniques)
    trace["prompt"] = prompt

    try:
        response_obj = client.generate(prompt)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("LLM error for %s: %s", cve_id, exc)
        trace["empty_reason"] = "llm_error"
        trace["llm_raw_response"] = str(exc)
        return [], trace

    trace["llm_raw_response"] = str(response_obj.get("response", "")) if isinstance(response_obj, dict) else str(response_obj)

    techniques = parse_techniques_from_llm_response(response_obj)
    trace["parsed_techniques"] = techniques

    candidate_ids = {str(t.get("id", "")).strip().upper() for t in candidate_techniques}

    filtered: List[str] = []
    seen: Set[str] = set()
    for technique_id in techniques:
        if technique_id in candidate_ids and technique_id not in seen:
            seen.add(technique_id)
            filtered.append(technique_id)
        if len(filtered) >= 3:
            break

    trace["filtered_techniques"] = filtered

    if not filtered:
        LOGGER.warning("JSON parse failed or no techniques for %s", cve_id)
        trace["empty_reason"] = "parse_or_filter_empty"

    return filtered, trace


def process_cve_wrapper(
    cve: tuple[str, Dict[str, Any]],
    domain_mapping: Dict[str, str],
    stage2_predictions: Dict[str, List[str]],
    tactic_to_techniques: Dict[str, Dict[str, List[Dict[str, Any]]]],
    client: LLMClient,
) -> tuple[str, List[str], Dict[str, Any]]:
    """Run single CVE mapping safely and return trace on failure too."""
    cve_id, record = cve
    try:
        techniques, trace = _map_single_cve(
            cve_id=cve_id,
            record=record,
            domain_mapping=domain_mapping,
            stage2_predictions=stage2_predictions,
            tactic_to_techniques=tactic_to_techniques,
            client=client,
        )
        return cve_id, techniques, trace
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Unhandled CVE processing error for %s: %s", cve_id, exc)
        return cve_id, [], {
            "cve_id": cve_id,
            "empty_reason": "unhandled_error",
            "error": str(exc),
        }


def run_mapping(
    project_root: Path,
    model: str,
    limit: int | None,
    max_retries: int,
    timeout: int,
    workers: int,
    start_year: int | None,
    end_year: int | None,
    stage2_result_dir: Path,
    result_dir: Path | None,
    save_trace: bool,
    trace_dir: Path | None,
) -> None:
    """Main mapping routine across yearly CVE files."""
    cve_dir = project_root / "og_data" / "cve"
    domain_file = project_root / "cve_to_attack_domain" / "result" / "cve_domain_mapping.json"

    if result_dir is None:
        result_dir = project_root / "stage3_cve_techniques" / "result_sec-i1"
    result_dir.mkdir(parents=True, exist_ok=True)

    if save_trace and trace_dir is None:
        trace_dir = result_dir / "trace"
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    if not domain_file.exists():
        raise FileNotFoundError(f"Missing domain mapping file: {domain_file}")
    if not stage2_result_dir.exists():
        raise FileNotFoundError(f"Missing stage-2 result directory: {stage2_result_dir}")

    extract_and_save_all_techniques(project_root)
    tactic_to_techniques = _load_tactic_to_techniques_by_domain(project_root)

    domain_mapping = load_json(domain_file)
    if not isinstance(domain_mapping, dict):
        raise ValueError("Domain mapping file must be a JSON object")

    stage2_predictions = _load_stage2_predictions_by_cve(stage2_result_dir)

    client = LLMClient(
        LLMClientConfig(
            model=model,
            max_retries=max_retries,
            timeout_seconds=timeout,
        )
    )

    processed = 0
    all_year_files = iter_cve_year_files(cve_dir)
    year_files_with_year = []
    for file_path in all_year_files:
        year = _extract_year_from_file(file_path)
        if year is not None:
            year_files_with_year.append((year, file_path))

    if not year_files_with_year:
        raise ValueError(f"No valid CVE year files found under: {cve_dir}")

    min_year = min(y for y, _ in year_files_with_year)
    max_year = max(y for y, _ in year_files_with_year)

    effective_start = start_year if start_year is not None else min_year
    effective_end = end_year if end_year is not None else max_year

    if effective_start > effective_end:
        raise ValueError(
            f"Invalid year range: start_year={effective_start} is greater than end_year={effective_end}"
        )
    if effective_start < min_year or effective_start > max_year:
        raise ValueError(
            f"start_year={effective_start} is outside available range [{min_year}, {max_year}]"
        )
    if effective_end < min_year or effective_end > max_year:
        raise ValueError(
            f"end_year={effective_end} is outside available range [{min_year}, {max_year}]"
        )

    year_files = [
        file_path for year, file_path in year_files_with_year if effective_start <= year <= effective_end
    ]

    LOGGER.info("Processing CVE years in range [%s, %s]", effective_start, effective_end)

    for year_file in year_files:
        year_results: Dict[str, List[str]] = {}
        records = list(iter_cve_records(year_file))
        if limit is not None:
            remaining = max(limit - processed, 0)
            records = records[:remaining]

        stage2_non_empty = sum(1 for cve_id, _ in records if stage2_predictions.get(cve_id, []))
        LOGGER.info(
            "Year %s records=%s stage2_non_empty=%s",
            year_file.name,
            len(records),
            stage2_non_empty,
        )

        trace_file = None
        if save_trace and trace_dir is not None:
            trace_file = trace_dir / f"{year_file.stem}.json"

        progress = tqdm(total=len(records), desc=f"Mapping {year_file.name}", unit="CVE")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map: Dict[Future[tuple[str, List[str], Dict[str, Any]]], str] = {
                executor.submit(
                    process_cve_wrapper,
                    cve=(cve_id, record),
                    domain_mapping=domain_mapping,
                    stage2_predictions=stage2_predictions,
                    tactic_to_techniques=tactic_to_techniques,
                    client=client,
                ): cve_id
                for cve_id, record in records
            }

            trace_records: List[Dict[str, Any]] = []
            for future in as_completed(future_map):
                cve_id, techniques, trace = future.result()
                year_results[cve_id] = techniques
                processed += 1
                progress.set_postfix_str(f"current={cve_id}")
                progress.update(1)

                if trace_file is not None:
                    trace_records.append(trace)

        if hasattr(progress, "close"):
            progress.close()

        output_file = result_dir / year_file.name
        save_json(year_results, output_file)
        LOGGER.info("Saved %s records to %s", len(year_results), output_file)
        if trace_file is not None:
            save_json(trace_records, trace_file)
            LOGGER.info("Saved trace to %s", trace_file)

        if limit is not None and processed >= limit:
            break

    LOGGER.info("Completed mapping for %s CVEs", processed)


def main() -> None:
    """CLI entry for stage 3 mapping."""
    parser = argparse.ArgumentParser(description="Run CVE to ATT&CK techniques mapping with LLM")
    parser.add_argument("-m", "--model", default="sec-i1", help="Model name for local LLM API")
    parser.add_argument("-l", "--limit", type=int, default=None, help="Process only first N CVEs")
    parser.add_argument("-r", "--max-retries", type=int, default=3, help="Max retries for LLM request")
    parser.add_argument("-t", "--timeout", type=int, default=60, help="HTTP timeout in seconds")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Thread pool workers for concurrent CVE mapping")
    parser.add_argument("-s", "--start-year", type=int, default=None, help="Start CVE year, e.g. 2022")
    parser.add_argument("-e", "--end-year", type=int, default=None, help="End CVE year, e.g. 2022")
    parser.add_argument("-P", "--project-root", type=Path, default=PROJECT_ROOT, help="Project root directory")
    parser.add_argument("-S", "--stage2-result-dir", type=Path, default=PROJECT_ROOT / "stage2_cve_tactics" / "result_sec-i1", help="Directory containing stage-2 yearly prediction JSON files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logs")
    parser.add_argument("-o", "--result-dir", type=Path, default=None, help="Directory to save yearly prediction JSON files")
    parser.add_argument("--save-trace", action="store_true", help="Save per-CVE prompt/response trace JSON")
    parser.add_argument("--trace-dir", type=Path, default=None, help="Directory to save trace JSON files")
    args = parser.parse_args()

    if args.start_year is not None and args.start_year < 1999:
        raise ValueError("--start-year must be >= 1999")
    if args.end_year is not None and args.end_year < 1999:
        raise ValueError("--end-year must be >= 1999")
    if args.start_year is not None and args.end_year is not None and args.start_year > args.end_year:
        raise ValueError("--start-year cannot be greater than --end-year")

    setup_logging(verbose=args.verbose)
    run_mapping(
        project_root=args.project_root,
        model=args.model,
        limit=args.limit,
        max_retries=args.max_retries,
        timeout=args.timeout,
        workers=args.workers,
        start_year=args.start_year,
        end_year=args.end_year,
        stage2_result_dir=args.stage2_result_dir,
        result_dir=args.result_dir,
        save_trace=args.save_trace,
        trace_dir=args.trace_dir,
    )


if __name__ == "__main__":
    main()
