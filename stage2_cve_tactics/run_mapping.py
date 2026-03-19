"""Batch runner for Stage 2 CVE to ATT&CK tactics mapping using LLM."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage2_cve_tactics.extract_tactics import extract_and_save_all_tactics
from stage2_cve_tactics.llm_client import LLMClient, LLMClientConfig
from stage2_cve_tactics.parser import parse_tactics_from_llm_response
from stage2_cve_tactics.prompt_builder import build_tactic_mapping_prompt
from stage2_cve_tactics.utils import (
    iter_cve_records,
    iter_cve_year_files,
    load_json,
    save_json,
    setup_logging,
    tqdm,
)

LOGGER = logging.getLogger("stage2.run_mapping")


def _normalize_domain(domain: str) -> str:
    """Normalize domain label to one of enterprise/ics/mobile."""
    normalized = str(domain).strip().lower()
    if normalized in {"enterprise", "ics", "mobile"}:
        return normalized
    return "enterprise"


def _load_tactics_by_domain(project_root: Path) -> Dict[str, List[Dict[str, str]]]:
    """Load per-domain tactics from extracted files."""
    data_dir = project_root / "stage2_cve_tactics" / "data"
    file_map = {
        "enterprise": data_dir / "enterprise_tactics.json",
        "ics": data_dir / "ics_tactics.json",
        "mobile": data_dir / "mobile_tactics.json",
    }

    tactics_by_domain: Dict[str, List[Dict[str, str]]] = {}
    for domain, file_path in file_map.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing tactics file: {file_path}")
        data = load_json(file_path)
        if not isinstance(data, list):
            raise ValueError(f"Invalid tactics file format: {file_path}")
        tactics_by_domain[domain] = [x for x in data if isinstance(x, dict)]

    return tactics_by_domain


def _map_single_cve(
    cve_id: str,
    record: Dict[str, Any],
    domain_mapping: Dict[str, str],
    tactics_by_domain: Dict[str, List[Dict[str, str]]],
    client: LLMClient,
) -> List[str]:
    """Map one CVE record to tactic IDs through LLM inference."""
    description = str(record.get("description", "")).strip()
    if not description:
        LOGGER.warning("Empty description, skip CVE: %s", cve_id)
        return []

    domain = _normalize_domain(domain_mapping.get(cve_id, "enterprise"))
    candidate_tactics = tactics_by_domain.get(domain, tactics_by_domain["enterprise"])

    prompt = build_tactic_mapping_prompt(description, candidate_tactics)

    try:
        response_obj = client.generate(prompt)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("LLM error for %s: %s", cve_id, exc)
        return []

    tactics = parse_tactics_from_llm_response(response_obj)
    candidate_ids = {str(t.get("id", "")).strip().upper() for t in candidate_tactics}
    tactics = [tid for tid in tactics if tid in candidate_ids]
    if not tactics:
        LOGGER.warning("JSON parse failed or no tactics for %s", cve_id)
    return tactics


def run_mapping(
    project_root: Path,
    model: str,
    limit: int | None,
    max_retries: int,
    timeout: int,
) -> None:
    """Main mapping routine across all yearly CVE files."""
    cve_dir = project_root / "og_data" / "cve"
    domain_file = project_root / "cve_to_attack_domain" / "result" / "cve_domain_mapping.json"
    result_dir = project_root / "stage2_cve_tactics" / "result"

    if not domain_file.exists():
        raise FileNotFoundError(f"Missing domain mapping file: {domain_file}")

    extract_and_save_all_tactics(project_root)
    tactics_by_domain = _load_tactics_by_domain(project_root)
    domain_mapping = load_json(domain_file)
    if not isinstance(domain_mapping, dict):
        raise ValueError("Domain mapping file must be a JSON object")

    client = LLMClient(
        LLMClientConfig(
            model=model,
            max_retries=max_retries,
            timeout_seconds=timeout,
        )
    )

    processed = 0
    year_files = iter_cve_year_files(cve_dir)

    for year_file in year_files:
        year_results: Dict[str, List[str]] = {}
        records = list(iter_cve_records(year_file))
        progress = tqdm(records, desc=f"Mapping {year_file.name}", unit="CVE")

        for cve_id, record in progress:
            if limit is not None and processed >= limit:
                break

            progress.set_postfix_str(f"current={cve_id}")
            tactics = _map_single_cve(
                cve_id=cve_id,
                record=record,
                domain_mapping=domain_mapping,
                tactics_by_domain=tactics_by_domain,
                client=client,
            )
            year_results[cve_id] = tactics
            processed += 1

        output_file = result_dir / year_file.name
        save_json(year_results, output_file)
        LOGGER.info("Saved %s records to %s", len(year_results), output_file)

        if limit is not None and processed >= limit:
            break

    LOGGER.info("Completed mapping for %s CVEs", processed)


def main() -> None:
    """CLI entry for stage 2 mapping."""
    parser = argparse.ArgumentParser(description="Run CVE to ATT&CK tactics mapping with LLM")
    parser.add_argument("--model", default="qwen3:32b", help="Model name for local LLM API")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N CVEs")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for LLM request")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    run_mapping(
        project_root=args.project_root,
        model=args.model,
        limit=args.limit,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
