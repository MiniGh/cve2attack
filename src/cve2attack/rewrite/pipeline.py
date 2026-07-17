"""Generate and checkpoint CVE attacker-action rewrites."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Sequence

from cve2attack.data.loaders import CVERepository
from cve2attack.rewrite.ollama import OllamaClient


SYSTEM_PROMPT = (
    "You are an offensive cybersecurity analyst with deep expertise in the MITRE ATT&CK framework. "
    "Translate vulnerability descriptions from mechanism-focused language into active attacker-action language."
)

USER_PROMPT = (
    "CVE Description: {description}\n\n"
    "CWE Information:\n{cwe_information}\n\n"
    "Rewrite the above into a concise paragraph (3-5 sentences) describing the attacker's actions in ATT&CK style. "
    "Cover exploitation method, primary capabilities gained, and likely next actions. Output only the paragraph."
)


def _format_duration(seconds: float) -> str:
    """Render elapsed or estimated time in a compact terminal-friendly form."""
    whole_seconds = max(0, int(round(seconds)))
    minutes, seconds_part = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds_part:02d}s"
    return f"{seconds_part}s"


def load_cwe_catalog(path: Path) -> dict[str, dict[str, str]]:
    """Load the CWE fields used to give the rewrite model vulnerability context."""
    namespace = {"cwe": "http://cwe.mitre.org/cwe-7"}
    root = ET.parse(path).getroot()
    result: dict[str, dict[str, str]] = {}
    for weakness in root.iterfind("cwe:Weaknesses/cwe:Weakness", namespace):
        identifier = str(weakness.get("ID") or "").strip()
        if not identifier:
            continue
        result[identifier] = {
            "name": str(weakness.get("Name") or "").strip(),
            "description": " ".join("".join(weakness.itertext()).split())[:1200],
        }
    return result


def build_prompt(
    description: str,
    cwe_ids: Sequence[str],
    cwe_catalog: Mapping[str, Mapping[str, str]],
) -> str:
    """Build the fixed attacker-action prompt for one CVE and its CWE labels."""
    lines: list[str] = []
    for cwe_id in cwe_ids:
        information = cwe_catalog.get(cwe_id)
        if information:
            lines.append(f"CWE-{cwe_id}: {information.get('name', '')}")
            if information.get("description"):
                lines.append(f"  Description: {information['description']}")
        else:
            lines.append(f"CWE-{cwe_id}: no catalog entry")
    cwe_information = "\n".join(lines) if lines else "No CWE information available."
    return USER_PROMPT.format(description=description, cwe_information=cwe_information)


def _save(cache: Mapping[str, str], path: Path) -> None:
    """Atomically checkpoint successful rewrites so an interrupted run can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def generate_rewrite_cache(
    *,
    cve_ids: Sequence[str],
    repository: CVERepository,
    cwe_xml: Path,
    client: OllamaClient,
    output_path: Path,
    workers: int,
    ignore_existing: bool = False,
) -> dict[str, int]:
    """Generate missing rewrites concurrently and report resumable progress.

    Existing non-empty cache records are never sent to the LLM again unless
    ``ignore_existing`` is requested. The cache is checkpointed periodically,
    so an interruption only leaves the current small batch to be retried.
    """
    cache: dict[str, str] = {}
    if output_path.exists() and not ignore_existing:
        value = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            cache = {str(key): str(text) for key, text in value.items() if str(text).strip()}

    catalog = load_cwe_catalog(cwe_xml)
    pending = []
    missing_description = 0
    for cve_id in cve_ids:
        if cve_id in cache:
            continue
        description = repository.description(cve_id)
        if not description:
            missing_description += 1
            continue
        pending.append((cve_id, description, repository.cwes(cve_id)))

    cached_count = len(cache)
    print(
        "[rewrite] "
        f"requested={len(cve_ids)}; cached={cached_count}; pending={len(pending)}; "
        f"missing_description={missing_description}; workers={max(1, workers)}; "
        f"cache={output_path}",
        flush=True,
    )
    if not pending:
        print("[rewrite] no missing rewrites; existing cache is ready to use.", flush=True)
        return {
            "requested": len(cve_ids),
            "already_cached": len(cve_ids) - missing_description,
            "success": 0,
            "failed": 0,
            "missing_description": missing_description,
            "cache_size": len(cache),
        }

    def rewrite_one(item: tuple[str, str, list[str]]) -> tuple[str, str] | None:
        cve_id, description, cwes = item
        text = client.generate(system=SYSTEM_PROMPT, prompt=build_prompt(description, cwes, catalog))
        return (cve_id, text) if text else None

    success = failed = 0
    started_at = time.perf_counter()
    progress_every = min(10, len(pending))
    checkpoint_every = 20
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(rewrite_one, item): item[0] for item in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            cve_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(
                    f"[rewrite] failed cve={cve_id}; error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                result = None
            if result:
                cache[result[0]] = result[1]
                success += 1
            else:
                failed += 1
            if completed % checkpoint_every == 0 or completed == len(futures):
                _save(cache, output_path)
                print(
                    f"[rewrite] checkpoint saved; cache_size={len(cache)}; path={output_path}",
                    flush=True,
                )
            if completed % progress_every == 0 or completed == len(futures):
                elapsed = time.perf_counter() - started_at
                rate = completed / elapsed if elapsed else 0.0
                remaining = len(futures) - completed
                eta = remaining / rate if rate else 0.0
                print(
                    "[rewrite] "
                    f"progress={completed}/{len(futures)}; succeeded={success}; failed={failed}; "
                    f"elapsed={_format_duration(elapsed)}; eta={_format_duration(eta)}",
                    flush=True,
                )

    return {
        "requested": len(cve_ids),
        "already_cached": len(cve_ids) - len(pending) - missing_description,
        "success": success,
        "failed": failed,
        "missing_description": missing_description,
        "cache_size": len(cache),
    }
