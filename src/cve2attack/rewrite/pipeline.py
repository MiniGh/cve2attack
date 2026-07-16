"""Generate and checkpoint CVE attacker-action rewrites."""

from __future__ import annotations

import json
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


def load_cwe_catalog(path: Path) -> dict[str, dict[str, str]]:
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

    def rewrite_one(item: tuple[str, str, list[str]]) -> tuple[str, str] | None:
        cve_id, description, cwes = item
        text = client.generate(system=SYSTEM_PROMPT, prompt=build_prompt(description, cwes, catalog))
        return (cve_id, text) if text else None

    success = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(rewrite_one, item) for item in pending]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception:
                result = None
            if result:
                cache[result[0]] = result[1]
                success += 1
            else:
                failed += 1
            if index % 20 == 0 or index == len(futures):
                _save(cache, output_path)

    return {
        "requested": len(cve_ids),
        "already_cached": len(cve_ids) - len(pending) - missing_description,
        "success": success,
        "failed": failed,
        "missing_description": missing_description,
        "cache_size": len(cache),
    }
