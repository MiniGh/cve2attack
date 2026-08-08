"""Parser utilities for extracting technique IDs from LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from stage3_cve_techniques.utils import normalize_main_technique_id


def _extract_json_candidates(text: str) -> List[str]:
    """Extract possible JSON objects from free-form model text."""
    candidates: List[str] = []

    text = text.strip()
    if text:
        candidates.append(text)

    block_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
    for match in block_pattern.findall(text):
        candidates.append(match.strip())

    for start in [m.start() for m in re.finditer(r"\{", text)]:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1].strip())
                    break

    return list(dict.fromkeys(candidates))


def parse_techniques_from_llm_response(response_obj: Dict[str, Any]) -> List[str]:
    """Parse a normalized list of main technique IDs from LLM JSON response."""
    raw_text = ""
    if isinstance(response_obj, dict):
        raw_text = str(response_obj.get("response", ""))

    for candidate in _extract_json_candidates(raw_text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        techniques = payload.get("techniques", [])
        if not isinstance(techniques, list):
            continue

        normalized: List[str] = []
        for technique in techniques:
            text = str(technique).strip().upper()
            if not text:
                continue

            # Accept common model variants such as "1059" or "T1059.001".
            match = re.search(r"T?\d{4}(?:\.\d+)?", text)
            if not match:
                continue

            technique_id = normalize_main_technique_id(match.group(0))
            if technique_id:
                normalized.append(technique_id)

        return list(dict.fromkeys(normalized))

    return []
