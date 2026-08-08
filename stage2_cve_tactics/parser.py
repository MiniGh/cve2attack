"""Parser utilities for extracting tactic IDs from LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


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

    # Preserve insertion order while deduplicating.
    return list(dict.fromkeys(candidates))


def parse_tactics_from_llm_response(response_obj: Dict[str, Any]) -> List[str]:
    """Parse a normalized list of tactic IDs from LLM JSON response object."""
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

        tactics = payload.get("tactics", [])
        if not isinstance(tactics, list):
            continue

        normalized: List[str] = []
        for tactic in tactics:
            tactic_id = str(tactic).strip().upper()
            if re.fullmatch(r"TA\d{4}", tactic_id):
                normalized.append(tactic_id)

        # Keep order and remove duplicates.
        return list(dict.fromkeys(normalized))

    return []
