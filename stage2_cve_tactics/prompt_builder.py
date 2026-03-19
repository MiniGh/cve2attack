"""Prompt construction for CVE to ATT&CK tactics mapping."""

from __future__ import annotations

from typing import Dict, List

from stage2_cve_tactics.utils import flatten_tactic_list


def build_tactic_mapping_prompt(description: str, candidate_tactics: List[Dict[str, str]]) -> str:
    """Build an instruction prompt for LLM-based tactic classification."""
    tactics_block = flatten_tactic_list(candidate_tactics)

    return (
        "You are a cybersecurity expert.\n\n"
        "Given the following CVE description, identify the most relevant MITRE ATT&CK tactics.\n"
        "You may return one or multiple tactics from the candidate list.\n"
        "Only use tactic IDs that appear in the candidate list.\n\n"
        f"CVE Description:\n{description}\n\n"
        f"Candidate Tactics:\n{tactics_block}\n\n"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        '  "tactics": ["TA0001", "TA0002"]\n'
        "}\n"
        "/no_think\n"
    )
