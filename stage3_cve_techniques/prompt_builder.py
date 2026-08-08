"""Prompt construction for CVE to ATT&CK technique mapping."""

from __future__ import annotations

from typing import Dict, List

from stage3_cve_techniques.utils import flatten_technique_list


def build_technique_mapping_prompt(
    description: str, candidate_techniques: List[Dict[str, str]]
) -> str:
    """Build an instruction prompt for LLM-based technique classification."""
    techniques_block = flatten_technique_list(candidate_techniques)

    return (
        "You are a cybersecurity expert for CVE analysis.\n\n"
        "Task: choose the ATT&CK techniques most directly supported by the CVE description.\n"
        "You must choose only from the candidate list.\n"
        "Candidate list is guaranteed non-empty.\n\n"
        f"CVE Description:\n{description}\n\n"
        f"Candidate Techniques:\n{techniques_block}\n\n"
        "Hard constraints:\n"
        "- Return 1 to 3 technique IDs.\n"
        "- Use main-technique format T#### only.\n"
        "- Every returned ID must come from the candidate list.\n"
        "- Prefer precision: choose fewer IDs when uncertain.\n\n"
        "Return ONLY JSON exactly like:\n"
        "{\n"
        '  "techniques": ["T1190"]\n'
        "}\n"
        "Do not include any extra text."
    )
