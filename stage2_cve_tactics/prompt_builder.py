"""Prompt construction for CVE to ATT&CK tactics mapping."""

from __future__ import annotations

from typing import Dict, List

from stage2_cve_tactics.utils import flatten_tactic_list


def build_tactic_mapping_prompt(
    description: str, candidate_tactics: List[Dict[str, str]]
) -> str:
    """Build an instruction prompt for LLM-based tactic classification."""
    tactics_block = flatten_tactic_list(candidate_tactics)

    # return f"""You are a cybersecurity expert specializing in vulnerability analysis and the MITRE ATT&CK framework.
    #            Task:
    #            Given a CVE description, identify which MITRE ATT&CK tactics are directly enabled by exploiting the vulnerability.

    #            Key requirement:
    #            Only map tactics that are directly supported by the vulnerability itself.
    #            Do NOT infer multi-step attack chains or post-exploitation behaviors.

    #            Constraints:
    #            - Only include tactics that can be directly achieved by exploiting the vulnerability.
    #            - Do NOT include tactics that require additional attacker actions after exploitation.
    #            - Avoid over-selection (do not include irrelevant tactics).
    #            - Prefer selecting at least ONE tactic if there is reasonable evidence.
    #            - Select at most 4 tactics.
    #            - Only choose from the provided candidate tactics.

    #            CVE Description:
    #            {description}

    #            Candidate Tactics (with descriptions):
    #            {tactics_block}

    #            Output JSON only,such as:
    #            {{
    #              "tactics": ["TA0001", "TA0002"]
    #            }}
    #         """
    return (
        "You are a cybersecurity expert.\n\n"
        "Given the following CVE description, identify the most relevant MITRE ATT&CK tactics.\n"
        f"CVE Description:\n{description}\n\n"
        "Only use tactic IDs that appear in the candidate list.\n\n"
        f"Candidate Tactics:\n{tactics_block}\n\n"
        "You may return one or multiple tactics from the candidate list.\n"
        "Prefer selecting at least ONE tactic if there is reasonable evidence.\n"
        "Select at most 4 tactics.\n"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        '  "tactics": ["TA0001", "TA0002"]\n'
        "}\n"
    )      
