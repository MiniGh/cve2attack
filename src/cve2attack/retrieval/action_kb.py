"""Build fine-grained ATT&CK action documents for Stage-1 retrieval.

Unlike the technique corpus, this module keeps every ATT&CK description or
procedure relationship as a separate searchable unit.  Each hit remains
traceable to its STIX source and is rolled up to a top-level Technique only
after retrieval.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from cve2attack.retrieval.technique_kb import normalize_text
from cve2attack.schemas import parent_technique_id


ACTION_CORPUS_VERSION = "1.0"
_CITATION_PATTERN = re.compile(r"\s*\(Citation:[^)]+\)", flags=re.IGNORECASE)
_CVE_PATTERN = re.compile(r"\b(?:CVE|CAN)-\d{4}-\d{4,}\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ActionDocument:
    """One independently embedded ATT&CK action and its parent Technique."""

    action_id: str
    technique_id: str
    technique_name: str
    tactics: tuple[str, ...]
    source_type: str
    source_stix_id: str
    text: str
    vulnerability_ids: tuple[str, ...] = ()


def _external_id(references: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the MITRE ATT&CK external ID from a STIX reference list."""
    for reference in references or []:
        if reference.get("source_name") == "mitre-attack":
            value = str(reference.get("external_id") or "").strip().upper()
            if value:
                return value
    return None


def extract_vulnerability_ids(value: str) -> tuple[str, ...]:
    """Extract canonical CVE IDs from ATT&CK prose, including legacy CAN IDs."""
    identifiers = {
        re.sub(r"^CAN-", "CVE-", match.group(0).upper())
        for match in _CVE_PATTERN.finditer(value or "")
    }
    return tuple(sorted(identifiers))


def sanitize_action_text(value: str, *, max_chars: int = 1200) -> str:
    """Normalize ATT&CK prose and remove exact vulnerability identifiers.

    Public procedure examples occasionally name the same CVE that appears in
    an evaluation benchmark.  Replacing both CVE and legacy CAN identifiers
    prevents an exact-identifier shortcut while retaining the surrounding
    attacker behavior and product context.
    """
    text = normalize_text(_CITATION_PATTERN.sub("", value or ""))
    text = _CVE_PATTERN.sub("[VULNERABILITY]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _tactics(obj: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(phase["phase_name"])
        for phase in obj.get("kill_chain_phases", [])
        if isinstance(phase, Mapping) and phase.get("phase_name")
    )


def load_action_documents(
    attack_bundle: Path,
    *,
    include_descriptions: bool = True,
    include_procedures: bool = True,
    min_chars: int = 20,
    max_chars: int = 1200,
) -> list[ActionDocument]:
    """Load description and procedure actions mapped to active parent Techniques."""
    if not include_descriptions and not include_procedures:
        raise ValueError("Action corpus must include descriptions, procedures, or both")
    if min_chars < 0:
        raise ValueError("action_document.min_chars must be non-negative")
    if max_chars < 0:
        raise ValueError("action_document.max_chars must be non-negative")

    with attack_bundle.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    objects = bundle.get("objects", []) if isinstance(bundle, Mapping) else []
    if not isinstance(objects, list):
        raise ValueError(f"ATT&CK bundle objects must be a list: {attack_bundle}")

    # First retain active parent metadata.  Sub-technique actions are later
    # rolled up through their external ID (for example T1059.001 -> T1059).
    parents: dict[str, tuple[str, tuple[str, ...], str]] = {}
    patterns: dict[str, tuple[str, str, bool, str]] = {}
    for obj in objects:
        if not isinstance(obj, Mapping) or obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue
        external_id = _external_id(obj.get("external_references", []))
        stix_id = str(obj.get("id") or "")
        if not external_id or not stix_id or not external_id.startswith("T"):
            continue
        name = normalize_text(str(obj.get("name") or "")) or external_id
        is_subtechnique = bool(obj.get("x_mitre_is_subtechnique", False) or "." in external_id)
        patterns[stix_id] = (external_id, name, is_subtechnique, str(obj.get("description") or ""))
        if not is_subtechnique:
            parents[external_id] = (name, _tactics(obj), stix_id)

    documents: list[ActionDocument] = []
    seen_text: dict[tuple[str, str], int] = {}

    def append_action(
        *,
        source_type: str,
        source_stix_id: str,
        target_stix_id: str,
        raw_text: str,
        source_name: str | None = None,
    ) -> None:
        pattern = patterns.get(target_stix_id)
        if pattern is None:
            return
        external_id, _, _, _ = pattern
        technique_id = parent_technique_id(external_id)
        parent = parents.get(technique_id)
        if parent is None:
            return
        text = sanitize_action_text(raw_text, max_chars=max_chars)
        if source_name:
            text = sanitize_action_text(f"{source_name}. {text}", max_chars=max_chars)
        if len(text) < min_chars:
            return
        deduplication_key = (technique_id, text.casefold())
        vulnerability_ids = extract_vulnerability_ids(raw_text)
        if deduplication_key in seen_text:
            # Two relationships can become identical after CVE/CAN masking.
            # Preserve the union so leave-one-CVE-out exclusion cannot be
            # bypassed merely because the other relationship appeared first.
            index = seen_text[deduplication_key]
            existing = documents[index]
            merged_ids = tuple(sorted(set(existing.vulnerability_ids) | set(vulnerability_ids)))
            if merged_ids != existing.vulnerability_ids:
                documents[index] = replace(existing, vulnerability_ids=merged_ids)
            return
        seen_text[deduplication_key] = len(documents)
        parent_name, tactics, _ = parent
        documents.append(
            ActionDocument(
                action_id=f"{source_type}:{source_stix_id}",
                technique_id=technique_id,
                technique_name=parent_name,
                tactics=tactics,
                source_type=source_type,
                source_stix_id=source_stix_id,
                text=text,
                vulnerability_ids=vulnerability_ids,
            )
        )

    if include_descriptions:
        for stix_id, (external_id, name, is_subtechnique, description) in patterns.items():
            append_action(
                source_type=("subtechnique_description" if is_subtechnique else "technique_description"),
                source_stix_id=stix_id,
                target_stix_id=stix_id,
                raw_text=description,
                source_name=name,
            )

    if include_procedures:
        for obj in objects:
            if not isinstance(obj, Mapping):
                continue
            if obj.get("type") != "relationship" or obj.get("relationship_type") != "uses":
                continue
            relationship_id = str(obj.get("id") or "")
            target_ref = str(obj.get("target_ref") or "")
            if not relationship_id or not target_ref:
                continue
            append_action(
                source_type="procedure",
                source_stix_id=relationship_id,
                target_stix_id=target_ref,
                raw_text=str(obj.get("description") or ""),
            )

    return sorted(documents, key=lambda item: (item.technique_id, item.action_id, item.text))


def action_corpus_stats(actions: Sequence[ActionDocument]) -> dict[str, Any]:
    """Return deterministic counts for terminal inspection and run manifests."""
    counts = Counter(item.source_type for item in actions)
    return {
        "action_count": len(actions),
        "technique_count": len({item.technique_id for item in actions}),
        "actions_with_vulnerability_ids": sum(bool(item.vulnerability_ids) for item in actions),
        "action_types": dict(sorted(counts.items())),
    }
