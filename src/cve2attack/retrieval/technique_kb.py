"""Build the parent-technique retrieval corpus from ATT&CK STIX."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TechniqueDocument:
    technique_id: str
    name: str
    tactics: tuple[str, ...]
    stix_id: str
    text: str


def normalize_text(value: str) -> str:
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value or "")
    return re.sub(r"\s+", " ", value).strip()


def _external_id(references: Sequence[Mapping[str, Any]]) -> str | None:
    for reference in references or []:
        if reference.get("source_name") == "mitre-attack":
            value = str(reference.get("external_id") or "").strip()
            if value:
                return value
    return None


def _procedure_map(objects: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for obj in objects:
        if obj.get("type") != "relationship" or obj.get("relationship_type") != "uses":
            continue
        target = obj.get("target_ref")
        description = normalize_text(str(obj.get("description") or ""))
        if isinstance(target, str) and description:
            result.setdefault(target, []).append(description)
    return result


def load_technique_documents(
    attack_bundle: Path,
    *,
    include_procedures: bool,
    procedure_char_limit: int = 1500,
) -> list[TechniqueDocument]:
    with attack_bundle.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    objects = bundle.get("objects", []) if isinstance(bundle, dict) else []
    procedures = _procedure_map(objects) if include_procedures else {}

    documents: list[TechniqueDocument] = []
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("x_mitre_is_subtechnique", False):
            continue
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        technique_id = _external_id(obj.get("external_references", []))
        if not technique_id or "." in technique_id:
            continue

        name = normalize_text(str(obj.get("name") or ""))
        description = normalize_text(str(obj.get("description") or ""))
        stix_id = str(obj.get("id") or "")
        parts = []
        if name:
            parts.append(f"Technique Name: {name}")
        if description:
            parts.append(f"Technique Description: {description}")

        if include_procedures:
            unique = list(dict.fromkeys(procedures.get(stix_id, [])))
            procedure_text = "\n".join(f"- {item}" for item in unique)
            if procedure_char_limit > 0:
                procedure_text = procedure_text[:procedure_char_limit].rstrip()
            if procedure_text:
                parts.append(f"Procedure Examples:\n{procedure_text}")

        text = "\n\n".join(parts).strip()
        if not text:
            continue
        tactics = tuple(
            str(phase["phase_name"])
            for phase in obj.get("kill_chain_phases", [])
            if isinstance(phase, dict) and phase.get("phase_name")
        )
        documents.append(
            TechniqueDocument(
                technique_id=technique_id,
                name=name or technique_id,
                tactics=tactics,
                stix_id=stix_id,
                text=text,
            )
        )
    return sorted(documents, key=lambda item: item.technique_id)
