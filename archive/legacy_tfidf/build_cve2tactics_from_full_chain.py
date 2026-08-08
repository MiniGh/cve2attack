"""Build CVE-to-tactics labels for full-chain CVEs using enterprise ATT&CK data.

This script enriches records from cve2technique_full.jsonl by adding a tactics
field derived from technique -> tactic relations in enterprise-attack.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


def _extract_external_id(references: Iterable[dict], prefix: str) -> str:
    """Return the first external_id starting with the provided prefix."""
    for ref in references or []:
        external_id = str(ref.get("external_id", "")).strip()
        if external_id.upper().startswith(prefix.upper()):
            return external_id
    return ""


def _normalize_technique_id(technique_id: str) -> str:
    """Normalize technique IDs to ATT&CK canonical shape without leading T."""
    normalized = str(technique_id).strip().upper()
    if normalized.startswith("T"):
        normalized = normalized[1:]
    return normalized


def build_technique_to_tactics_map(enterprise_attack_file: Path) -> Dict[str, List[str]]:
    """Create mapping from technique id (without leading T) to tactic TA IDs."""
    with open(enterprise_attack_file, "r", encoding="utf-8") as f:
        attack_data = json.load(f)

    objects = attack_data.get("objects", []) if isinstance(attack_data, dict) else []

    shortname_to_tactic_id: Dict[str, str] = {}
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("type") != "x-mitre-tactic":
            continue

        shortname = str(obj.get("x_mitre_shortname", "")).strip().lower()
        tactic_id = _extract_external_id(obj.get("external_references", []), "TA")
        if shortname and tactic_id:
            shortname_to_tactic_id[shortname] = tactic_id

    technique_to_tactics: Dict[str, List[str]] = {}
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("type") != "attack-pattern":
            continue

        technique_external_id = _extract_external_id(obj.get("external_references", []), "T")
        if not technique_external_id:
            continue

        technique_key = _normalize_technique_id(technique_external_id)
        tactic_ids: List[str] = []
        seen: Set[str] = set()

        for phase in obj.get("kill_chain_phases", []) or []:
            if not isinstance(phase, dict):
                continue
            if str(phase.get("kill_chain_name", "")).strip().lower() != "mitre-attack":
                continue

            phase_name = str(phase.get("phase_name", "")).strip().lower()
            tactic_id = shortname_to_tactic_id.get(phase_name)
            if tactic_id and tactic_id not in seen:
                seen.add(tactic_id)
                tactic_ids.append(tactic_id)

        if tactic_ids:
            technique_to_tactics[technique_key] = tactic_ids

    return technique_to_tactics


def enrich_cve_records_with_tactics(
    input_jsonl: Path,
    output_jsonl: Path,
    technique_to_tactics: Dict[str, List[str]],
) -> None:
    """Add deduplicated tactics list to each CVE JSONL record and save output."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as src, open(
        output_jsonl, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue

            record = json.loads(stripped)
            techniques = record.get("techniques", [])

            tactics: List[str] = []
            seen: Set[str] = set()
            for technique in techniques:
                technique_key = _normalize_technique_id(str(technique))
                for tactic_id in technique_to_tactics.get(technique_key, []):
                    if tactic_id not in seen:
                        seen.add(tactic_id)
                        tactics.append(tactic_id)

            record["tactics"] = tactics
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """CLI entry point for enriching full-chain CVE records with tactics."""
    parser = argparse.ArgumentParser(
        description="Add tactics labels to full-chain CVE records from enterprise ATT&CK data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Validate_data") / "cve2technique_full.jsonl",
        help="Input JSONL containing full-chain CVE records",
    )
    parser.add_argument(
        "--attack",
        type=Path,
        default=Path("og_data") / "enterprise-attack.json",
        help="Enterprise ATT&CK STIX JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Validate_data") / "cve2technique_full_with_tactics.jsonl",
        help="Output JSONL with added tactics field",
    )
    args = parser.parse_args()

    mapping = build_technique_to_tactics_map(args.attack)
    enrich_cve_records_with_tactics(args.input, args.output, mapping)

    print(f"[INFO] Technique->tactics entries: {len(mapping)}")
    print(f"[INFO] Saved enriched file: {args.output}")


if __name__ == "__main__":
    main()
