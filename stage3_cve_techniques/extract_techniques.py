"""Extract ATT&CK technique metadata for Stage 3 CVE mapping.

Output is tactic-first hierarchy to support tactic-constrained lookup:
{
  "domain": "enterprise",
  "tactics": [
    {
      "id": "TA0001",
      "name": "Initial Access",
      "shortname": "initial-access",
      "techniques": [
        {"id": "T1190", "name": "...", "description": "...", "sub_techniques": []}
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage3_cve_techniques.utils import load_json, normalize_main_technique_id, save_json


def _get_external_id(external_references: Iterable[Dict[str, Any]], prefix: str) -> str:
    """Return first external id that starts with prefix."""
    for ref in external_references or []:
        external_id = str(ref.get("external_id", "")).strip().upper()
        if external_id.startswith(prefix.upper()):
            return external_id
    return ""


def _collect_tactics(objects: Iterable[Dict[str, Any]]) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Build tactic metadata and shortname->tactic-id mapping."""
    tactic_by_id: Dict[str, Dict[str, Any]] = {}
    shortname_to_tactic_id: Dict[str, str] = {}

    for obj in objects:
        if not isinstance(obj, dict) or obj.get("type") != "x-mitre-tactic":
            continue

        tactic_id = _get_external_id(obj.get("external_references", []), "TA")
        if not tactic_id:
            continue

        shortname = str(obj.get("x_mitre_shortname", "")).strip().lower()
        name = str(obj.get("name", "")).strip()
        description = str(obj.get("description", "")).strip()

        tactic_by_id[tactic_id] = {
            "id": tactic_id,
            "name": name,
            "shortname": shortname,
            "description": description,
            "techniques": [],
        }
        if shortname:
            shortname_to_tactic_id[shortname] = tactic_id

    return tactic_by_id, shortname_to_tactic_id


def extract_tactic_hierarchy_from_attack_file(attack_json: Path) -> Dict[str, Any]:
    """Extract tactic-first hierarchy with main techniques from one ATT&CK file."""
    data = load_json(attack_json)
    objects = data.get("objects", []) if isinstance(data, dict) else []

    tactic_by_id, shortname_to_tactic_id = _collect_tactics(objects)

    # Deduplicate techniques per tactic.
    seen_pairs: Set[tuple[str, str]] = set()

    for obj in objects:
        if not isinstance(obj, dict) or obj.get("type") != "attack-pattern":
            continue
        if bool(obj.get("x_mitre_is_subtechnique", False)):
            continue
        if bool(obj.get("revoked", False)) or bool(obj.get("x_mitre_deprecated", False)):
            continue

        raw_id = _get_external_id(obj.get("external_references", []), "T")
        if not raw_id or "." in raw_id:
            continue

        technique_id = normalize_main_technique_id(raw_id)
        if not technique_id:
            continue

        name = str(obj.get("name", "")).strip()
        description = str(obj.get("description", "")).strip()
        if not name or not description:
            continue

        mapped_tactics: List[str] = []
        for phase in obj.get("kill_chain_phases", []) or []:
            if not isinstance(phase, dict):
                continue
            phase_name = str(phase.get("phase_name", "")).strip().lower()
            tactic_id = shortname_to_tactic_id.get(phase_name, "").strip().upper()
            if tactic_id:
                mapped_tactics.append(tactic_id)

        for tactic_id in list(dict.fromkeys(mapped_tactics)):
            tactic_obj = tactic_by_id.get(tactic_id)
            if not tactic_obj:
                continue
            key = (tactic_id, technique_id)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            tactic_obj["techniques"].append(
                {
                    "id": technique_id,
                    "name": name,
                    "description": description,
                    # Reserve field for future extension.
                    "sub_techniques": [],
                }
            )

    tactics: List[Dict[str, Any]] = []
    for tactic in tactic_by_id.values():
        tactic["techniques"].sort(key=lambda x: x["id"])
        tactics.append(tactic)

    tactics.sort(key=lambda x: x["id"])
    return {"tactics": tactics}


def extract_and_save_all_techniques(project_root: Path) -> Dict[str, Path]:
    """Extract hierarchy for enterprise, ICS and mobile and save to data folder."""
    og_data_dir = project_root / "og_data"
    out_dir = project_root / "stage3_cve_techniques" / "data"

    domain_sources = {
        "enterprise": og_data_dir / "enterprise-attack.json",
        "ics": og_data_dir / "ics-attack.json",
        "mobile": og_data_dir / "mobile-attack.json",
    }

    outputs: Dict[str, Path] = {}
    for domain_key, source_file in domain_sources.items():
        hierarchy = extract_tactic_hierarchy_from_attack_file(source_file)
        output_file = out_dir / f"{domain_key}_techniques.json"
        save_json(hierarchy, output_file)
        outputs[domain_key] = output_file

    return outputs


def main() -> None:
    """CLI entry for technique extraction stage."""
    parser = argparse.ArgumentParser(description="Extract ATT&CK techniques by domain.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root directory.",
    )
    args = parser.parse_args()

    outputs = extract_and_save_all_techniques(args.project_root)
    for domain_key, out_file in outputs.items():
        print(f"[INFO] Saved {domain_key} techniques: {out_file}")


if __name__ == "__main__":
    main()
