"""Extract ATT&CK tactics metadata from official ATT&CK JSON datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage2_cve_tactics.utils import load_json, save_json


def _get_external_id(external_references: List[Dict[str, Any]]) -> str:
    """Find ATT&CK external_id such as TA0001 from external references."""
    for ref in external_references or []:
        external_id = str(ref.get("external_id", "")).strip()
        if external_id:
            return external_id
    return ""


def extract_tactics_from_attack_file(attack_json: Path) -> List[Dict[str, str]]:
    """Extract tactics from one ATT&CK JSON file."""
    data = load_json(attack_json)
    objects = data.get("objects", []) if isinstance(data, dict) else []

    tactics: List[Dict[str, str]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "x-mitre-tactic":
            continue

        tactic_id = _get_external_id(obj.get("external_references", []))
        name = str(obj.get("name", "")).strip()
        description = str(obj.get("description", ""))

        if not tactic_id or not name:
            continue

        tactics.append(
            {
                "id": tactic_id,
                "name": name,
                "description": description,
            }
        )

    tactics.sort(key=lambda x: x["id"])
    return tactics


def extract_and_save_all_tactics(project_root: Path) -> Dict[str, Path]:
    """Extract tactics for enterprise, ICS, and mobile and save to data folder."""
    og_data_dir = project_root / "og_data"
    out_dir = project_root / "stage2_cve_tactics" / "data"

    domain_sources = {
        "enterprise": og_data_dir / "enterprise-attack.json",
        "ics": og_data_dir / "ics-attack.json",
        "mobile": og_data_dir / "mobile-attack.json",
    }

    outputs: Dict[str, Path] = {}
    for domain_key, source_file in domain_sources.items():
        tactics = extract_tactics_from_attack_file(source_file)
        output_file = out_dir / f"{domain_key}_tactics.json"
        save_json(tactics, output_file)
        outputs[domain_key] = output_file

    return outputs


def main() -> None:
    """CLI entry for tactic extraction stage."""
    parser = argparse.ArgumentParser(description="Extract ATT&CK tactics by domain.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root directory.",
    )
    args = parser.parse_args()

    outputs = extract_and_save_all_tactics(args.project_root)
    for domain_key, out_file in outputs.items():
        print(f"[INFO] Saved {domain_key} tactics: {out_file}")


if __name__ == "__main__":
    main()
