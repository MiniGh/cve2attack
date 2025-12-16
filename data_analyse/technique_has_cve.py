import json
import re
from pathlib import Path
from typing import Dict, List


def extract_cves(text: str) -> List[str]:
    return sorted(set(re.findall(r"CVE-\d{4}-\d{4,7}", text.upper())))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    attack_src = root / "og_source" / "enterprise-attack.json"
    output_path = root / "data_analyse" / "result" / "technique_has_cve.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not attack_src.is_file():
        raise FileNotFoundError(f"enterprise-attack.json not found: {attack_src}")

    with attack_src.open("r", encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])

    count = 0
    cve_refs = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for obj in objects:
            if obj.get("type") != "attack-pattern":
                continue

            tech_id = None
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack" and ref.get("external_id", "").startswith("T"):
                    tech_id = ref["external_id"]
                    break
            if not tech_id:
                continue

            texts = [obj.get("description", "")]
            for ref in obj.get("external_references", []):
                if "description" in ref:
                    texts.append(ref.get("description", ""))

            cves = set()
            for t in texts:
                cves.update(extract_cves(t))
            if not cves:
                continue

            cve_list = sorted(cves)
            record = {
                "technique_id": tech_id,
                "name": obj.get("name", ""),
                "cves": cve_list,
            }
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
            cve_refs += len(cve_list)

    print(f"Techniques with CVE: {count}; total CVE references: {cve_refs}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
