import json
from pathlib import Path
from typing import Dict, List, Set


def load_cve_techniques(full_path: Path) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = {}
    with full_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cve = obj.get("cve_id")
            if not cve:
                continue
            techs = {str(t) for t in obj.get("techniques", []) if str(t)}
            mapping[cve] = techs
    return mapping


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    attack_path = root / "source" / "attack_db.json"
    full_chain_path = root / "data_analyse" / "result" / "cve2technique_full.jsonl"
    output_path = root / "data_analyse" / "result" / "technique_has_cve.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not attack_path.is_file():
        raise FileNotFoundError(f"attack_db.json not found: {attack_path}")
    if not full_chain_path.is_file():
        raise FileNotFoundError(f"cve2technique_full.jsonl not found: {full_chain_path}")

    with attack_path.open("r", encoding="utf-8") as f:
        attack_db = json.load(f)

    cve_to_techniques = load_cve_techniques(full_chain_path)

    techniques_with_cve = 0
    cve_refs = 0
    missing_cves: List[str] = []
    mismatch_pairs: List[str] = []
    matched = 0
    mismatch = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for tech_id in sorted(attack_db.keys()):
            entry = attack_db[tech_id]
            cves = [str(cve).strip() for cve in entry.get("cves", []) if str(cve).strip()]
            if not cves:
                continue

            record = {
                "technique_id": tech_id,
                "name": entry.get("name", ""),
                "cves": cves,
            }
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")

            techniques_with_cve += 1
            cve_refs += len(cves)

            for cve in cves:
                techs = cve_to_techniques.get(cve)
                if techs is None:
                    missing_cves.append(cve)
                    continue
                if tech_id in techs:
                    matched += 1
                else:
                    mismatch += 1
                    mismatch_pairs.append(f"{cve} -> {tech_id} (expected one of {sorted(techs)})")

    print(f"Techniques with CVE: {techniques_with_cve}; total CVE references: {cve_refs}")
    print(f"Output file: {output_path}")

    unique_missing = sorted(set(missing_cves))
    print(
        f"Comparison vs cve2technique_full: matched {matched}, missing_cve {len(unique_missing)}, technique mismatch {mismatch}"
    )
    if unique_missing:
        preview = ", ".join(unique_missing[:10])
        print(f"Missing CVE not in cve2technique_full (first 10): {preview}")
    if mismatch_pairs:
        preview_pairs = " | ".join(mismatch_pairs[:10])
        print(f"Technique mismatches (first 10): {preview_pairs}")


if __name__ == "__main__":
    main()
