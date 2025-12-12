import json
from pathlib import Path
from typing import Dict, Iterable, List


def unique_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def load_mapping(file_path: Path, key_field: str, value_field: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = str(obj.get(key_field, ""))
            if not key:
                continue
            values = obj.get(value_field, []) or []
            mapping[key] = [str(v) for v in values]
    return mapping


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cve2cwe_dir = root / "data_analyse" / "result" / "cve2cwe"
    cwe2capec_file = root / "data_analyse" / "result" / "cwe2capec.jsonl"
    capec2tech_file = root / "data_analyse" / "result" / "capec2techniques.jsonl"

    output_dir = root / "data_analyse" / "result" / "cve2attack"
    output_dir.mkdir(parents=True, exist_ok=True)
    full_output = root / "data_analyse" / "result" / "cve2technique_full.jsonl"

    if not cve2cwe_dir.is_dir():
        raise FileNotFoundError(f"CVE->CWE input dir not found: {cve2cwe_dir}")
    if not cwe2capec_file.is_file():
        raise FileNotFoundError(f"CWE->CAPEC input file not found: {cwe2capec_file}")
    if not capec2tech_file.is_file():
        raise FileNotFoundError(f"CAPEC->techniques input file not found: {capec2tech_file}")

    cwe_to_capec = load_mapping(cwe2capec_file, "cwe_id", "original_capecs")
    capec_to_tech = load_mapping(capec2tech_file, "capec_id", "original_techniques")

    total_records = 0
    total_with_tech = 0
    full_chain: List[str] = []

    with full_output.open("w", encoding="utf-8") as full_out:
        for path in sorted(cve2cwe_dir.glob("CVE-*.jsonl")):
            count = 0
            with_tech = 0
            out_path = output_dir / path.name
            with path.open("r", encoding="utf-8") as fin, out_path.open(
                "w", encoding="utf-8"
            ) as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    cve_id = obj.get("cve_id")
                    cwes = obj.get("original_cwes") or obj.get("cwes") or []
                    cwes = [str(x) for x in cwes]

                    capecs = []
                    for cwe in cwes:
                        capecs.extend(cwe_to_capec.get(cwe, []))
                    capecs = unique_preserve(capecs)

                    techniques = []
                    for capec in capecs:
                        techniques.extend(capec_to_tech.get(capec, []))
                    techniques = unique_preserve(techniques)

                    record = {
                        "cve_id": cve_id,
                        "cwes": cwes,
                        "capecs": capecs,
                        "techniques": techniques,
                    }
                    fout.write(json.dumps(record, ensure_ascii=True) + "\n")

                    if techniques:
                        with_tech += 1
                    if cwes and capecs and techniques:
                        full_out.write(json.dumps(record, ensure_ascii=True) + "\n")
                        full_chain.append(cve_id)

                    count += 1
            print(f"{path.name}: {count} records, {with_tech} with techniques")
            total_records += count
            total_with_tech += with_tech

    print(
        f"Total CVE records processed: {total_records}; with techniques: {total_with_tech}; full chain: {len(full_chain)}"
    )
    print(f"Full chain output file: {full_output}")


if __name__ == "__main__":
    main()
