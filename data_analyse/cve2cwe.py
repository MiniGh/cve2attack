import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_dir = root / "og_source" / "cve"
    output_dir = root / "data_analyse" / "result" / "cve2cwe"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    total_records = 0
    total_files = 0

    for path in sorted(input_dir.glob("CVE-*.json")):
        total_files += 1
        out_path = output_dir / path.with_suffix('.jsonl').name 
        count = 0

        with path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            data = json.load(fin)
            for cve_id, obj in data.items():
                    cwes = obj.get("cwes", []) or []
                    if not cwes:
                        continue
                    record = {"cve_id": cve_id, "cwes": cwes}
                    fout.write(json.dumps(record, ensure_ascii=True) + "\n")
                    count += 1

        print(f"{path.name}: {count} records")
        total_records += count

    print(
        f"Total CVE->CWE records: {total_records} across {total_files} files. Output dir: {output_dir}"
    )


if __name__ == "__main__":
    main()
