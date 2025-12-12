import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_dir = root / "result" / "existing_cve2cwe"
    output_dir = root / "data_analyse" / "result" / "cve2cwe"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    total_records = 0
    total_files = 0

    for path in sorted(input_dir.glob("CVE-*.jsonl")):
        total_files += 1
        out_path = output_dir / path.name
        count = 0

        with path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out = {
                    "cve_id": obj["cve_id"],
                    "original_cwes": obj.get("original_cwes", []),
                }
                fout.write(json.dumps(out, ensure_ascii=True) + "\n")
                count += 1

        print(f"{path.name}: {count} records")
        total_records += count

    print(
        f"Total CVE->CWE records: {total_records} across {total_files} files. Output dir: {output_dir}"
    )


if __name__ == "__main__":
    main()
