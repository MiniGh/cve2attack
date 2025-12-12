import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_file = root / "result" / "existing_cwe2capec.json"
    output_file = root / "data_analyse" / "result" / "cwe2capec.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for cwe_id in sorted(data.keys(), key=lambda x: (len(x), x)):
            entry = data[cwe_id]
            out = {
                "cwe_id": cwe_id,
                "original_capecs": entry.get("original_capecs", []),
            }
            fout.write(json.dumps(out, ensure_ascii=True) + "\n")
            count += 1

    print(f"Total CWE->CAPEC records: {count}. Output file: {output_file}")


if __name__ == "__main__":
    main()
