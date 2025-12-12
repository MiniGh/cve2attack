import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_file = root / "result" / "existing_capec2techniques.json"
    output_file = root / "data_analyse" / "result" / "capec2techniques.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for capec_id in sorted(data.keys(), key=lambda x: (len(x), x)):
            entry = data[capec_id]
            out = {
                "capec_id": capec_id,
                "original_techniques": entry.get("original_techniques", []),
            }
            fout.write(json.dumps(out, ensure_ascii=True) + "\n")
            count += 1

    print(f"Total CAPEC->technique records: {count}. Output file: {output_file}")


if __name__ == "__main__":
    main()
