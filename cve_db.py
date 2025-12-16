import json
import os
from pathlib import Path

SOURCE_DIR = Path("./og_source/cve")
TARGET_DIR = Path("./source/cve")

EXCLUDE_KEYS = {"cwes"}


def sanitize_record(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in EXCLUDE_KEYS}


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Source CVE directory not found: {SOURCE_DIR}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for src_file in sorted(SOURCE_DIR.glob("*.json")):
        with src_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        sanitized = {}
        for cve_id, record in data.items():
            sanitized[cve_id] = sanitize_record(record)

        target_file = TARGET_DIR / src_file.name
        with target_file.open("w", encoding="utf-8") as f:
            json.dump(sanitized, f, ensure_ascii=False, indent=2)

        print(f"[{src_file.name}] exported {len(sanitized)} CVEs to {target_file}")


if __name__ == "__main__":
    main()
