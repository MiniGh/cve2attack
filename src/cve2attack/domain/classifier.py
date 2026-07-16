"""Classify CVEs as Enterprise, ICS or Mobile before retrieval."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DOMAIN_KEYWORDS = {
    "ICS": {
        "plc", "scada", "rtu", "modbus", "dnp3", "hmi", "siemens",
        "rockwell", "schneider", "abb", "honeywell",
    },
    "Mobile": {"android", "ios", "iphone", "ipad", "watchos", "tvos"},
    "Enterprise": {
        "windows", "linux", "macos", "apache", "nginx", "tomcat", "confluence",
    },
}


def tokenize(text: str) -> set[str]:
    return {item for item in re.sub(r"[^a-z0-9]+", " ", text.lower()).split() if item}


def cpe_tokens(cpes: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for cpe in cpes or []:
        parts = str(cpe).split(":")
        if len(parts) >= 5 and parts[:2] == ["cpe", "2.3"]:
            result.update(part for part in (parts[3], parts[4]) if part and part != "*")
    return result


def classify(record: Mapping[str, Any]) -> str:
    tokens = tokenize(str(record.get("description") or ""))
    tokens |= tokenize(str(record.get("sourceIdentifier") or ""))
    tokens |= cpe_tokens(record.get("cpes", []) or [])
    if tokens & DOMAIN_KEYWORDS["ICS"]:
        return "ICS"
    if tokens & DOMAIN_KEYWORDS["Mobile"]:
        return "Mobile"
    return "Enterprise"


def classify_directory(cve_dir: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for input_path in sorted(cve_dir.glob("CVE-*.json")):
        year = input_path.stem.split("-", 1)[1]
        value = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            continue
        output_path = output_dir / f"CVE-{year}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for cve_id, record in sorted(value.items()):
                if not isinstance(record, dict):
                    continue
                domain = classify(record)
                handle.write(json.dumps({"cve_id": cve_id, "domain": domain}) + "\n")
                counts[domain] += 1
                total += 1
    return {"total": total, **dict(counts)}
