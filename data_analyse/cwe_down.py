#!/usr/bin/env python3
"""Extract downward mappings from CWE source: CWE → CVE (Observed_Examples).

Reads og_source/cwe.xml and writes jsonl to data_analyse/result/cwe_down.jsonl.
Skips CWEs with no CVE examples.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_cwe_observed_examples(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = "{http://cwe.mitre.org/cwe-2.0}"  # typical CWE 2.0 namespace

    results = []

    for weakness in root.findall(f".//{ns}Weakness"):
        cwe_id = weakness.get("ID") or weakness.get("External_Reference_ID")
        if not cwe_id:
            continue

        cves = []
        for obs in weakness.findall(f".//{ns}Observed_Example"):
            ref = obs.find(f"{ns}Reference")
            if ref is not None and ref.text:
                text = ref.text.strip()
                # Expect CVE-like strings
                if text.upper().startswith("CVE-"):
                    cves.append(text.upper())

        cves = sorted(set(cves))
        if cves:
            results.append({"cwe_id": cwe_id, "cves": cves})

    return results


def main():
    root = Path(__file__).resolve().parent.parent
    xml_path = root / "og_source" / "cwe.xml"
    out_path = root / "data_analyse" / "result" / "cwe_down.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = parse_cwe_observed_examples(xml_path)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")

    print(f"[✓] CWE downward mappings written: {len(items)} items -> {out_path}")


if __name__ == "__main__":
    main()
