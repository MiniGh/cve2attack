#!/usr/bin/env python3
"""Extract downward mappings from CWE source: CWE → CVE (Observed_Examples).

Reads og_source/cwe.xml and writes jsonl to data_analyse/result/cwe_down.jsonl.
Skips CWEs with no CVE examples.
"""

import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET


def _localname(tag: str) -> str:
    return tag.split('}')[-1] if '}' in tag else tag


_cve_re = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def parse_cwe_observed_examples(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = []

    for weakness in root.iter():
        if _localname(weakness.tag) != "Weakness":
            continue
        cwe_id = weakness.get("ID") or weakness.get("External_Reference_ID")
        if not cwe_id:
            continue

        cves = set()
        for child in weakness.iter():
            if _localname(child.tag) == "Observed_Example":
                # Scan all descendant texts for CVE patterns
                texts = []
                if child.text:
                    texts.append(child.text)
                for sub in child.iter():
                    if sub is not child and sub.text:
                        texts.append(sub.text)
                joined = " \n ".join(t.strip() for t in texts if t.strip())
                for m in _cve_re.findall(joined):
                    cves.add(m.upper())

        if cves:
            results.append({"cwe_id": cwe_id, "cves": sorted(cves)})

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
