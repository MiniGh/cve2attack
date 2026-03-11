#!/usr/bin/env python3
"""Extract downward mappings from CAPEC source:
- CAPEC → CWE (Related_Weaknesses)
- CAPEC → CVE (Example_Instances)

Reads og_source/capec.xml and writes jsonl to data_analyse/result/capec_down.jsonl.
Skips CAPECs with no downward references.
"""

import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET


def _localname(tag: str) -> str:
    return tag.split('}')[-1] if '}' in tag else tag


_cve_re = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def parse_capec_down(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = []

    for ap in root.iter():
        if _localname(ap.tag) != "Attack_Pattern":
            continue
        capec_id = ap.get("ID")
        if not capec_id:
            continue

        cwes = set()
        cves = set()

        for child in ap.iter():
            ln = _localname(child.tag)
            if ln == "Related_Weaknesses":
                for rel in child.iter():
                    if _localname(rel.tag) == "Related_Weakness":
                        cwe_id = rel.get("CWE_ID")
                        if cwe_id:
                            cwes.add(cwe_id)
            elif ln == "Example_Instances" or ln == "Example_Instance":
                texts = []
                if child.text:
                    texts.append(child.text)
                for sub in child.iter():
                    if sub is not child and sub.text:
                        texts.append(sub.text)
                joined = " \n ".join(t.strip() for t in texts if t.strip())
                for m in _cve_re.findall(joined):
                    cves.add(m.upper())

        if cwes or cves:
            obj = {"capec_id": capec_id}
            if cwes:
                obj["cwes"] = sorted(cwes)
            if cves:
                obj["cves"] = sorted(cves)
            results.append(obj)

    return results


def main():
    root = Path(__file__).resolve().parent.parent
    xml_path = root / "og_source" / "capec.xml"
    out_path = root / "data_analyse" / "result" / "capec_down.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = parse_capec_down(xml_path)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")

    print(f"[✓] CAPEC downward mappings written: {len(items)} items -> {out_path}")


if __name__ == "__main__":
    main()
