#!/usr/bin/env python3
"""Extract downward mappings from CAPEC source:
- CAPEC → CWE (Related_Weaknesses)
- CAPEC → CVE (Example_Instances)

Reads og_source/capec.xml and writes jsonl to data_analyse/result/capec_down.jsonl.
Skips CAPECs with no downward references.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_capec_down(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = []

    for ap in root.findall(".//Attack_Pattern"):
        capec_id = ap.get("ID")
        if not capec_id:
            continue

        cwes = []
        cves = []

        # Related_Weaknesses → Related_Weakness/@CWE_ID
        rw = ap.find("Related_Weaknesses")
        if rw is not None:
            for rel in rw.findall("Related_Weakness"):
                cwe_id = rel.get("CWE_ID")
                if cwe_id:
                    cwes.append(cwe_id)

        # Example_Instances → Example_Instance/Reference or /Reference/Text containing CVE-*
        exs = ap.find("Example_Instances")
        if exs is not None:
            for ex in exs.findall("Example_Instance"):
                # Try various child tags that may hold a CVE ref
                for tag in ("Reference", "Description", "Title"):
                    elem = ex.find(tag)
                    if elem is not None and elem.text:
                        text = elem.text.strip()
                        if text.upper().startswith("CVE-"):
                            cves.append(text.upper())

        cwes = sorted(set(cwes))
        cves = sorted(set(cves))

        if cwes or cves:
            obj = {"capec_id": capec_id}
            if cwes:
                obj["cwes"] = cwes
            if cves:
                obj["cves"] = cves
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
