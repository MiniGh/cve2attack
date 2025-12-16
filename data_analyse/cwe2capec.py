import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_file = root / "og_source" / "cwe.xml"
    output_file = root / "data_analyse" / "result" / "cwe2capec.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    tree = ET.parse(input_file)
    ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}
    weaknesses = tree.getroot().findall('.//cwe:Weakness', ns)

    count = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for wk in weaknesses:
            cwe_id = wk.get('ID')
            if not cwe_id:
                continue

            capecs = []
            rap = wk.find('.//cwe:Related_Attack_Patterns', ns)
            if rap is not None:
                for node in rap.findall('.//cwe:Related_Attack_Pattern', ns):
                    capec_id = node.get('CAPEC_ID')
                    if capec_id:
                        capecs.append(capec_id)
            capecs = sorted(set(capecs))
            if not capecs:
                continue

            record = {"cwe_id": cwe_id, "capecs": capecs}
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1

    print(f"Total CWE->CAPEC records: {count}. Output file: {output_file}")


if __name__ == "__main__":
    main()
