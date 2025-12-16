import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_file = root / "og_source" / "capec.xml"
    output_file = root / "data_analyse" / "result" / "capec2techniques.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    tree = ET.parse(input_file)
    root_elem = tree.getroot()
    ns = {'capec': 'http://capec.mitre.org/capec-3'}

    count = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for attack_pattern in root_elem.findall('.//capec:Attack_Pattern', ns):
            capec_id = attack_pattern.get('ID')
            if not capec_id:
                continue

            techniques = []
            taxonomy_mappings = attack_pattern.find('.//capec:Taxonomy_Mappings', ns)
            if taxonomy_mappings is not None:
                for tm in taxonomy_mappings.findall('.//capec:Taxonomy_Mapping[@Taxonomy_Name="ATTACK"]', ns):
                    entry_id_elem = tm.find('capec:Entry_ID', ns)
                    if entry_id_elem is not None and entry_id_elem.text:
                        techniques.append(entry_id_elem.text.strip())
            techniques = sorted(set(techniques))
            if not techniques:
                continue

            record = {"capec_id": capec_id, "techniques": techniques}
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1

    print(f"Total CAPEC->technique records: {count}. Output file: {output_file}")


if __name__ == "__main__":
    main()
