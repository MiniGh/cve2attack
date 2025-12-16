#从capec.xml文件中提取capec中的
#Description;ATT&CK Techniques ID;CVE
import xml.etree.ElementTree as ET
import json
import sys

CAPEC_XML_FILE = sys.argv[1] if len(sys.argv) > 1 else "./og_source/capec.xml"
OUTPUT_FILE = "./source/capec_db.json"


def main():
    print(f"[!] Loading CAPEC XML: {CAPEC_XML_FILE}")
    tree = ET.parse(CAPEC_XML_FILE)
    root = tree.getroot()
    ns = {'capec': 'http://capec.mitre.org/capec-3'}

    capec_data = {}

    print("[✓] Parsing CAPEC entries (metadata only, no mappings)...")

    for attack_pattern in root.findall('.//capec:Attack_Pattern', ns):
        capec_id = attack_pattern.get('ID')
        name = attack_pattern.get('Name')
        abstraction = attack_pattern.get('Abstraction', "Unknown") #获取 abstraction 属性
        if not capec_id or not name:
            continue


        # 提取 Description
        desc_elem = attack_pattern.find('.//capec:Description', ns)
        description = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

        # ========== 提取 Extended_Description ==========
        ext_desc_elem = attack_pattern.find('.//capec:Extended_Description', ns)
        extended_description = ""
        if ext_desc_elem is not None:
            extended_description = "".join(ext_desc_elem.itertext()).strip()

        capec_data[capec_id] = {
            "name": name,
            "abstraction": abstraction,
            "description": description,
            "extended_description": extended_description,
        }

    print(f"[✓] Processed {len(capec_data)} CAPEC entries")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(capec_data, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
