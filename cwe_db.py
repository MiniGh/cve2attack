import xml.etree.ElementTree as ET
import json
import sys

CWE_XML_FILE = sys.argv[1] if len(sys.argv) > 1 else "./og_source/cwe.xml"
OUTPUT_FILE = "./source/cwe_db.json"

def main():
    print(f"[!] Loading CWE XML: {CWE_XML_FILE}")
    tree = ET.parse(CWE_XML_FILE)
    root = tree.getroot()

    # CWE XML 命名空间（v4.x 使用 xmlns="http://cwe.mitre.org/cwe-6"）
    # 若无命名空间，可设 ns = {}
    ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}  # 常见 v4.x namespace

    weaknesses = root.findall('.//cwe:Weakness', ns)
    print(f"[✓] Found {len(weaknesses)} CWE entries")

    if not weaknesses:
        print("[❌] No <Weakness> found. Check namespace.")
        return

    cwe_data = {}

    # CWE 在 XML 中有两种容器：<Weaknesses> 和 <Categories>/<Views>，我们只取 <Weakness>
    for weakness in weaknesses: 
        cwe_id = weakness.get('ID')
        name = weakness.get('Name')
        abstraction = weakness.get('Abstraction', "Unknown")

        if not cwe_id or not name:
            continue

        # 1. Description
        desc_elem = weakness.find('.//cwe:Description', ns)
        description = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

        # 2. Extended_Description
        ext_desc_elem = weakness.find('.//cwe:Extended_Description', ns)
        extended_description = "" 
        if ext_desc_elem is not None:
            extended_description = "".join(ext_desc_elem.itertext()).strip()

        #存储
        cwe_data[cwe_id] = {
            "name": name,
            "abstraction": abstraction,
            "description": description,
            "extended_description": extended_description,
        }

    sorted_cwe = dict(sorted(cwe_data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]))
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted_cwe, f, indent=2, ensure_ascii=False)

    print(f"[✅] Saved {len(sorted_cwe)} CWEs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
