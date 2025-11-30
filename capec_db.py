#从capec.xml文件中提取capec中的
#Description;ATT&CK Techniques ID;CVE
import xml.etree.ElementTree as ET
import re
import json
import sys

CAPEC_XML_FILE = sys.argv[1] if len(sys.argv) > 1 else "./og_source/capec.xml"
OUTPUT_FILE = "./source/capec_db.json"

def extract_cves_from_text(text: str) -> list[str]:
    """提取 CVE-ID，兼容 CVE-YYYY-NNNN ~ CVE-YYYY-NNNNNNN"""
    return re.findall(r'CVE-\d{4}-\d{4,7}', text.upper())

def main():
    print(f"[!] Loading CAPEC XML: {CAPEC_XML_FILE}")
    tree = ET.parse(CAPEC_XML_FILE)
    root = tree.getroot()
    ns = {'capec': 'http://capec.mitre.org/capec-3'}

    capec_data = {}

    print("[✓] Parsing CAPEC entries...")

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
            extended_description = ''.join(ext_desc_elem.itertext()).strip()

        # 提取 ATT&CK Techniques (Entry_ID)
        techniques = []
        taxonomy_mappings = attack_pattern.find('.//capec:Taxonomy_Mappings', ns)
        if taxonomy_mappings is not None:
            for tm in taxonomy_mappings.findall('.//capec:Taxonomy_Mapping[@Taxonomy_Name="ATTACK"]', ns):
                entry_id_elem = tm.find('capec:Entry_ID', ns)
                if entry_id_elem is not None and entry_id_elem.text:
                    techniques.append(entry_id_elem.text.strip())

        # 提取 CVEs from Example_Instances
        cves = set()
        examples_section = attack_pattern.find('.//capec:Example_Instances', ns)
        if examples_section is not None:
            for example_elem in examples_section.findall('.//capec:Example', ns):
                full_text = ''.join(example_elem.itertext())
                cves.update(extract_cves_from_text(full_text))
        cves = sorted(cves)

        # ========== 提取 CWE IDs ==========
        cwes = []
        related_weaknesses = attack_pattern.find('.//capec:Related_Weaknesses', ns)
        if related_weaknesses is not None:
            for rw in related_weaknesses.findall('.//capec:Related_Weakness', ns):
                cwe_id = rw.get('CWE_ID')
                if cwe_id:
                    cwes.append(f"CWE-{cwe_id}")
        cwes = sorted(set(cwes))

        # 存储结果
        capec_data[capec_id] = {
            "name": name,
            "abstraction":abstraction,
            "description": description,
            "extended_description": extended_description,
            "techniques": sorted(techniques),  # 保持有序
            "cves": cves,
            "cwes": cwes
        }

    print(f"[✓] Processed {len(capec_data)} CAPEC entries")
    total_techniques = sum(len(v["techniques"]) for v in capec_data.values())
    total_cves = sum(len(v["cves"]) for v in capec_data.values())
    print(f"[✓] Total ATT&CK techniques mapped: {total_techniques}")
    print(f"[✓] Total CVEs extracted: {total_cves}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(capec_data, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved to {OUTPUT_FILE}")

    # 示例输出
    print("\n[ℹ] Sample (first 2 entries with data):")
    count = 0
    for cid, data in capec_data.items():
        if count >= 2:
            break
        if data["techniques"] or data["cves"]:
            print(f"  CAPEC-{cid}: {data['name']}")
            if data["techniques"]:
                print(f"    Techniques: {data['techniques']}")
            if data["cves"]:
                print(f"    CVEs: {data['cves']}")
            count += 1

if __name__ == "__main__":
    main()
