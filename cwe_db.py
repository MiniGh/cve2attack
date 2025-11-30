import xml.etree.ElementTree as ET
import json
import sys
import re

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
            extended_description = ''.join(ext_desc_elem.itertext()).strip()

        # 3. CAPEC IDs
        capecs = []
        related_attack_patterns = weakness.find('.//cwe:Related_Attack_Patterns', ns)
        if related_attack_patterns is not None:
            for rap in related_attack_patterns.findall('.//cwe:Related_Attack_Pattern', ns):
                capec_id = rap.get('CAPEC_ID')
                if capec_id:
                    capecs.append(capec_id)
        capecs = sorted(set(capecs))

        # 4. CVE IDs —— 直接从 <Reference> 提取！
        cves = []
        observed_examples = weakness.find('.//cwe:Observed_Examples', ns)
        if observed_examples is not None:
            for example in observed_examples.findall('.//cwe:Observed_Example', ns):
                ref_elem = example.find('.//cwe:Reference', ns)
                if ref_elem is not None and ref_elem.text:
                    # 提取 CVE-XXXX-XXXX（更健壮：支持前后空格）
                    cve_match = re.search(r'CVE-\d{4}-\d{4,7}', ref_elem.text, re.IGNORECASE)
                    if cve_match:
                        cves.append(cve_match.group(0).upper())
        cves = sorted(set(cves))

        # 存储
        cwe_data[cwe_id] = {
            "name": name,
            "abstraction": abstraction,
            "description": description,
            "extended_description": extended_description,
            "capecs": capecs,
            "cves": cves
        }

    print(f"[✓] Processed {len(cwe_data)} CWE entries")
    total_capecs = sum(len(v["capecs"]) for v in cwe_data.values())
    total_cves = sum(len(v["cves"]) for v in cwe_data.values())
    print(f"[✓] Total CAPEC references: {total_capecs}")
    print(f"[✓] Total CVE references: {total_cves}")

    # 保存（按 CWE ID 排序）
    sorted_cwe = dict(sorted(cwe_data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]))
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted_cwe, f, indent=2, ensure_ascii=False)

    print(f"[✅] Saved to {OUTPUT_FILE}")

    # 示例输出
    print("\n[ℹ] Sample (first 2 CWEs with data):")
    count = 0
    for cid, data in sorted_cwe.items():
        if count >= 2:
            break
        if data["capecs"] or data["cves"]:
            print(f"  CWE-{cid}: {data['name']}")
            if data["capecs"]:
                print(f"    CAPECs: {data['capecs']}")
            if data["cves"]:
                print(f"    CVEs: {data['cves']}")
            count += 1

if __name__ == "__main__":
    main()
