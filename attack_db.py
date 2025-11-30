import json
import datetime
import re
from collections import defaultdict

# ====== 1. 加载数据 ======
print("[!] Loading enterprise-attack.json...")
with open("./og_source/enterprise-attack.json", "r", encoding="utf-8") as f:
    bundle = json.load(f)

objects = bundle["objects"]
id_to_obj = {obj["id"]: obj for obj in objects}

#构建 phase_name → Tactic Name 映射
phase_to_tactic = {}
for obj in objects:
    if obj.get("type") == "x-mitre-tactic":
        shortname = obj.get("x_mitre_shortname")  # e.g., "execution"
        name = obj.get("name")                    # e.g., "Execution"
        if shortname and name:
            phase_to_tactic[shortname] = name
print(f"[✓] Loaded {len(phase_to_tactic)} tactics mapping (phase_name → name)")

# 构建 technique → mitigations / analytics （复用你的逻辑）
tech_to_coas = defaultdict(list)
tech_to_analytics = defaultdict(list)

for obj in objects:
    if obj["type"] == "relationship":
        rel_type = obj.get("relationship_type")
        src, tgt = obj["source_ref"], obj["target_ref"]

        if rel_type == "mitigates" and src.startswith("course-of-action--") and tgt.startswith("attack-pattern--"):
            tech_to_coas[tgt].append(src)

        if obj.get("target_ref", "").startswith("attack-pattern--") and "analytic" in src:
            tech_to_analytics[obj["target_ref"]].append(src)


# ====== 2. CVE 提取函数（复用，但去掉高亮，因不再输出 evidence）======
def extract_cves(text: str) -> list[str]:
    if not text:
        return []
    # 更健壮的 CVE 正则（支持大小写、边界）
    cve_pattern = r"\b(CVE-\d{4}-\d{4,})\b"
    return sorted(set(re.findall(cve_pattern, text, re.IGNORECASE)))


# ====== 3. 遍历 techniques，汇总信息 ======
print("[✓] Processing techniques...")

technique_data = {}

for obj in objects:
    if obj["type"] != "attack-pattern":
        continue

    stix_id = obj["id"]
    tech_name = obj.get("name", "Unknown")

    # 获取 ATT&CK ID (e.g., "T1059")
    tech_id = None
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id", "").startswith("T"):
            tech_id = ref["external_id"]
            break
    if not tech_id:
        tech_id = stix_id.split("--")[-1][:6].upper()  # fallback: e.g., attack-pattern--abc123 → "ABC123"

    # description
    description = obj.get("description", "").strip()

    # tactics (ensure sorted & deduplicated)
    #tactics = sorted(set(tech_to_tactics[stix_id]))

    tactic_names = []
    for phase in obj.get("kill_chain_phases", []):
        if phase.get("kill_chain_name") == "mitre-attack":
            pn = phase.get("phase_name")
            if pn and pn in phase_to_tactic:
                tactic_names.append(phase_to_tactic[pn])
    tactics = sorted(set(tactic_names))

    # collect texts to scan for CVE
    texts = [
        description,
        *[ref.get("description", "") for ref in obj.get("external_references", []) if "description" in ref],
    ]

    # mitigations
    for coa_id in tech_to_coas[stix_id]:
        coa = id_to_obj.get(coa_id)
        if coa:
            texts.append(coa.get("description", ""))

    # analytics
    for analytic_id in tech_to_analytics[stix_id]:
        analytic = id_to_obj.get(analytic_id)
        if analytic:
            desc = analytic.get("analytic_description", "") or analytic.get("description", "")
            texts.append(desc)

    # extract all CVEs
    cves = set()
    for text in texts:
        cves.update(extract_cves(text))
    cves = sorted(cves)

    # 只保存非空条目（或你也可保留所有）
    technique_data[tech_id] = {
        "name": tech_name,
        "description": description,
        "tactics": tactics,
        "cves": cves
    }

print(f"[✓] Processed {len(technique_data)} techniques")
total_cves = sum(len(data["cves"]) for data in technique_data.values())
distinct_cves = len({cve for data in technique_data.values() for cve in data["cves"]})
print(f"[✓] Total CVE mentions: {total_cves}")
print(f"[✓] Distinct CVEs: {distinct_cves}")


# ====== 4. 保存结果（与 capec_full.json 统一风格）======
OUTPUT_FILE = "./source/attack_db.json"

# 确保按 Technique ID（Txxx）排序
sorted_techniques = dict(sorted(technique_data.items(), key=lambda x: x[0]))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sorted_techniques, f, indent=2, ensure_ascii=False)

print(f"[✅] Saved to {OUTPUT_FILE}")

# Sample output
print("\n[ℹ] Sample (first 3 techniques with CVEs):")
count = 0
for tid, data in sorted_techniques.items():
    if data["cves"] and count < 3:
        print(f"  {tid}: {data['name']}")
        print(f"    Tactics: {data['tactics']}")
        print(f"    CVEs: {data['cves']}")
        count += 1
