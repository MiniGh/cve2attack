import json
from collections import defaultdict

# ====== 1. 加载数据 ======
print("[!] Loading enterprise-attack.json...")
with open("./og_source/enterprise-attack.json", "r", encoding="utf-8") as f:
    bundle = json.load(f)

objects = bundle["objects"]

#构建 phase_name → Tactic Name 映射
phase_to_tactic = {}
for obj in objects:
    if obj.get("type") == "x-mitre-tactic":
        shortname = obj.get("x_mitre_shortname")  # e.g., "execution"
        name = obj.get("name")                    # e.g., "Execution"
        if shortname and name:
            phase_to_tactic[shortname] = name
print(f"[✓] Loaded {len(phase_to_tactic)} tactics mapping (phase_name → name)")


print("[✓] Processing techniques (metadata only, no mappings)...")
technique_data = {}

for obj in objects:
    if obj.get("type") != "attack-pattern":
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
        tech_id = stix_id.split("--")[-1][:6].upper()

    description = (obj.get("description") or "").strip()

    tactic_names = []
    for phase in obj.get("kill_chain_phases", []):
        if phase.get("kill_chain_name") == "mitre-attack":
            pn = phase.get("phase_name")
            if pn and pn in phase_to_tactic:
                tactic_names.append(phase_to_tactic[pn])
    tactics = sorted(set(tactic_names))

    technique_data[tech_id] = {
        "name": tech_name,
        "description": description,
        "tactics": tactics,
    }

print(f"[✓] Processed {len(technique_data)} techniques")

OUTPUT_FILE = "./source/attack_db.json"

# 确保按 Technique ID（Txxx）排序
sorted_techniques = dict(sorted(technique_data.items(), key=lambda x: x[0]))
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sorted_techniques, f, indent=2, ensure_ascii=False)

print(f"[✅] Saved to {OUTPUT_FILE}")
