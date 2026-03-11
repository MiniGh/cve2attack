import json
import os
import re
from collections import defaultdict
from pathlib import Path

def extract_year_from_cve(cve_id):
    """
    从 CVE ID 中提取年份，例如：
      "CVE-2019-0547" -> "2019"
      "CVE-2000-0342" -> "2000"
    如果格式不合法，返回 None
    """
    match = re.match(r"CVE-(\d{4})-", cve_id, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def load_target_cve_capecs(cve_set, cve_dir):
    """
    只加载与目标 CVE 相关的年份文件，构建 {cve_id: set(capecs)} 映射
    :param cve_set: 所有需要查询的 CVE 集合
    :param cve_dir: CVE 文件所在目录（如 "./cve_data/"）
    :return: dict {cve_id: set_of_capecs}
    """
    cve_to_capecs = {}
    
    # 按年份分组 CVE
    year_to_cves = defaultdict(set)
    invalid_cves = []
    for cve in cve_set:
        year = extract_year_from_cve(cve)
        if year:
            year_to_cves[year].add(cve)
        else:
            invalid_cves.append(cve)

    if invalid_cves:
        print(f"警告：跳过 {len(invalid_cves)} 个无效 CVE 格式，例如: {invalid_cves[:3]}")

    # 只读取涉及的年份文件
    for year, cves_in_year in year_to_cves.items():
        file_path = os.path.join(cve_dir, f"CVE-{year}.jsonl")
        if not os.path.exists(file_path):
            print(f"注意：年份文件不存在: {file_path}，跳过")
            continue

        print(f"正在加载 {file_path} ...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    cve_id = record.get("cve_id")
                    capecs = record.get("capecs", [])
                    
                    # 只处理我们关心的 CVE
                    if cve_id in cves_in_year and isinstance(capecs, list):
                        cve_to_capecs[cve_id] = set(capecs)
                except json.JSONDecodeError:
                    continue  # 跳过损坏行

    return cve_to_capecs

def main():
    root = Path(__file__).resolve().parent.parent
    # 👇 请根据实际情况修改路径
    DATA_TWO_FILE = root /  "data_analyse" / "result" / "capec_down.jsonl"      # 数据二文件（CAPEC -> CVE）
    CVE_DATA_DIR =  root /  "data_analyse" / "result" / "cve2attack"       # CVE 按年份存放的目录

    # 第一步：从数据二中提取所有唯一 CVE
    all_target_cves = set()
    data_two_records = []  # 保存原始记录用于后续匹配

    print("从 capec_down.jsonl 中提取 CVE...")
    with open(DATA_TWO_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                cves = record.get("cves", [])
                capec_id = record.get("capec_id")
                if cves and capec_id:
                    # 保存记录用于后续验证
                    data_two_records.append((capec_id, cves))
                    all_target_cves.update(cves)
            except json.JSONDecodeError:
                continue

    print(f"共提取 {len(all_target_cves)} 个唯一 CVE，来自 {len(data_two_records)} 条capec_down.jsonl")

    # 第二步：只加载这些 CVE 对应的年份文件
    cve_to_capecs = load_target_cve_capecs(all_target_cves, CVE_DATA_DIR)

    print(f"成功加载 {len(cve_to_capecs)} 个目标 CVE 的 CAPEC 信息")

    # 第三步：进行匹配
    matched_cves = set()
    for capec_id, cves in data_two_records:
        for cve in cves:
            if cve in cve_to_capecs:
                if capec_id in cve_to_capecs[cve]:
                    matched_cves.add(cve)
                    print(f"✅ 匹配: CVE={cve}, CAPEC={capec_id}")

    # 输出结果
    print("\n" + "="*60)
    print(f"总计匹配到 {len(matched_cves)} 个 CVE")
    if matched_cves:
        print("匹配的 CVE 列表（按字母序）:")
        for cve in sorted(matched_cves):
            print(f"  - {cve}")
    else:
        print("❌ 未找到任何匹配项")

if __name__ == "__main__":
    main()