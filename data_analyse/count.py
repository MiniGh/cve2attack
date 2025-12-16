import os
import glob
import json
from collections import Counter
from typing import Counter as CounterType

def count_cwe_distribution(base_dir="./result/cve2cwe", start_year=1999, end_year=2025):
    """
    统计指定目录下所有 CVE-*.jsonl 文件中每个 CVE 条目所含 CWE 数量的分布。

    参数:
        base_dir (str): 存放 JSONL 文件的目录路径，默认为 './result/cve2cwe'
        start_year (int): 起始年份（包含）
        end_year (int): 结束年份（包含）

    返回:
        collections.Counter: 键为 CWE 数量（int），值为具有该数量的 CVE 条目数
    """
    # 构建匹配的文件路径模式：CVE-1999.jsonl 到 CVE-2025.jsonl
    file_paths = []
    for year in range(start_year, end_year + 1):
        pattern = os.path.join(base_dir, f"CVE-{year}.jsonl")
        file_paths.extend(glob.glob(pattern))

    cwe_counts = []

    for file_path in file_paths:
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                    try:
                        record = json.loads(line)
                        cwes = record.get("cwes", [])
                        # 确保 cwes 是列表（兼容性处理）
                        if not isinstance(cwes, list):
                            cwes = []
                        cwe_counts.append(len(cwes))
                    except json.JSONDecodeError as e:
                        print(f"警告：{file_path} 第 {line_num} 行 JSON 解析失败: {e}")
                        continue
        except Exception as e:
            print(f"错误：无法读取文件 {file_path}: {e}")

    return Counter(cwe_counts)

def count_target_distribution(
    file_path: str,
    target_key: str,
    id_key: str = None  # 可选：用于调试或未来扩展
) -> CounterType[int]:
    """
    通用函数：统计 JSONL 文件中每条记录的某个列表字段的长度分布。

    参数:
        file_path (str): JSONL 文件路径
        target_key (str): 要统计长度的字段名（如 'capecs' 或 'techniques'）
        id_key (str, optional): ID 字段名（如 'cwe_id'），仅用于错误提示

    返回:
        Counter: 键为列表长度（int），值为出现次数
    """
    counts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    targets = record.get(target_key, [])
                    if not isinstance(targets, list):
                        targets = []
                    counts.append(len(targets))
                except json.JSONDecodeError as e:
                    print(f"警告：{file_path} 第 {line_num} 行 JSON 解析失败: {e}")
    except FileNotFoundError:
        print(f"错误：文件未找到 - {file_path}")
        return Counter()
    except Exception as e:
        print(f"错误：读取 {file_path} 时发生异常: {e}")
        return Counter()

    return Counter(counts)


def count_cwe_to_capec_distribution(file_path: str = "./result/cwe2capec.jsonl") -> CounterType[int]:
    """统计每个 CWE 映射到多少个 CAPEC 的分布"""
    return count_target_distribution(file_path, target_key="capecs", id_key="cwe_id")


def count_capec_to_technique_distribution(file_path: str = "./result/capec2techniques.jsonl") -> CounterType[int]:
    """统计每个 CAPEC 映射到多少个 Technique 的分布"""
    return count_target_distribution(file_path, target_key="techniques", id_key="capec_id")


# 使用示例
if __name__ == "__main__":
    # 1. CVE -> CWE 分布
    # distribution = count_cwe_distribution()

    # print("CWE 数量分布（每个 CVE 对应的 CWE 个数）:")
    # total_cves = sum(distribution.values())
    # print(f"总计 CVE 条目数: {total_cves}\n")

    # for cwe_num in sorted(distribution):
    #     count = distribution[cwe_num]
    #     percentage = (count / total_cves) * 100 if total_cves > 0 else 0
    #     print(f"{cwe_num:2d} 个 CWE: {count:6d} 个 CVE ({percentage:5.2f}%)")
    

    # 2. CWE → CAPEC 分布
    print("=== CWE → CAPEC 映射分布 ===")
    cwe_capec_dist = count_cwe_to_capec_distribution()
    total_cwes = sum(cwe_capec_dist.values())
    print(f"总计 CWE 条目数: {total_cwes}\n")
    for n in sorted(cwe_capec_dist):
        cnt = cwe_capec_dist[n]
        pct = (cnt / total_cwes * 100) if total_cwes else 0
        print(f"{n:2d} 个 CAPEC: {cnt:6d} 个 CWE ({pct:5.2f}%)")

    print("\n" + "="*40 + "\n")

    # 3. CAPEC → Technique 分布
    print("=== CAPEC → Technique 映射分布 ===")
    capec_tech_dist = count_capec_to_technique_distribution()
    total_capecs = sum(capec_tech_dist.values())
    print(f"总计 CAPEC 条目数: {total_capecs}\n")
    for n in sorted(capec_tech_dist):
        cnt = capec_tech_dist[n]
        pct = (cnt / total_capecs * 100) if total_capecs else 0
        print(f"{n:2d} 个 Technique: {cnt:6d} 个 CAPEC ({pct:5.2f}%)")