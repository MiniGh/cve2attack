#!/usr/bin/env python3
"""
统计按年份存放的 CVE JSON 文件中每个 CVE 条目的 "cwes" 数量分布，
并将本次统计结果以带时间戳的方式追加写入 count_cwes.log（默认）。

用法示例:
  python count_cwes.py
  python count_cwes.py --dir ./data
  python count_cwes.py --files a.json b.json
  python count_cwes.py --json-out out.json
  python count_cwes.py --log my_log.log
"""
import argparse
import json
import glob
import os
from collections import Counter
from datetime import datetime

def process_file(path, counter):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"跳过无法读取或解析的文件 {path}: {e}")
        return 0

    processed = 0

    # 情况1: 顶层为 dict，key 是 "CVE-..."，value 是对象
    if isinstance(data, dict):
        items = data.items()
    # 情况2: 顶层为 list，每项是 CVE 对象
    elif isinstance(data, list):
        items = [(None, obj) for obj in data]
    else:
        print(f"文件 {path} 的顶级 JSON 不是 dict 或 list，跳过。")
        return 0

    for key, entry in items:
        if not isinstance(entry, dict):
            continue
        cwes = entry.get('cwes', [])
        if not isinstance(cwes, list):
            count = 0
        else:
            count = len(cwes)
        counter[count] += 1
        processed += 1

    return processed

def format_results(counter, total):
    """
    返回一个字符串列表，每行对应要打印/写入日志的内容。
    """
    lines = []
    for cwe_count in sorted(counter.keys()):
        lines.append(f"包含{cwe_count}个cwe的cve数量是：{counter[cwe_count]}个")
    lines.append(f"已处理 CVE 条目总数：{total} 个")
    return lines

def append_log(log_path, lines):
    """
    将本次统计结果以时间戳追加写入日志文件。
    """
    try:
        with open(log_path, 'a', encoding='utf-8') as lf:
            ts = datetime.now().astimezone().isoformat(sep=' ')
            lf.write(f"=== {ts} ===\n")
            for ln in lines:
                lf.write(ln + "\n")
            lf.write("\n")
    except Exception as e:
        print(f"写入日志文件 {log_path} 失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="统计 CVE 中 cwes 数量分布并写入日志")
    parser.add_argument("--dir", "-d", default=".", help="包含 CVE-*.json 文件的目录（默认当前目录）")
    parser.add_argument("--pattern", "-p", default="CVE-*.json", help="文件匹配模式（默认 CVE-*.json）")
    parser.add_argument("--files", "-f", nargs="*", help="显式指定要处理的文件（覆盖 --dir/--pattern）")
    parser.add_argument("--json-out", "-j", help="将结果以 JSON 写入指定文件")
    parser.add_argument("--log", default="count_cwes.log", help="追加写入的日志文件路径（默认 count_cwes.log）")
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        search = os.path.join(args.dir, args.pattern)
        files = glob.glob(search)
    if not files:
        print("未找到任何文件，检查目录和匹配模式是否正确。")
        return

    counter = Counter()
    total = 0
    for p in sorted(files):
        processed = process_file(p, counter)
        total += processed

    lines = format_results(counter, total)

    # 打印到 stdout
    for ln in lines:
        print(ln)

    # 追加到日志
    append_log(args.log, lines)
    print(f"统计结果已追加到日志文件：{args.log}")

    # 可选的 JSON 输出
    if args.json_out:
        try:
            with open(args.json_out, 'w', encoding='utf-8') as jf:
                json.dump({str(k): v for k, v in sorted(counter.items())}, jf, ensure_ascii=False, indent=2)
            print(f"已将直方图写入 {args.json_out}")
        except Exception as e:
            print(f"写入 JSON 输出文件失败: {e}")

if __name__ == "__main__":
    main()
