# import json

# input_file = "./data_analyse/result/capec_down.jsonl"
# count = 0

# with open(input_file, "r", encoding="utf-8") as f:
#     for line in f:
#         obj = json.loads(line)

#         # 判断是否包含 cves 且不为空
#         if "cves" in obj and obj["cves"]:
#             count += 1
#             # 原样打印完整 JSON 行
#             print(json.dumps(obj, ensure_ascii=False))

# print(f"\n包含 cves 的记录数：{count}")
################# 运行结果 ######################
# {"capec_id": "10", "cwes": ["118", "119", "120", "20", "302", "680", "697", "733", "74", "99"], "cves": ["CVE-1999-0046", "CVE-1999-0906"]}
# {"capec_id": "108", "cwes": ["114", "20", "74", "78", "89"], "cves": ["CVE-2006-6799"]}
# {"capec_id": "13", "cwes": ["15", "20", "200", "285", "302", "353", "73", "74"], "cves": ["CVE-1999-0073"]}
# {"capec_id": "135", "cwes": ["134", "20", "74"], "cves": ["CVE-2007-2027"]}
# {"capec_id": "136", "cwes": ["20", "77", "90"], "cves": ["CVE-2005-2301"]}
# {"capec_id": "26", "cwes": ["1223", "1254", "1298", "362", "363", "366", "368", "370", "662", "665", "667", "689"], "cves": ["CVE-2007-1057"]}
# {"capec_id": "267", "cwes": ["172", "173", "180", "181", "20", "692", "697", "73", "74"], "cves": ["CVE-2010-0488"]}
# {"capec_id": "27", "cwes": ["367", "61", "662", "667", "689"], "cves": ["CVE-2000-0972", "CVE-2005-0894", "CVE-2006-6939"]}
# {"capec_id": "273", "cwes": ["436", "444", "74"], "cves": ["CVE-2006-2786", "CVE-2017-2666"]}
# {"capec_id": "29", "cwes": ["362", "366", "367", "368", "370", "662", "663", "665", "691"], "cves": ["CVE-2007-1057"]}
# {"capec_id": "31", "cwes": ["113", "20", "302", "311", "315", "384", "472", "539", "565", "602", "642"], "cves": ["CVE-2010-5148", "CVE-2016-0353"]}
# {"capec_id": "33", "cwes": ["444"], "cves": ["CVE-2005-2088", "CVE-2006-6276", "CVE-2020-8287"]}
# {"capec_id": "34", "cwes": ["113", "138", "436", "74"], "cves": ["CVE-2006-0207"]}
# {"capec_id": "39", "cwes": ["233", "285", "302", "315", "353", "384", "472", "539", "565"], "cves": ["CVE-2006-0944"]}
# {"capec_id": "475", "cwes": ["295", "327", "347"], "cves": ["CVE-2020-0601"]}
# {"capec_id": "49", "cwes": ["257", "262", "263", "307", "308", "309", "521", "654"], "cves": ["CVE-2004-1143"]}
# {"capec_id": "54", "cwes": ["209"], "cves": ["CVE-2006-4705"]}
# {"capec_id": "55", "cwes": ["261", "262", "263", "308", "309", "521", "654", "916"], "cves": ["CVE-2006-1058"]}
# {"capec_id": "59", "cwes": ["200", "285", "290", "330", "331", "346", "384", "488", "539", "6", "693"], "cves": ["CVE-2001-1534", "CVE-2006-6969"]}
# {"capec_id": "60", "cwes": ["200", "285", "290", "294", "346", "384", "488", "539", "664", "732"], "cves": ["CVE-1999-0428", "CVE-2002-0258"]}
# {"capec_id": "61", "cwes": ["384", "664", "732"], "cves": ["CVE-2004-2182"]}
# {"capec_id": "657", "cwes": ["494"], "cves": ["CVE-2006-3976", "CVE-2006-3977"]}
# {"capec_id": "66", "cwes": ["1286", "89"], "cves": ["CVE-2006-5525"]}
# {"capec_id": "67", "cwes": ["120", "134", "20", "680", "697", "74"], "cves": ["CVE-2002-0412"]}
# {"capec_id": "697", "cwes": ["923"], "cves": ["CVE-2019-0547"]}
# {"capec_id": "7", "cwes": ["20", "209", "697", "707", "74", "89"], "cves": ["CVE-2006-4705"]}
# {"capec_id": "70", "cwes": ["262", "263", "308", "309", "521", "654", "798"], "cves": ["CVE-2006-5288"]}
# {"capec_id": "92", "cwes": ["120", "122", "128", "190", "196", "680", "697"], "cves": ["CVE-2007-1544"]}
# {"capec_id": "93", "cwes": ["117", "150", "75"], "cves": ["CVE-2006-0201"]}

# 包含 cves 的记录数：29
############################

############################
# 统计每年CVE中官方给的CWE映射为空的数量以及比例
############################
import os
import json
import sys

def analyze_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ 无法解析 JSON 文件: {filepath} - {e}")
            return None

    total = len(data)
    empty_cwes = 0

    for cve_id, entry in data.items():
        cwes = entry.get("cwes", [])
        if isinstance(cwes, list) and len(cwes) == 0:
            empty_cwes += 1

    if total == 0:
        ratio = 0.0
    else:
        ratio = empty_cwes / total

    return {
        "file": os.path.basename(filepath),
        "total_cves": total,
        "empty_cwes": empty_cwes,
        "ratio": ratio
    }

def main(target_path):
    if os.path.isfile(target_path):
        files = [target_path]
    elif os.path.isdir(target_path):
        files = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.endswith('.json')]
    else:
        print("❌ 输入路径既不是文件也不是目录。")
        sys.exit(1)

    results = []
    for file in sorted(files):
        res = analyze_file(file)
        if res:
            results.append(res)
            print(f"{res['file']}: 空 cwes = {res['empty_cwes']} / {res['total_cves']} ({res['ratio']:.2%})")

    # 可选：输出汇总
    if results:
        total_all = sum(r['total_cves'] for r in results)
        empty_all = sum(r['empty_cwes'] for r in results)
        overall_ratio = empty_all / total_all if total_all > 0 else 0
        print("\n📊 总体统计:")
        print(f"总计 CVE: {total_all}")
        print(f"空 cwes: {empty_all} ({overall_ratio:.2%})")

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("用法: python count_empty_cwes.py <json_file_or_directory>")
    #     sys.exit(1)
    # main(sys.argv[1])
    main("../og_source/cve")
################# 运行结果 ##################
# CVE-1999.json: 空 cwes = 1480 / 1579 (93.73%)
# CVE-2000.json: 空 cwes = 1202 / 1242 (96.78%)
# CVE-2001.json: 空 cwes = 1462 / 1556 (93.96%)
# CVE-2002.json: 空 cwes = 2075 / 2393 (86.71%)
# CVE-2003.json: 空 cwes = 1230 / 1555 (79.10%)
# CVE-2004.json: 空 cwes = 2454 / 2707 (90.65%)
# CVE-2005.json: 空 cwes = 4293 / 4769 (90.02%)
# CVE-2006.json: 空 cwes = 6106 / 7143 (85.48%)
# CVE-2007.json: 空 cwes = 3883 / 6580 (59.01%)
# CVE-2008.json: 空 cwes = 770 / 7177 (10.73%)
# CVE-2009.json: 空 cwes = 844 / 5052 (16.71%)
# CVE-2010.json: 空 cwes = 1189 / 5244 (22.67%)
# CVE-2011.json: 空 cwes = 1020 / 4896 (20.83%)
# CVE-2012.json: 空 cwes = 1538 / 5938 (25.90%)
# CVE-2013.json: 空 cwes = 1743 / 6823 (25.55%)
# CVE-2014.json: 空 cwes = 1651 / 9000 (18.34%)
# CVE-2015.json: 空 cwes = 1899 / 8768 (21.66%)
# CVE-2016.json: 空 cwes = 2354 / 10570 (22.27%)
# CVE-2017.json: 空 cwes = 4193 / 17035 (24.61%)
# CVE-2018.json: 空 cwes = 3465 / 17505 (19.79%)
# CVE-2019.json: 空 cwes = 3977 / 17084 (23.28%)
# CVE-2020.json: 空 cwes = 5300 / 20664 (25.65%)
# CVE-2021.json: 空 cwes = 4188 / 23123 (18.11%)
# CVE-2022.json: 空 cwes = 4820 / 27091 (17.79%)
# CVE-2023.json: 空 cwes = 3704 / 30483 (12.15%)
# CVE-2024.json: 空 cwes = 3008 / 38783 (7.76%)
# CVE-2025.json: 空 cwes = 2533 / 34852 (7.27%)

# 📊 总体统计:
# 总计 CVE: 319612
# 空 cwes: 72381 (22.65%)
#####################################################



