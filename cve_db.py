import requests
import json
import os
import time
import re
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# ====== 配置 ======
API_KEY = "YOUR_API_KEY_HERE"  # ← ← ← 替换为你的 API Key（强烈建议）
HEADERS = {"apiKey": API_KEY} if API_KEY != "YOUR_API_KEY_HERE" else {}
BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

OUTPUT_DIR = Path("./source/cve")
OUTPUT_DIR.mkdir(exist_ok=True)

# 分页参数（NVD 限制：max 2000/page）
RESULTS_PER_PAGE = 2000

# 重试配置
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒


def fetch_with_retry(url, params, headers, retries=MAX_RETRIES):
    for i in range(retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                print(f"⚠️  Rate limited (403). Retrying in {RETRY_DELAY}s... (attempt {i+1}/{retries+1})")
            elif response.status_code == 503:
                print(f"⚠️  Service unavailable (503). Retrying in {RETRY_DELAY}s...")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
            if i < retries:
                time.sleep(RETRY_DELAY * (2 ** i))  # 指数退避
        except Exception as e:
            print(f"❌ Request failed: {e}. Retrying in {RETRY_DELAY}s...")
            if i < retries:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries")


def extract_cwe_list(weaknesses):
    """按你的逻辑：优先 Primary，再 Secondary；只取 CWE-数字"""
    primary_cwes = []
    secondary_cwes = []
    
    for w in weaknesses:
        w_type = w.get("type", "")
        desc_list = w.get("description", [])
        if not desc_list:
            continue
        cwe_val = desc_list[0].get("value", "")
        # 匹配 CWE-123 或 CWE-1234（严格）
        match_obj = re.match(r"CWE-(\d{1,5})", cwe_val)
        if not match_obj:
            continue
        cwe_num = match_obj.group(1)
        
        if w_type == "Primary":
            primary_cwes.append(cwe_num)
        elif w_type == "Secondary":
            secondary_cwes.append(cwe_num)
    
    # 优先用 Primary；若无，则用 Secondary
    return sorted(set(primary_cwes)) if primary_cwes else sorted(set(secondary_cwes))


def main():
    print("[🚀] Starting full NVD CVE fetch (by year)...")
    
    # Step 1: 获取总数（用于分页）
    print("[🔍] Fetching total CVE count...")
    try:
        data = fetch_with_retry(BASE_URL, {"resultsPerPage": 1}, HEADERS)
        total_results = data["totalResults"]
        print(f"[✅] Total CVEs: {total_results:,}")
    except Exception as e:
        print(f"[❌] Failed to get total: {e}")
        return

    # Step 2: 分页拉取
    total_pages = (total_results + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
    print(f"[📊] Total pages: {total_pages}")

    year_buckets = {}  # { "2017": { "CVE-2017-0001": {...}, ... }, ... }

    for page in tqdm(range(total_pages), desc="Pages", unit="page"):
        start_index = page * RESULTS_PER_PAGE
        params = {
            "resultsPerPage": RESULTS_PER_PAGE,
            "startIndex": start_index
        }

        # 拉取一页
        try:
            data = fetch_with_retry(BASE_URL, params, HEADERS)
        except Exception as e:
            print(f"\n[🛑] Page {page} failed. Skip. ({e})")
            continue

        # 处理每个 CVE
        for item in tqdm(data.get("vulnerabilities", []), desc=f"Page {page}", leave=False, unit="CVE"):
            cve_obj = item.get("cve", {})
            cve_id = cve_obj.get("id", "")
            if not cve_id or not cve_id.startswith("CVE-"):
                continue

            # 提取年份
            try:
                year = cve_id.split("-")[1]
                if not year.isdigit() or len(year) != 4:
                    year = "unknown"
            except:
                year = "unknown"

            # 提取英文描述
            description = ""
            for desc in cve_obj.get("descriptions", []):
                if desc.get("lang") == "en":
                    description = desc.get("value", "").strip()
                    break

            # 提取 CWEs（按你的逻辑）
            weaknesses = cve_obj.get("weaknesses", [])
            cwe_list = extract_cwe_list(weaknesses)

            # 构造精简结构
            record = {
                "id": cve_id,
                "description": description,
                "cwes": cwe_list
            }

            # 加入年份 bucket
            if year not in year_buckets:
                year_buckets[year] = {}
            year_buckets[year][cve_id] = record

        # 每 10 页或最后一页保存一次（防内存溢出）
        if (page + 1) % 10 == 0 or page == total_pages - 1:
            for y, data in year_buckets.items():
                if not data:
                    continue
                out_file = OUTPUT_DIR / f"CVE-{y}.json"
                # 合并：若文件已存在，读取后更新
                if out_file.exists():
                    with open(out_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    existing.update(data)
                else:
                    existing = data
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2, ensure_ascii=False)
            year_buckets.clear()  # 清空内存
            print(f"[💾] Saved up to page {page + 1}")

    print(f"\n[✅] Done! Files saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
