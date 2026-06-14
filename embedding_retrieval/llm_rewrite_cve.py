#!/usr/bin/env python3
"""
LLM-based CVE description rewriter (V3).

Translates CVE descriptions from mechanism language ("what's broken")
into attacker-action language ("how would an attacker exploit this"),
in the style of MITRE ATT&CK technique descriptions.

This is a preprocessing step; output goes into a JSON cache that
run_embedding_retrieval.py reads via --rewrite-cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional
from urllib import error, request

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("rewrite")

# ---------- LLM config (adapted from temp_stage2 branch) ----------
LLM_BASE_URL = "http://172.23.216.73:11434/api/generate"
LLM_MODEL = "sec-i1"
LLM_TIMEOUT = 120
LLM_MAX_RETRIES = 3


def _call_llm(system: str, user: str) -> str:
    """Call local Ollama-compatible /api/generate endpoint with retries."""
    payload = json.dumps({
        "model": LLM_MODEL,
        "system": system,
        "prompt": user,
        "stream": False,
    }).encode("utf-8")

    last_exc = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        req = request.Request(
            LLM_BASE_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        time.sleep(0.1)
        try:
            with request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw).get("response", "").strip()
        except (error.URLError, error.HTTPError, TimeoutError,
                json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt < LLM_MAX_RETRIES:
                delay = float(2 ** (attempt - 1))
                LOGGER.warning("LLM retry %s/%s: %s", attempt,
                               LLM_MAX_RETRIES, exc)
                time.sleep(delay)

    raise RuntimeError(f"LLM request failed after retries: {last_exc}")


# ---------- CWE XML parser ----------

def load_cwe_db(xml_path: Path) -> Dict[str, Dict[str, str]]:
    """Parse MITRE CWE XML into {cwe_id: {name, description}}."""
    ns = {"cwe": "http://cwe.mitre.org/cwe-7"}
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    db: Dict[str, Dict[str, str]] = {}
    for weakness in root.iterfind("cwe:Weaknesses/cwe:Weakness", ns):
        cwe_id = weakness.get("ID", "").strip()
        if not cwe_id:
            continue
        name_el = weakness.find("cwe:Name", ns)
        desc_el = weakness.find("cwe:Description", ns)
        name = name_el.text.strip() if name_el is not None and name_el.text else ""
        description = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
        if name or description:
            db[cwe_id] = {"name": name, "description": description}
    return db


# ---------- Data loading ----------

def _year_from_cve_id(cve_id: str) -> str:
    parts = cve_id.split("-")
    if len(parts) < 3 or not parts[1].isdigit():
        raise ValueError(f"Bad CVE id: {cve_id}")
    return parts[1]


def load_cve2attack_entries(data_dir: Path) -> list:
    """Load all CVE entries from Validate_data/cve2attack/ JSONL files."""
    entries = []
    for jsonl_path in sorted(data_dir.glob("CVE-*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entries.append(json.loads(line))
    return entries


def load_cve_description(cve_id: str, cve_dir: Path) -> Optional[str]:
    """Load the raw English description for a given CVE ID."""
    year = _year_from_cve_id(cve_id)
    year_file = cve_dir / f"CVE-{year}.json"
    if not year_file.exists():
        return None
    with year_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    record = data.get(cve_id) if isinstance(data, dict) else None
    if not isinstance(record, dict):
        return None
    desc = (record.get("description") or "").strip()
    return desc if desc else None


# ---------- Prompt builder ----------

SYSTEM_PROMPT = (
    "You are an offensive cybersecurity analyst with deep expertise in the "
    "MITRE ATT&CK framework. Your task is to translate vulnerability (CVE) "
    "descriptions from passive, mechanism-focused language into active, "
    "attacker-action language that matches ATT&CK technique descriptions."
)

USER_PROMPT_TEMPLATE = (
    "CVE Description: {cve_desc}\n\n"
    "CWE Information:\n{cwe_info}\n\n"
    "Rewrite the above into a concise paragraph (3\u20135 sentences) describing the "
    "attacker\u2019s actions in ATT&CK style. Cover: 1) exploitation method, "
    "2) primary impact (capabilities gained), 3) secondary impact (what the "
    "attacker can do next). Use action-oriented language. Output ONLY the "
    "paragraph, no other text."
)


def build_prompt(cve_desc: str, cwe_ids: list,
                 cwe_db: Dict[str, Dict[str, str]]) -> str:
    """Construct the user prompt with CVE description + CWE context (system prompt is separate)."""
    cwe_lines = []
    for cwe_id in cwe_ids:
        info = cwe_db.get(cwe_id)
        if info:
            cwe_lines.append(f"CWE-{cwe_id}: {info['name']}")
            if info["description"]:
                cwe_lines.append(f"  Description: {info['description']}")
        else:
            cwe_lines.append(f"CWE-{cwe_id}: (no description available)")

    cwe_info = "\n".join(cwe_lines) if cwe_lines else "No CWE information available."
    return USER_PROMPT_TEMPLATE.format(cve_desc=cve_desc, cwe_info=cwe_info)


# ---------- Cache ----------

def load_rewrite_cache(cache_path: Path) -> Dict[str, str]:
    """Load existing rewrite cache (cve_id -> rewritten_text)."""
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_rewrite_cache(cache: Dict[str, str], cache_path: Path) -> None:
    """Persist rewrite cache as JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------- Main ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM rewrite of CVE descriptions for V3 retrieval.")
    p.add_argument("--cve2attack-dir", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "Validate_data" / "cve2attack",
                   help="Directory with CVE ground-truth JSONL files.")
    p.add_argument("--cve-dir", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "og_data" / "cve",
                   help="Directory with raw CVE JSON dicts.")
    p.add_argument("--cwe-xml", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "og_data" / "cwe.xml",
                   help="Path to MITRE cwe.xml.")
    p.add_argument("--cache-path", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "output" / "retrieval" / "llm_rewritten"
                   / "rewrite_cache.json",
                   help="Where to save the rewrite cache.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Number of concurrent LLM workers.")
    p.add_argument("--max-entries", type=int, default=0,
                   help="Process at most N entries (0 = all).")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore existing cache; rewrite everything.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load CWE database
    print(f"[INFO] Loading CWE database from {args.cwe_xml} ...")
    cwe_db = load_cwe_db(args.cwe_xml)
    print(f"[INFO] Loaded {len(cwe_db)} CWE entries.")

    # 2. Load CVE entries from CVE2ATT&CK evaluation set
    entries = load_cve2attack_entries(args.cve2attack_dir)
    total = len(entries)
    print(f"[INFO] Loaded {total} CVE entries from {args.cve2attack_dir}.")

    if args.max_entries > 0:
        entries = entries[:args.max_entries]
        print(f"[INFO] Capped to {len(entries)} entries.")

    # 3. Load existing cache
    cache = {} if args.no_cache else load_rewrite_cache(args.cache_path)
    print(f"[INFO] Cache has {len(cache)} entries (path: {args.cache_path}).")

    # 4. Determine which CVEs need rewriting
    to_process = []
    already_cached = 0
    missing_desc = 0

    for entry in entries:
        cve_id = entry["cve_id"]
        if cve_id in cache:
            already_cached += 1
            continue
        cve_desc = load_cve_description(cve_id, args.cve_dir)
        if not cve_desc:
            missing_desc += 1
            continue
        to_process.append({
            "cve_id": cve_id,
            "cve_desc": cve_desc,
            "cwe_ids": entry.get("cwes", []),
        })

    print(f"[INFO] Already cached: {already_cached}")
    print(f"[INFO] Missing description: {missing_desc}")
    print(f"[INFO] Need LLM rewrite: {len(to_process)}")

    if not to_process:
        print("[INFO] Nothing to process. Exiting.")
        return

    # 5. Rewrite with thread pool
    success = 0
    fail = 0

    def _rewrite_one(item):
        try:
            user_prompt = build_prompt(
                cve_desc=item["cve_desc"],
                cwe_ids=item["cwe_ids"],
                cwe_db=cwe_db,
            )
            rewritten = _call_llm(system=SYSTEM_PROMPT, user=user_prompt)
            if not rewritten:
                return None
            return (item["cve_id"], rewritten)
        except Exception as exc:
            LOGGER.error("Rewrite failed for %s: %s", item["cve_id"], exc)
            return None

    print(f"[INFO] Starting {args.num_workers} workers ...")
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_rewrite_one, item): item
                   for item in to_process}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            if result:
                cve_id, rewritten = result
                cache[cve_id] = rewritten
                success += 1
            else:
                fail += 1
            # Save periodically
            if i % 20 == 0 or i == len(to_process):
                save_rewrite_cache(cache, args.cache_path)
                pct = i / len(to_process) * 100
                elapsed = time.monotonic() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"\r[{i}/{len(to_process)} {pct:.0f}%] "
                      f"ok={success} fail={fail} {rate:.1f} item/s  ",
                      end="", flush=True)

    if len(to_process) > 0:
        print()

    elapsed = time.monotonic() - t0
    print(f"[INFO] Done in {elapsed:.0f}s. Success: {success}, Fail: {fail}.")
    print(f"[INFO] Cache saved: {args.cache_path}")


if __name__ == "__main__":
    main()
