#!/usr/bin/env python3
"""TF-IDF based CVE → CWE mapping (existing or missing mappings)."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_cwe_corpus(cwe_db: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    for cwe_id, meta in cwe_db.items():
        parts = [meta.get("name", ""), meta.get("description", ""), meta.get("extended_description", "")]
        text = " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())
        if text:
            ids.append(cwe_id)
            texts.append(text)
    return ids, texts


def load_all_cves(cve_dir: Path) -> Dict[str, dict]:
    merged: Dict[str, dict] = {}
    for cve_file in sorted(cve_dir.glob("CVE-*.json")):
        with cve_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            merged.update(data)
    return merged


def year_of_cve(cve_id: str) -> str:
    try:
        return cve_id.split("-")[1]
    except Exception:
        return "unknown"


def recommend(text: str, vectorizer: TfidfVectorizer, target_matrix, target_ids: List[str], top_k: int, threshold: float) -> List[Dict[str, float]]:
    if not text.strip():
        return []
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, target_matrix).flatten()
    order = np.argsort(sims)[::-1]
    results: List[Dict[str, float]] = []
    for idx in order[: top_k * 2]:  # grab a bit more then filter by threshold
        score = float(sims[idx])
        if score < threshold:
            continue
        results.append({"cwe_id": target_ids[idx], "score": round(score, 3)})
        if len(results) >= top_k:
            break
    return results


def recall_at_10(truth: List[str], recs: List[Dict[str, float]]) -> float:
    truth_set = set(str(t) for t in truth)
    if not truth_set:
        return 0.0
    top10 = recs[:10]
    return 1.0 if any(r["cwe_id"] in truth_set for r in top10) else 0.0


def mrr(truth: List[str], recs: List[Dict[str, float]]) -> float:
    truth_set = set(str(t) for t in truth)
    if not truth_set:
        return 0.0
    for rank, rec in enumerate(recs, start=1):
        if rec["cwe_id"] in truth_set:
            return 1.0 / rank
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF CVE→CWE mapping")
    parser.add_argument("--mode", choices=["existing", "missing"], default="existing", help="existing: evaluate CVEs with known CWEs from data_analyse/result; missing: predict for others")
    parser.add_argument("--top_k", type=int, default=20, help="Max recommendations per item")
    parser.add_argument("--threshold", type=float, default=0.05, help="Similarity threshold")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cwe_path = root / "source" / "cwe_db.json"
    cve_dir = root / "source" / "cve"
    truth_dir = root / "data_analyse" / "result" / "cve2cwe"
    result_dir = root / "result" / ("existing_cve2cwe" if args.mode == "existing" else "cve2cwe")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / (("existing_" if args.mode == "existing" else "") + "cve2cwe.log")

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    # Header
    sep = "=" * 70
    log("\n" + sep)
    log(f"🚀 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"📂 CVE dir: {cve_dir}")
    log(f"📂 CWE path: {cwe_path}")
    log(f"📂 Truth dir: {truth_dir}")
    log(f"📤 Output dir: {result_dir}")
    log(f"📄 Log file: {log_path}")
    log(f"⚙️  Mode: {args.mode}, top_k: {args.top_k}, threshold: {args.threshold}")
    log(sep + "\n")

    log(f"[✓] Mode: {args.mode}")
    log(f"[✓] Loading CWE DB from {cwe_path}")
    with cwe_path.open("r", encoding="utf-8") as f:
        cwe_db = json.load(f)

    cwe_ids, cwe_texts = build_cwe_corpus(cwe_db)
    log(f"[✓] Loaded {len(cwe_ids)} CWE documents for corpus")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.85, min_df=2, token_pattern=r"(?u)\b\w\w+\b", lowercase=True)
    cwe_matrix = vectorizer.fit_transform(cwe_texts)
    log("[✓] TF-IDF vectorizer fitted on CWE corpus")

    # Load CVE descriptions (cleaned, no cwes) and existing truth mappings
    all_cves = load_all_cves(cve_dir)
    existing_truth: Dict[str, List[str]] = {}
    if truth_dir.is_dir():
        for jf in sorted(truth_dir.glob("CVE-*.jsonl")):
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    cve_id = obj.get("cve_id")
                    if not cve_id:
                        continue
                    truth_list = obj.get("original_cwes") or obj.get("cwes") or []
                    if truth_list:
                        existing_truth[cve_id] = list(truth_list)
    log(f"[✓] Loaded {len(existing_truth)} CVEs with existing CWE mappings from {truth_dir}")

    touched_files: set[Path] = set()

    def write_record(path: Path, record: dict):
        mode = "a"
        if path not in touched_files:
            mode = "w"  # truncate on first write per file for clean reruns
            touched_files.add(path)
        with path.open(mode, encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")

    total_items = 0
    written = 0
    hit_sum = 0.0
    mrr_sum = 0.0
    eval_count = 0

    if args.mode == "existing":
        for cve_id, truth in existing_truth.items():
            desc = (all_cves.get(cve_id, {}).get("description") or "").strip()
            total_items += 1
            recs = recommend(desc, vectorizer, cwe_matrix, cwe_ids, args.top_k, args.threshold)
            eval_count += 1
            hit_sum += recall_at_10(truth, recs)
            mrr_sum += mrr(truth, recs)

            record = {"cve_id": cve_id, "original_cwes": truth, "recommendations": recs}
            year = year_of_cve(cve_id)
            out_path = result_dir / f"CVE-{year}.jsonl"
            write_record(out_path, record)
            written += 1

    else:  # missing
        existing_ids = set(existing_truth.keys())
        for cve_id, meta in all_cves.items():
            if cve_id in existing_ids:
                continue
            desc = (meta.get("description") or "").strip()
            total_items += 1
            recs = recommend(desc, vectorizer, cwe_matrix, cwe_ids, args.top_k, args.threshold)
            record = {"cve_id": cve_id, "original_cwes": [], "recommendations": recs}
            year = year_of_cve(cve_id)
            out_path = result_dir / f"CVE-{year}.jsonl"
            write_record(out_path, record)
            written += 1

    if args.mode == "existing" and eval_count:
        recall10 = hit_sum / eval_count
        mrr_score = mrr_sum / eval_count
        log(f"[★] Eval Recall@10: {recall10:.4f} | MRR: {mrr_score:.4f} over {eval_count} CVEs")
    log(f"[✓] Done. processed={total_items}, written={written}, log={log_path}")


if __name__ == "__main__":
    main()
