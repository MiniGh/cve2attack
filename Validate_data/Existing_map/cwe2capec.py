#!/usr/bin/env python3
"""TF-IDF based CWE → CAPEC mapping (existing or missing mappings)."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_capec_corpus(capecs: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    for capec_id, meta in capecs.items():
        parts = [meta.get("name", ""), meta.get("description", ""), meta.get("extended_description", "")]
        text = " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())
        if text:
            ids.append(capec_id)
            texts.append(text)
    return ids, texts


def get_cwe_text(cwe: dict) -> str:
    parts = [cwe.get("name", ""), cwe.get("description", ""), cwe.get("extended_description", "")]
    return " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())


def recommend(text: str, vectorizer: TfidfVectorizer, target_matrix, target_ids: List[str], top_k: int, threshold: float):
    if not text.strip():
        return []
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, target_matrix).flatten()
    order = np.argsort(sims)[::-1]
    recs = []
    for idx in order[: top_k * 2]:
        score = float(sims[idx])
        if score < threshold:
            continue
        recs.append({"capec_id": target_ids[idx], "score": round(score, 3)})
        if len(recs) >= top_k:
            break
    return recs


def recall_at_10(truth: List[str], recs: List[dict]) -> float:
    truth_set = set(str(t) for t in truth)
    if not truth_set:
        return 0.0
    return 1.0 if any(r["capec_id"] in truth_set for r in recs[:10]) else 0.0


def mrr(truth: List[str], recs: List[dict]) -> float:
    truth_set = set(str(t) for t in truth)
    if not truth_set:
        return 0.0
    for rank, rec in enumerate(recs, start=1):
        if rec["capec_id"] in truth_set:
            return 1.0 / rank
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF CWE→CAPEC mapping")
    parser.add_argument("--mode", choices=["existing", "missing"], default="existing", help="existing: evaluate CWEs with CAPECs from data_analyse/result; missing: predict for CWEs without CAPECs")
    parser.add_argument("--top_k", type=int, default=20, help="Max recommendations")
    parser.add_argument("--threshold", type=float, default=0.05, help="Similarity threshold")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cwe_path = root / "source" / "cwe_db.json"
    capec_path = root / "source" / "capec_db.json"
    truth_path = root / "data_analyse" / "result" / "cwe2capec.jsonl"
    result_path = root / "result" / (("existing_" if args.mode == "existing" else "") + "cwe2capec.jsonl")
    log_path = root / "result" / (("existing_" if args.mode == "existing" else "") + "cwe2capec.log")
    os.makedirs(result_path.parent, exist_ok=True)

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    # Header
    sep = "=" * 70
    log("\n" + sep)
    log(f"🚀 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"📂 CWE path: {cwe_path}")
    log(f"📂 CAPEC path: {capec_path}")
    log(f"📂 Truth path: {truth_path}")
    log(f"📤 Output path: {result_path}")
    log(f"📄 Log file: {log_path}")
    log(f"⚙️  Mode: {args.mode}, top_k: {args.top_k}, threshold: {args.threshold}")
    log(sep + "\n")

    log(f"[✓] Mode: {args.mode}")
    log(f"[✓] Loading CWE DB: {cwe_path}")
    with cwe_path.open("r", encoding="utf-8") as f:
        cwe_db = json.load(f)
    log(f"[✓] Loading CAPEC DB: {capec_path}")
    with capec_path.open("r", encoding="utf-8") as f:
        capec_db = json.load(f)

    capec_ids, capec_texts = build_capec_corpus(capec_db)
    log(f"[✓] CAPEC corpus size: {len(capec_ids)}")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.85, min_df=2, token_pattern=r"(?u)\b\w\w+\b", lowercase=True)
    capec_matrix = vectorizer.fit_transform(capec_texts)
    log("[✓] TF-IDF vectorizer fitted on CAPEC corpus")

    # Load truth mappings from data_analyse output (non-empty only)
    existing_truth: Dict[str, List[str]] = {}
    if truth_path.is_file():
        with truth_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cwe_id = obj.get("cwe_id")
                if not cwe_id:
                    continue
                truth_list = obj.get("original_capecs") or obj.get("capecs") or []
                if truth_list:
                    existing_truth[cwe_id] = list(truth_list)
    log(f"[✓] Loaded {len(existing_truth)} CWEs with CAPEC mappings from {truth_path}")

    touched_files: set[Path] = set()

    def write_record(path: Path, record: dict):
        mode = "a"
        if path not in touched_files:
            mode = "w"  # truncate on first write per run
            touched_files.add(path)
        with path.open(mode, encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")

    total = 0
    written = 0
    eval_n = 0
    hit_sum = 0.0
    mrr_sum = 0.0

    if args.mode == "existing":
        for cwe_id, truth in existing_truth.items():
            meta = cwe_db.get(cwe_id, {})
            text = get_cwe_text(meta)
            total += 1
            recs = recommend(text, vectorizer, capec_matrix, capec_ids, args.top_k, args.threshold)
            eval_n += 1
            hit_sum += recall_at_10(truth, recs)
            mrr_sum += mrr(truth, recs)

            record = {"cwe_id": cwe_id, "original_capecs": truth, "recommendations": recs}
            write_record(result_path, record)
            written += 1

    else:  # missing
        existing_ids = set(existing_truth.keys())
        for cwe_id, meta in cwe_db.items():
            if cwe_id in existing_ids:
                continue
            text = get_cwe_text(meta)
            total += 1
            recs = recommend(text, vectorizer, capec_matrix, capec_ids, args.top_k, args.threshold)
            record = {"cwe_id": cwe_id, "original_capecs": [], "recommendations": recs}
            write_record(result_path, record)
            written += 1

    if args.mode == "existing" and eval_n:
        log(f"[★] Eval Recall@10: {hit_sum/eval_n:.4f} | MRR: {mrr_sum/eval_n:.4f} over {eval_n} CWEs")
    log(f"[✓] Done. processed={total}, written={written}, result={result_path}")


if __name__ == "__main__":
    main()
