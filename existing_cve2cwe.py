#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVE → CWE Mapping via TF-IDF (Modular Design)
Author: Your Name
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from typing import List,Tuple,Set
from datetime import datetime
import sys
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# 1. Data Loading Modules
# -------------------------------

def load_cwe_db(filepath: str) -> Dict[str, dict]:
    """Load CWE database from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_cves_by_year(cve_dir: str) -> Dict[str, List[Tuple[str, dict]]]:
    """
    Load all CVEs grouped by year (e.g., 'CVE-2023').
    Returns: { 'CVE-2023': [('CVE-2023-1234', {...}), ...], ... }
    """
    cve_by_year = defaultdict(list)
    json_paths = sorted(glob.glob(os.path.join(cve_dir, "CVE-*.json")))
    
    for path in json_paths:
        year_key = Path(path).stem  # e.g., "CVE-2023"
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for cve_id, info in data.items():
                cve_by_year[year_key].append((cve_id, info))
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")
    
    return dict(cve_by_year)


# -------------------------------
# 2. Text Processing & Corpus Building
# -------------------------------

def build_cwe_text_corpus(cwes: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    """
    Build corpus for TF-IDF: list of CWE IDs and their unified texts.
    Text = name + description + extended_description
    """
    cwe_ids = []
    texts = []
    for cwe_id, meta in cwes.items():
        parts = [
            meta.get("name", ""),
            meta.get("description", ""),
            meta.get("extended_description", "")
        ]
        text = " ".join(p.strip() for p in parts if isinstance(p, str))
        if text:
            cwe_ids.append(cwe_id)
            texts.append(text)
    return cwe_ids, texts


def preprocess_text(text: str) -> str:
    """Optional: custom preprocessing (e.g., lower, clean, etc.)"""
    return text.strip()


# -------------------------------
# 3. TF-IDF Recommender Engine
# -------------------------------

class TfidfCveCweMapper:
    def __init__(self, cwe_ids: List[str], cwe_texts: List[str]):
        self.cwe_ids = cwe_ids
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2,
            token_pattern=r'(?u)\b\w\w+\b',
            lowercase=True
        )
        # Fit ONLY on CWE corpus → domain-aware IDF
        self.X_cwe = self.vectorizer.fit_transform(cwe_texts)

    def recommend(self,
                   cve_desc: str,
                   top_k: int = 2,
                   threshold: float = 0.05,
                   decay_ratio: float = 0.7) -> List[Dict[str, float]]:
        """
        Recommend top-K CWEs for a CVE description.
        Returns: [{"cwe_id": "123", "score": 0.21}, ...]
        """
        if not cve_desc.strip():
            return []

        try:
            X_cve = self.vectorizer.transform([cve_desc])
        except Exception:
            return []

        sims = cosine_similarity(X_cve, self.X_cwe).flatten()
        top_indices = np.argsort(sims)[::-1]

        results = []
        prev_score = None

        for idx in top_indices:
            score = float(sims[idx])
            if score < threshold:
                break
            cwe_id = self.cwe_ids[idx]

            if len(results) == 0:
                results.append({"cwe_id": cwe_id, "score": score})
                prev_score = score
            elif len(results) < top_k:
                # Stop if score drops too fast (e.g., 0.21 → 0.06)
                if score < prev_score * decay_ratio:
                    break
                results.append({"cwe_id": cwe_id, "score": score})
                prev_score = score
            else:
                break

        return results

# -------------------------------
# 新增：评估 & 统计模块
# -------------------------------
class CweConsistencyEvaluator:
    def __init__(self,logger: logging.Logger):
        self.logger = logger
        self.total = 0
        self.consistent = 0
        self.inconsistent = 0
        self.inconsistent_samples: List[dict] = []
        self.max_warnings_per_year = 30
        self.year_mismatch_count = defaultdict(int)  # e.g., {"CVE-2022": 5}

    def evaluate_and_log(self,
                         cve_id: str,
                         original_cwes: List[str],
                         recommendations: List[dict],
                         year_key:str) -> bool:
        """
        Returns: True if consistent (any overlap), False otherwise
        """
        self.total += 1
        orig_set = set(str(x) for x in original_cwes)  # 确保是 str
        rec_set = {str(r["cwe_id"]) for r in recommendations}

        if orig_set & rec_set:  # 交集非空 → 一致
            self.consistent += 1
            return True
        else:
            self.inconsistent += 1
            self.year_mismatch_count[year_key] += 1

            # 只输出该年前30个不一致样本的详情
            if self.year_mismatch_count[year_key] <= self.max_warnings_per_year:
                self.logger.warning(f"[⚠️ MISMATCH] {cve_id}")
                self.logger.warning(f"    Original CWEs : {sorted(orig_set)}")
                rec_ids = [r['cwe_id'] for r in recommendations]
                rec_scores = [round(r['score'], 3) for r in recommendations]
                self.logger.warning(f"    TF-IDF recs   : {rec_ids} (scores: {rec_scores})")
            elif self.year_mismatch_count[year_key] == self.max_warnings_per_year + 1:
                self.logger.warning(f"    ... (more mismatches in {year_key}, suppressed)")

            return False

    def summary(self) -> str:
        return (
            f"\n📊 Consistency Summary:\n"
            f"  Total CVEs with CWE: {self.total}\n"
            f"  Consistent (≥1 overlap): {self.consistent} ({self.consistent/self.total*100:.1f}%)\n"
            f"  Inconsistent (no overlap): {self.inconsistent} ({self.inconsistent/self.total*100:.1f}%)\n"
        )

#    def save_inconsistent_to_jsonl(self, filepath: str):
#        """保存所有不一致样本到 JSONL，便于人工分析"""
#        os.makedirs(os.path.dirname(filepath), exist_ok=True)
#        with open(filepath, 'w', encoding='utf-8') as f:
#            for sample in self.inconsistent_samples:
#                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
#        self.logger.warning(f"💾 Inconsistent samples saved to: {filepath}")

# -------------------------------
# 4. Output Writer
# -------------------------------

def write_recommendations_to_jsonl(
    output_dir: str,
    year_key: str,
    cve_list: List[Tuple[str, dict]],
    mapper: TfidfCveCweMapper,
    evaluator: CweConsistencyEvaluator
) -> int:
    """
    Write recommendations for one year to JSONL.
    Returns: number of CVEs written (with recs)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{year_key}.jsonl")

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for cve_id, info in cve_list:
            if not info.get("cwes"):  # skip CVE don't have CWE
                continue
            desc = preprocess_text(info.get("description", ""))
            recs = mapper.recommend(desc)

            #evalutor
            evaluator.evaluate_and_log(cve_id, info["cwes"], recs,year_key)

            if recs:
                line = {
                    "cve_id": cve_id,
                    "original_cwes": info["cwes"],
                    "recommendations": recs
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
    return count

# -------------------------------
# 5. 设置日志 
# -------------------------------
def setup_logger(log_file: str, quiet:bool=False)-> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger("CVE_CWE_MAPPER")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers.clear()

    # File handler (always enabled, append mode)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# -------------------------------
# 6. Main Orchestration
# -------------------------------

def main(
    cve_dir: str = "./source/cve",
    cwe_path: str = "./source/cwe_db.json",
    output_dir: str = "./result/existing_cve2cwe",
    top_k: int = 2,
    threshold: float = 0.05,
    year: Optional[str] = None,
    log_file: str = "./result/existing_cve2cwe/out.log",
    quiet: bool = False 
):
    logger=setup_logger(log_file,quiet=quiet)

    # Log run header
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Target year: {year if year else 'ALL'}")
    logger.info(f"📂 CVE dir: {cve_dir}")
    logger.info(f"📤 Output dir: {output_dir}")
    logger.info(f"📄 Log file: {log_file}")
    logger.info(f"🔇 Quiet mode: {'ON' if quiet else 'OFF'}")
    logger.info(f"{'='*70}\n")

    print("[✓] Loading CWE database...")
    cwes = load_cwe_db(cwe_path)

    print("[✓] Building CWE text corpus...")
    cwe_ids, cwe_texts = build_cwe_text_corpus(cwes)

    print("[✓] Initializing TF-IDF mapper...")
    mapper = TfidfCveCweMapper(cwe_ids, cwe_texts)

    print("[✓] Loading CVEs by year...")
    cve_by_year = load_cves_by_year(cve_dir)

    # 确定要处理的年份
    if year:
        target_year_key = f"CVE-{year}"
        if target_year_key not in cve_by_year:
            available = sorted([k.replace("CVE-", "") for k in cve_by_year.keys()])
            raise ValueError(f"Year '{year}' not found. Available years: {available}")
        years_to_process = [(target_year_key, cve_by_year[target_year_key])]
    else:
        years_to_process = sorted(cve_by_year.items())

    #create evaluator
    evaluator = CweConsistencyEvaluator(logger)
    total_written = 0
    print(f"[✓] Processing {len(years_to_process)} years...")

    for year_key, cve_list in years_to_process:
        n = write_recommendations_to_jsonl(
            output_dir=output_dir,
            year_key=year_key,
            cve_list=cve_list,
            mapper=mapper,
            evaluator=evaluator
        )
        print(f"  → {year_key}.jsonl: {n} CVEs recommended")
        total_written += n

    # 打印统计
    logger.info(evaluator.summary())

    print(f"\n✅ Done. {total_written} recommendations saved to {output_dir}/")


# -------------------------------
# 6. CLI Entry Point
# -------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map CVE → CWE using TF-IDF")
    parser.add_argument("--cve_dir", default="./source/cve", help="Directory of CVE JSONs (by year)")
    parser.add_argument("--cwe_path", default="./source/cwe_db.json", help="Path to cwe_db.json")
    parser.add_argument("--output_dir", default="./result/existing_cve2cwe", help="Output directory for JSONLs")
    parser.add_argument("--log_file", default="./result/existing_cve2cwe/out.log", help="Log file path (appended on each run)")
    parser.add_argument("--top_k", type=int, default=2, help="Max CWEs per CVE")
    parser.add_argument("--threshold", type=float, default=0.05, help="Min similarity score")
    parser.add_argument("--year", type=str, default=None, help="Process only a specific year (e.g., '2023')")
    parser.add_argument("--quiet", action="store_false", help="Suppress console output (log only to file)")

    args = parser.parse_args()

    main(
        cve_dir=args.cve_dir,
        cwe_path=args.cwe_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        threshold=args.threshold,
        year=args.year,
        log_file=args.log_file,
        quiet=args.quiet
    )
