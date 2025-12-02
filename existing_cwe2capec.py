#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CWE → CAPEC Mapping via TF-IDF (For CWEs with existing CAPECs - Evaluation)
"""

import os
import json
import sys
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

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


def load_capec_db(filepath: str) -> Dict[str, dict]:
    """Load CAPEC database from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# -------------------------------
# 2. Text Processing & Corpus Building
# -------------------------------

def build_capec_text_corpus(capecs: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    """
    Build corpus for TF-IDF: list of CAPEC IDs and their unified texts.
    Text = name + description + extended_description
    """
    capec_ids = []
    texts = []
    for capec_id, meta in capecs.items():
        parts = [
            meta.get("name", ""),
            meta.get("description", ""),
            meta.get("extended_description", "")
        ]
        text = " ".join(p.strip() for p in parts if isinstance(p, str))
        if text:
            capec_ids.append(capec_id)
            texts.append(text)
    return capec_ids, texts


def get_cwe_text(cwe_info: dict) -> str:
    """Get unified text from CWE info."""
    parts = [
        cwe_info.get("name", ""),
        cwe_info.get("description", ""),
        cwe_info.get("extended_description", "")
    ]
    return " ".join(p.strip() for p in parts if isinstance(p, str))


# -------------------------------
# 3. TF-IDF Recommender Engine
# -------------------------------

class TfidfCweCapecMapper:
    def __init__(self, capec_ids: List[str], capec_texts: List[str]):
        self.capec_ids = capec_ids
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2,
            token_pattern=r'(?u)\b\w\w+\b',
            lowercase=True
        )
        # Fit ONLY on CAPEC corpus → domain-aware IDF
        self.X_capec = self.vectorizer.fit_transform(capec_texts)

    def recommend(self,
                  cwe_text: str,
                  top_k: int = 3,
                  threshold: float = 0.05,
                  decay_ratio: float = 0.7) -> List[Dict[str, float]]:
        """
        Recommend top-K CAPECs for a CWE text.
        Returns: [{"capec_id": "123", "score": 0.21}, ...]
        """
        if not cwe_text.strip():
            return []

        try:
            X_cwe = self.vectorizer.transform([cwe_text])
        except Exception:
            return []

        sims = cosine_similarity(X_cwe, self.X_capec).flatten()
        top_indices = np.argsort(sims)[::-1]

        results = []
        prev_score = None

        for idx in top_indices:
            score = float(sims[idx])
            if score < threshold:
                break
            capec_id = self.capec_ids[idx]

            if len(results) == 0:
                results.append({"capec_id": capec_id, "score": score})
                prev_score = score
            elif len(results) < top_k:
                # Stop if score drops too fast
                if score < prev_score * decay_ratio:
                    break
                results.append({"capec_id": capec_id, "score": score})
                prev_score = score
            else:
                break

        return results


# -------------------------------
# 4. Evaluation Module
# -------------------------------

class CapecConsistencyEvaluator:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.total = 0
        self.consistent = 0
        self.inconsistent = 0
        self.max_warnings = 30
        self.warning_count = 0

    def evaluate_and_log(self,
                         cwe_id: str,
                         original_capecs: List[str],
                         recommendations: List[dict]) -> bool:
        """
        Returns: True if consistent (any overlap), False otherwise
        """
        self.total += 1
        orig_set = set(str(x) for x in original_capecs)
        rec_set = {str(r["capec_id"]) for r in recommendations}

        if orig_set & rec_set:  # Intersection non-empty → consistent
            self.consistent += 1
            return True
        else:
            self.inconsistent += 1
            self.warning_count += 1

            # Only output first 30 mismatch details
            if self.warning_count <= self.max_warnings:
                self.logger.warning(f"[⚠️ MISMATCH] CWE-{cwe_id}")
                self.logger.warning(f"    Original CAPECs : {sorted(orig_set)}")
                rec_ids = [r['capec_id'] for r in recommendations]
                rec_scores = [round(r['score'], 3) for r in recommendations]
                self.logger.warning(f"    TF-IDF recs     : {rec_ids} (scores: {rec_scores})")
            elif self.warning_count == self.max_warnings + 1:
                self.logger.warning("    ... (more mismatches, suppressed)")

            return False

    def summary(self) -> str:
        if self.total == 0:
            return "\n📊 Consistency Summary:\n  No CWEs with existing CAPECs found.\n"
        return (
            f"\n📊 Consistency Summary:\n"
            f"  Total CWEs with CAPEC: {self.total}\n"
            f"  Consistent (≥1 overlap): {self.consistent} ({self.consistent/self.total*100:.1f}%)\n"
            f"  Inconsistent (no overlap): {self.inconsistent} ({self.inconsistent/self.total*100:.1f}%)\n"
        )


# -------------------------------
# 5. Logger Setup
# -------------------------------

def setup_logger(log_file: str, quiet: bool = False) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("CWE_CAPEC_MAPPER")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# -------------------------------
# 6. Main Orchestration
# -------------------------------

def main(
    cwe_path: str = "./source/cwe_db.json",
    capec_path: str = "./source/capec_db.json",
    output_path: str = "./result/existing_cwe2capec.json",
    log_file: str = "./result/existing_cwe2capec.log",
    top_k: int = 3,
    threshold: float = 0.05,
    quiet: bool = False
):
    logger = setup_logger(log_file, quiet=quiet)

    # Log run header
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📂 CWE path: {cwe_path}")
    logger.info(f"📂 CAPEC path: {capec_path}")
    logger.info(f"📤 Output path: {output_path}")
    logger.info(f"📄 Log file: {log_file}")
    logger.info(f"🔇 Quiet mode: {'ON' if quiet else 'OFF'}")
    logger.info(f"{'='*70}\n")

    print("[✓] Loading CWE database...")
    cwes = load_cwe_db(cwe_path)

    print("[✓] Loading CAPEC database...")
    capecs = load_capec_db(capec_path)

    print("[✓] Building CAPEC text corpus...")
    capec_ids, capec_texts = build_capec_text_corpus(capecs)

    print("[✓] Initializing TF-IDF mapper...")
    mapper = TfidfCweCapecMapper(capec_ids, capec_texts)

    print("[✓] Processing CWEs with existing CAPECs...")
    evaluator = CapecConsistencyEvaluator(logger)
    results = {}
    count = 0

    for cwe_id, cwe_info in cwes.items():
        # Only process CWEs that have existing CAPECs
        if not cwe_info.get("capecs"):
            continue

        cwe_text = get_cwe_text(cwe_info)
        recs = mapper.recommend(cwe_text, top_k=top_k, threshold=threshold)

        # Evaluate consistency
        evaluator.evaluate_and_log(cwe_id, cwe_info["capecs"], recs)

        if recs:
            results[cwe_id] = {
                "original_capecs": cwe_info["capecs"],
                "recommendations": recs
            }
            count += 1

    # Log summary
    logger.info(evaluator.summary())

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. {count} CWE→CAPEC mappings saved to {output_path}")


# -------------------------------
# 7. CLI Entry Point
# -------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map CWE → CAPEC using TF-IDF (for CWEs with existing CAPECs - Evaluation)")
    parser.add_argument("--cwe_path", default="./source/cwe_db.json", help="Path to cwe_db.json")
    parser.add_argument("--capec_path", default="./source/capec_db.json", help="Path to capec_db.json")
    parser.add_argument("--output_path", default="./result/existing_cwe2capec.json", help="Output file path")
    parser.add_argument("--log_file", default="./result/existing_cwe2capec.log", help="Log file path")
    parser.add_argument("--top_k", type=int, default=3, help="Max CAPECs per CWE")
    parser.add_argument("--threshold", type=float, default=0.05, help="Min similarity score")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output (log only to file)")
    args = parser.parse_args()

    main(
        cwe_path=args.cwe_path,
        capec_path=args.capec_path,
        output_path=args.output_path,
        log_file=args.log_file,
        top_k=args.top_k,
        threshold=args.threshold,
        quiet=args.quiet
    )
