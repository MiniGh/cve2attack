#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAPEC → ATT&CK Techniques Mapping via TF-IDF (For CAPECs with existing techniques - Evaluation)
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

def load_capec_db(filepath: str) -> Dict[str, dict]:
    """Load CAPEC database from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_attack_db(filepath: str) -> Dict[str, dict]:
    """Load ATT&CK database from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# -------------------------------
# 2. Text Processing & Corpus Building
# -------------------------------

def build_attack_text_corpus(attacks: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    """
    Build corpus for TF-IDF: list of technique IDs and their unified texts.
    Text = name + description
    """
    tech_ids = []
    texts = []
    for tech_id, meta in attacks.items():
        parts = [
            meta.get("name", ""),
            meta.get("description", "")
        ]
        text = " ".join(p.strip() for p in parts if isinstance(p, str))
        if text:
            tech_ids.append(tech_id)
            texts.append(text)
    return tech_ids, texts


def get_capec_text(capec_info: dict) -> str:
    """Get unified text from CAPEC info."""
    parts = [
        capec_info.get("name", ""),
        capec_info.get("description", ""),
        capec_info.get("extended_description", "")
    ]
    return " ".join(p.strip() for p in parts if isinstance(p, str))


# -------------------------------
# 3. TF-IDF Recommender Engine
# -------------------------------

class TfidfCapecTechMapper:
    def __init__(self, tech_ids: List[str], tech_texts: List[str]):
        self.tech_ids = tech_ids
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2,
            token_pattern=r'(?u)\b\w\w+\b',
            lowercase=True
        )
        # Fit ONLY on ATT&CK corpus → domain-aware IDF
        self.X_tech = self.vectorizer.fit_transform(tech_texts)

    def recommend(self,
                  capec_text: str,
                  top_k: int = 2,
                  threshold: float = 0.05,
                  decay_ratio: float = 0.7) -> List[Dict[str, float]]:
        """
        Recommend top-K techniques for a CAPEC text.
        Returns: [{"technique_id": "T1234", "score": 0.21}, ...]
        """
        if not capec_text.strip():
            return []

        try:
            X_capec = self.vectorizer.transform([capec_text])
        except Exception:
            return []

        sims = cosine_similarity(X_capec, self.X_tech).flatten()
        top_indices = np.argsort(sims)[::-1]

        results = []
        prev_score = None

        for idx in top_indices:
            score = float(sims[idx])
            if score < threshold:
                break
            tech_id = self.tech_ids[idx]

            if len(results) == 0:
                results.append({"technique_id": tech_id, "score": score})
                prev_score = score
            elif len(results) < top_k:
                # Stop if score drops too fast
                if score < prev_score * decay_ratio:
                    break
                results.append({"technique_id": tech_id, "score": score})
                prev_score = score
            else:
                break

        return results


# -------------------------------
# 4. Evaluation Module
# -------------------------------

class TechniqueConsistencyEvaluator:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.total = 0
        self.consistent = 0
        self.inconsistent = 0
        self.max_warnings = 30
        self.warning_count = 0

    @staticmethod
    def normalize_tech_id(tech_id: str) -> str:
        """Normalize technique ID by removing 'T' prefix if present."""
        tech_str = str(tech_id)
        if tech_str.startswith('T'):
            return tech_str[1:]
        return tech_str

    def evaluate_and_log(self,
                         capec_id: str,
                         original_techs: List[str],
                         recommendations: List[dict]) -> bool:
        """
        Returns: True if consistent (any overlap), False otherwise
        """
        self.total += 1
        # Normalize technique IDs for comparison (remove 'T' prefix)
        orig_set = set(self.normalize_tech_id(x) for x in original_techs)
        rec_set = {self.normalize_tech_id(r["technique_id"]) for r in recommendations}

        if orig_set & rec_set:  # Intersection non-empty → consistent
            self.consistent += 1
            return True
        else:
            self.inconsistent += 1
            self.warning_count += 1

            # Only output first 30 mismatch details
            if self.warning_count <= self.max_warnings:
                self.logger.warning(f"[⚠️ MISMATCH] CAPEC-{capec_id}")
                self.logger.warning(f"    Original Techniques : {sorted(orig_set)}")
                rec_ids = [r['technique_id'] for r in recommendations]
                rec_scores = [round(r['score'], 3) for r in recommendations]
                self.logger.warning(f"    TF-IDF recs         : {rec_ids} (scores: {rec_scores})")
            elif self.warning_count == self.max_warnings + 1:
                self.logger.warning("    ... (more mismatches, suppressed)")

            return False

    def summary(self) -> str:
        if self.total == 0:
            return "\n📊 Consistency Summary:\n  No CAPECs with existing techniques found.\n"
        return (
            f"\n📊 Consistency Summary:\n"
            f"  Total CAPECs with Techniques: {self.total}\n"
            f"  Consistent (≥1 overlap): {self.consistent} ({self.consistent/self.total*100:.1f}%)\n"
            f"  Inconsistent (no overlap): {self.inconsistent} ({self.inconsistent/self.total*100:.1f}%)\n"
        )


# -------------------------------
# 5. Logger Setup
# -------------------------------

def setup_logger(log_file: str, quiet: bool = False) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("CAPEC_TECH_MAPPER")
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
    capec_path: str = "./source/capec_db.json",
    attack_path: str = "./source/attack_db.json",
    output_path: str = "./result/existing_capec2techniques.json",
    log_file: str = "./result/existing_capec2techniques.log",
    top_k: int = 2,
    threshold: float = 0.05,
    quiet: bool = False
):
    logger = setup_logger(log_file, quiet=quiet)

    # Log run header
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📂 CAPEC path: {capec_path}")
    logger.info(f"📂 ATT&CK path: {attack_path}")
    logger.info(f"📤 Output path: {output_path}")
    logger.info(f"📄 Log file: {log_file}")
    logger.info(f"🔇 Quiet mode: {'ON' if quiet else 'OFF'}")
    logger.info(f"{'='*70}\n")

    print("[✓] Loading CAPEC database...")
    capecs = load_capec_db(capec_path)

    print("[✓] Loading ATT&CK database...")
    attacks = load_attack_db(attack_path)

    print("[✓] Building ATT&CK text corpus...")
    tech_ids, tech_texts = build_attack_text_corpus(attacks)

    print("[✓] Initializing TF-IDF mapper...")
    mapper = TfidfCapecTechMapper(tech_ids, tech_texts)

    print("[✓] Processing CAPECs with existing techniques...")
    evaluator = TechniqueConsistencyEvaluator(logger)
    results = {}
    count = 0

    for capec_id, capec_info in capecs.items():
        # Only process CAPECs that have existing techniques
        if not capec_info.get("techniques"):
            continue

        capec_text = get_capec_text(capec_info)
        recs = mapper.recommend(capec_text, top_k=top_k, threshold=threshold)

        # Evaluate consistency
        evaluator.evaluate_and_log(capec_id, capec_info["techniques"], recs)

        if recs:
            results[capec_id] = {
                "original_techniques": capec_info["techniques"],
                "recommendations": recs
            }
            count += 1

    # Log summary
    logger.info(evaluator.summary())

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. {count} CAPEC→Technique mappings saved to {output_path}")


# -------------------------------
# 7. CLI Entry Point
# -------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map CAPEC → ATT&CK Techniques using TF-IDF (for CAPECs with existing techniques - Evaluation)")
    parser.add_argument("--capec_path", default="./source/capec_db.json", help="Path to capec_db.json")
    parser.add_argument("--attack_path", default="./source/attack_db.json", help="Path to attack_db.json")
    parser.add_argument("--output_path", default="./result/existing_capec2techniques.json", help="Output file path")
    parser.add_argument("--log_file", default="./result/existing_capec2techniques.log", help="Log file path")
    parser.add_argument("--top_k", type=int, default=2, help="Max techniques per CAPEC")
    parser.add_argument("--threshold", type=float, default=0.05, help="Min similarity score")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output (log only to file)")
    args = parser.parse_args()

    main(
        capec_path=args.capec_path,
        attack_path=args.attack_path,
        output_path=args.output_path,
        log_file=args.log_file,
        top_k=args.top_k,
        threshold=args.threshold,
        quiet=args.quiet
    )
