#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CWE → CAPEC Mapping via TF-IDF (For CWEs without existing CAPECs)
"""

import os
import json
from typing import Dict, List, Tuple

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
# 4. Main Orchestration
# -------------------------------

def main(
    cwe_path: str = "./source/cwe_db.json",
    capec_path: str = "./source/capec_db.json",
    output_path: str = "./result/cwe2capec.json",
    top_k: int = 3,
    threshold: float = 0.05
):
    print("[✓] Loading CWE database...")
    cwes = load_cwe_db(cwe_path)

    print("[✓] Loading CAPEC database...")
    capecs = load_capec_db(capec_path)

    print("[✓] Building CAPEC text corpus...")
    capec_ids, capec_texts = build_capec_text_corpus(capecs)

    print("[✓] Initializing TF-IDF mapper...")
    mapper = TfidfCweCapecMapper(capec_ids, capec_texts)

    print("[✓] Processing CWEs without existing CAPECs...")
    results = {}
    count = 0

    for cwe_id, cwe_info in cwes.items():
        # Skip CWEs that already have CAPECs
        if cwe_info.get("capecs"):
            continue

        cwe_text = get_cwe_text(cwe_info)
        recs = mapper.recommend(cwe_text, top_k=top_k, threshold=threshold)

        if recs:
            results[cwe_id] = {
                "recommendations": recs
            }
            count += 1

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. {count} CWE→CAPEC mappings saved to {output_path}")


# -------------------------------
# 5. CLI Entry Point
# -------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map CWE → CAPEC using TF-IDF (for CWEs without existing CAPECs)")
    parser.add_argument("--cwe_path", default="./source/cwe_db.json", help="Path to cwe_db.json")
    parser.add_argument("--capec_path", default="./source/capec_db.json", help="Path to capec_db.json")
    parser.add_argument("--output_path", default="./result/cwe2capec.json", help="Output file path")
    parser.add_argument("--top_k", type=int, default=3, help="Max CAPECs per CWE")
    parser.add_argument("--threshold", type=float, default=0.05, help="Min similarity score")
    args = parser.parse_args()

    main(
        cwe_path=args.cwe_path,
        capec_path=args.capec_path,
        output_path=args.output_path,
        top_k=args.top_k,
        threshold=args.threshold
    )
