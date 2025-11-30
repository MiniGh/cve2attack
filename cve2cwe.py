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
# 4. Output Writer
# -------------------------------

def write_recommendations_to_jsonl(
    output_dir: str,
    year_key: str,
    cve_list: List[Tuple[str, dict]],
    mapper: TfidfCveCweMapper
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
            if info.get("cwes"):  # skip if already has CWE
                continue
            desc = preprocess_text(info.get("description", ""))
            recs = mapper.recommend(desc)
            if recs:
                line = {
                    "cve_id": cve_id,
                    "recommendations": recs
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
    return count


# -------------------------------
# 5. Main Orchestration
# -------------------------------

def main(
    cve_dir: str = "./source/cve",
    cwe_path: str = "./source/cwe_db.json",
    output_dir: str = "./result/cve2cwe",
    top_k: int = 2,
    threshold: float = 0.05
):
    print("[✓] Loading CWE database...")
    cwes = load_cwe_db(cwe_path)

    print("[✓] Building CWE text corpus...")
    cwe_ids, cwe_texts = build_cwe_text_corpus(cwes)

    print("[✓] Initializing TF-IDF mapper...")
    mapper = TfidfCveCweMapper(cwe_ids, cwe_texts)

    print("[✓] Loading CVEs by year...")
    cve_by_year = load_cves_by_year(cve_dir)

    print(f"[✓] Processing {len(cve_by_year)} years...")
    total_written = 0
    for year_key, cve_list in sorted(cve_by_year.items()):
        n = write_recommendations_to_jsonl(
            output_dir=output_dir,
            year_key=year_key,
            cve_list=cve_list,
            mapper=mapper
        )
        print(f"  → {year_key}.jsonl: {n} CVEs recommended")
        total_written += n

    print(f"\n✅ Done. {total_written} recommendations saved to {output_dir}/")


# -------------------------------
# 6. CLI Entry Point
# -------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map CVE → CWE using TF-IDF")
    parser.add_argument("--cve_dir", default="./source/cve", help="Directory of CVE JSONs (by year)")
    parser.add_argument("--cwe_path", default="./source/cwe_db.json", help="Path to cwe_db.json")
    parser.add_argument("--output_dir", default="./result/cve2cwe", help="Output directory for JSONLs")
    parser.add_argument("--top_k", type=int, default=2, help="Max CWEs per CVE")
    parser.add_argument("--threshold", type=float, default=0.05, help="Min similarity score")
    args = parser.parse_args()

    main(
        cve_dir=args.cve_dir,
        cwe_path=args.cwe_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        threshold=args.threshold
    )
