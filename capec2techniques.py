#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAPEC → ATT&CK Techniques Mapping via TF-IDF (For CAPECs without existing techniques)
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
# 4. Main Orchestration
# -------------------------------

def main(
    capec_path: str = "./source/capec_db.json",
    attack_path: str = "./source/attack_db.json",
    output_path: str = "./result/capec2techniques.json",
    top_k: int = 2,
    threshold: float = 0.05
):
    print("[✓] Loading CAPEC database...")
    capecs = load_capec_db(capec_path)

    print("[✓] Loading ATT&CK database...")
    attacks = load_attack_db(attack_path)

    print("[✓] Building ATT&CK text corpus...")
    tech_ids, tech_texts = build_attack_text_corpus(attacks)

    print("[✓] Initializing TF-IDF mapper...")
    mapper = TfidfCapecTechMapper(tech_ids, tech_texts)

    print("[✓] Processing CAPECs without existing techniques...")
    results = {}
    count = 0

    for capec_id, capec_info in capecs.items():
        # Skip CAPECs that already have techniques
        if capec_info.get("techniques"):
            continue

        capec_text = get_capec_text(capec_info)
        recs = mapper.recommend(capec_text, top_k=top_k, threshold=threshold)

        if recs:
            results[capec_id] = {
                "recommendations": recs
            }
            count += 1

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. {count} CAPEC→Technique mappings saved to {output_path}")


# -------------------------------
# 5. CLI Entry Point
# -------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map CAPEC → ATT&CK Techniques using TF-IDF (for CAPECs without existing techniques)")
    parser.add_argument("--capec_path", default="./source/capec_db.json", help="Path to capec_db.json")
    parser.add_argument("--attack_path", default="./source/attack_db.json", help="Path to attack_db.json")
    parser.add_argument("--output_path", default="./result/capec2techniques.json", help="Output file path")
    parser.add_argument("--top_k", type=int, default=2, help="Max techniques per CAPEC")
    parser.add_argument("--threshold", type=float, default=0.05, help="Min similarity score")
    args = parser.parse_args()

    main(
        capec_path=args.capec_path,
        attack_path=args.attack_path,
        output_path=args.output_path,
        top_k=args.top_k,
        threshold=args.threshold
    )
