#!/usr/bin/env python3
"""
V4: Structured chain fusion — CWE→CAPEC→ATT&CK + embedding retrieval.

Fuses retrieval candidates (V3, LLM-rewritten) with structured chain
mappings to produce a combined top-20 candidate set per CVE.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ─────────────────────────────────────────────────────────────
#  Configurable parameters
# ─────────────────────────────────────────────────────────────
ALPHA = 0.3             # chain score multiplier
FANOUT_THRESHOLD = 10   # ≤10: add new candidates; >10: boost only
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAIN_FILE = PROJECT_ROOT / "Validate_data" / "cve2technique_full.jsonl"
CWE_XML = PROJECT_ROOT / "og_data" / "cwe.xml"
V3_DIR = PROJECT_ROOT / "output" / "retrieval" / "llm_rewritten_proc"
OUTPUT_DIR = PROJECT_ROOT / "output" / "retrieval" / "fused"


def load_cwe_abstraction(xml_path: Path) -> Dict[str, str]:
    """Parse CWE XML and return {cwe_id: abstraction} for Base/Variant only."""
    ns = {"cwe": "http://cwe.mitre.org/cwe-7"}
    tree = ET.parse(str(xml_path))

    allowed = {"Base", "Variant"}
    result: Dict[str, str] = {}
    for w in tree.getroot().iterfind("cwe:Weaknesses/cwe:Weakness", ns):
        cid = w.get("ID", "").strip()
        ab = w.get("Abstraction", "").strip()
        if cid and ab in allowed:
            result[cid] = ab
    return result


def parent_tech(tech_id: str) -> str:
    """Roll up sub-technique (T1027.006) to parent (T1027)."""
    dot = tech_id.find(".")
    return tech_id[:dot] if dot != -1 else tech_id


def load_chain_data(chain_path: Path) -> Dict[str, dict]:
    """Load cve2technique_full.jsonl as {cve_id: {cwes, capecs, techniques}}."""
    data: Dict[str, dict] = {}
    with chain_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            data[entry["cve_id"]] = {
                "cwes": entry.get("cwes", []),
                "capecs": entry.get("capecs", []),
                "techniques": entry.get("techniques", []),
            }
    return data


def load_v3_candidates(v3_dir: Path) -> Dict[str, List[Tuple[str, float]]]:
    """
    Load V3 retrieval results from CVE-*.jsonl files.
    Returns {cve_id: [(tech_id, score), ...]} sorted by score desc.
    Output format: {"cve_id": "...", "techniques": [{"id": "T1204", "score": 0.81}, ...]}
    """
    v3: Dict[str, List[Tuple[str, float]]] = {}
    for fpath in sorted(v3_dir.glob("CVE-*.jsonl")):
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                cve_id = entry["cve_id"]
                techs = entry.get("techniques", [])
                # Convert [{"id": ..., "score": ...}] to [(id, score)]
                v3[cve_id] = [(t["id"], t["score"]) for t in techs]
    return v3


def build_global_capec_tech_map(chain_data: Dict[str, dict]) -> Dict[str, int]:
    """
    Build global mapping: CAPEC ID -> number of unique parent techniques.
    Used for fanout_CAPEC computation.
    """
    capec_techs: Dict[str, Set[str]] = defaultdict(set)
    for entry in chain_data.values():
        techs = entry["techniques"]
        for capec in entry["capecs"]:
            for t in techs:
                pt = parent_tech(t)
                if pt:
                    capec_techs[capec].add("T" + pt)
    return {c: len(ts) for c, ts in capec_techs.items()}


def compute_chain_contributions(
    cve_id: str,
    chain_data: Dict[str, dict],
    cwe_ab: Dict[str, str],
    capec_tech_count: Dict[str, int],
) -> Tuple[Dict[str, float], int]:
    """
    For a given CVE, compute chain-derived technique scores.

    Returns:
        (chain_scores, fanout) where chain_scores maps tech_id -> chain_score
        and fanout is the total unique parent technique count.
    """
    chain = chain_data.get(cve_id)
    if not chain:
        return {}, 0

    # Step 1: Filter CWEs to Base/Variant only
    valid_cwes = [c for c in chain["cwes"] if c in cwe_ab]
    if not valid_cwes:
        return {}, 0

    # Step 2: Collect all techniques from valid CWEs, roll-up, dedup, add T-prefix
    all_techs: Set[str] = set()
    for t in chain["techniques"]:
        pt = parent_tech(t)
        if pt:
            all_techs.add("T" + pt)  # chain data uses no T-prefix; add it

    fanout = len(all_techs)

    # Step 3: Compute per-technique chain_score
    # fanout_CWE = number of CAPECs per CWE
    fanout_cwe = len(chain["capecs"]) if chain["capecs"] else 1

    chain_scores: Dict[str, float] = {}
    for tech in all_techs:
        # fanout_CAPEC: average over all CAPECs for this CVE
        avg_capec_fanout = 1.0
        if chain["capecs"]:
            capec_fanouts = [capec_tech_count.get(c, 1) for c in chain["capecs"]]
            avg_capec_fanout = sum(capec_fanouts) / len(capec_fanouts)
        chain_scores[tech] = 1.0 / (fanout_cwe * avg_capec_fanout)

    return chain_scores, fanout


def fuse_one_cve(
    cve_id: str,
    v3_candidates: List[Tuple[str, float]],
    chain_scores: Dict[str, float],
    fanout: int,
) -> List[Tuple[str, float]]:
    """Fuse V3 retrieval scores with chain scores for one CVE."""
    # Build lookup: tech_id -> retrieval score
    v3_map: Dict[str, float] = {t: s for t, s in v3_candidates}

    # β = half of the lowest retrieval score in V3 top-20
    if v3_candidates:
        lowest_v3_score = v3_candidates[-1][1]
        beta = lowest_v3_score / 2.0
    else:
        beta = 0.0

    # Build final scores
    final: Dict[str, float] = {}

    # All techniques: union of V3 and chain
    all_techs = set(v3_map.keys()) | set(chain_scores.keys())

    for tech in all_techs:
        in_v3 = tech in v3_map
        in_chain = tech in chain_scores

        if in_v3 and in_chain:
            # Case 1: intersection → retrieval + α × chain
            final[tech] = v3_map[tech] + ALPHA * chain_scores[tech]
        elif in_v3:
            # Case 2: V3 only → keep retrieval score
            final[tech] = v3_map[tech]
        else:
            # Case 3: chain only → β + α × chain (only if low fanout)
            if fanout <= FANOUT_THRESHOLD:
                final[tech] = beta + ALPHA * chain_scores[tech]
            # else: high fanout, do NOT add

    # Sort by score desc, take top-20
    sorted_candidates = sorted(final.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates[:20]


def main() -> None:
    print("[INFO] Loading CWE abstraction filter ...")
    cwe_ab = load_cwe_abstraction(CWE_XML)
    print(f"[INFO] Base+Variant CWEs retained: {len(cwe_ab)}")

    print("[INFO] Loading chain data ...")
    chain_data = load_chain_data(CHAIN_FILE)
    print(f"[INFO] Chain entries: {len(chain_data)}")

    print("[INFO] Building global CAPEC→technique map ...")
    capec_tech_count = build_global_capec_tech_map(chain_data)
    print(f"[INFO] Unique CAPECs with technique counts: {len(capec_tech_count)}")

    print("[INFO] Loading V3 candidates ...")
    v3_data = load_v3_candidates(V3_DIR)
    print(f"[INFO] V3 CVE entries: {len(v3_data)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group by year for output
    years: Dict[str, List[dict]] = defaultdict(list)
    total_cves = 0
    total_chain_hits = 0

    for cve_id, v3_candidates in sorted(v3_data.items()):
        chain_scores, fanout = compute_chain_contributions(
            cve_id, chain_data, cwe_ab, capec_tech_count
        )

        if chain_scores:
            total_chain_hits += 1

        fused = fuse_one_cve(cve_id, v3_candidates, chain_scores, fanout)

        year = cve_id.split("-")[1]
        years[year].append({
            "cve_id": cve_id,
            "techniques": [t[0] for t in fused],
        })
        total_cves += 1

    # Write per-year output
    for year in sorted(years):
        fpath = OUTPUT_DIR / f"CVE-{year}.jsonl"
        with fpath.open("w", encoding="utf-8") as f:
            for entry in sorted(years[year], key=lambda x: x["cve_id"]):
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[INFO] Wrote {len(years[year])} CVEs to {fpath.name}")

    print(f"[INFO] Total CVEs processed: {total_cves}")
    print(f"[INFO] CVEs with chain data: {total_chain_hits}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
