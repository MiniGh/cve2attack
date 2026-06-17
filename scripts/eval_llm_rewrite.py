#!/usr/bin/env python3
"""Compare 3 retrieval methods on the CVE2ATT&CK evaluation set.

Methods:
  V1 baseline:  raw description -> technique name+desc
  V2 +procedures: raw description -> technique name+desc+procedures (git commit e295530)
  V3 +LLM rewrite: rewritten text -> technique name+desc

Ground truth: cve2attack_result/CVE-*.jsonl (1661 CVEs, 2008–2022).

Output: output/retrieval/eval_comparison.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set


# ---------------------------------------------------------------------------
# Paths (relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CVE2ATTACK_DIR = PROJECT_ROOT / "cve2attack_result"
V1_DEFAULT_DIR = PROJECT_ROOT / "output" / "retrieval"
V3_DEFAULT_DIR = PROJECT_ROOT / "output" / "retrieval" / "llm_rewritten"
V4_DEFAULT_DIR = PROJECT_ROOT / "output" / "retrieval" / "fused"
OUTPUT_MD = PROJECT_ROOT / "output" / "retrieval" / "eval_comparison.md"

# Git commit holding the V2 (+procedures) retrieval results
V2_GIT_COMMIT = "e295530"
V2_GIT_PATH = "output/retrieval"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Metrics:
    cves: int
    truth_techniques: int
    predicted_techniques: int
    recall_at_10: float
    recall_at_20: float


@dataclass
class MethodSummary:
    name: str
    dir_path: Path
    cves: int
    pred: Dict[str, List[str]]
    metrics: Optional[Metrics] = None


# ---------------------------------------------------------------------------
# Core logic (reused from evaluate_embedding_recall_by_year.py)
# ---------------------------------------------------------------------------
def load_jsonl_directory(directory: Path) -> Dict[str, List[str]]:
    """Load yearly JSONL files into a CVE -> techniques mapping."""
    per_cve: Dict[str, List[str]] = {}
    if not directory.exists():
        return per_cve
    for path in sorted(directory.glob("CVE-*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                cve_id = str(record["cve_id"])
                raw_techs = record.get("techniques", [])
                techniques: List[str] = []
                for t in raw_techs:
                    if isinstance(t, dict):
                        tid = str(t.get("id", "")).strip()
                    else:
                        tid = str(t).strip()
                    if tid:
                        techniques.append(tid)
                per_cve[cve_id] = techniques
    return per_cve


def load_truth(directory: Path) -> Dict[str, Set[str]]:
    """Load ground-truth techniques.  Each line: {"cve_id": "...", "techniques": [...]}."""
    truth: Dict[str, Set[str]] = {}
    if not directory.exists():
        return truth
    for path in sorted(directory.glob("CVE-*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                cve_id = str(record["cve_id"])
                raw_techs = record.get("techniques", [])
                techniques = set()
                for t in raw_techs:
                    if isinstance(t, dict):
                        tid = str(t.get("id", "")).strip()
                    else:
                        tid = str(t).strip()
                    if tid:
                        techniques.add(tid)
                if techniques:
                    truth[cve_id] = techniques
    return truth


def compute_metrics(
    pred: Mapping[str, List[str]], truth: Mapping[str, Set[str]]
) -> Metrics:
    """Compute recall@10 and recall@20 against the ground truth."""
    # Only evaluate CVEs that appear in both prediction and truth
    common_cves = set(pred) & set(truth)
    if not common_cves:
        return Metrics(
            cves=0,
            truth_techniques=0,
            predicted_techniques=0,
            recall_at_10=0.0,
            recall_at_20=0.0,
        )

    total_truth_techniques = sum(len(truth[c]) for c in common_cves)
    total_predicted_techniques = sum(len(pred[c]) for c in common_cves)
    recall_at_10_sum = 0.0
    recall_at_20_sum = 0.0

    for cve_id in common_cves:
        truth_set = truth[cve_id]
        predicted = pred[cve_id]
        top10 = predicted[:10]
        top20 = predicted[:20]

        truth_size = len(truth_set)
        if truth_size == 0:
            continue

        top10_set = set(top10)
        top20_set = set(top20)
        top10_hits = len(truth_set & top10_set)
        top20_hits = len(truth_set & top20_set)

        recall_at_10_sum += top10_hits / truth_size
        recall_at_20_sum += top20_hits / truth_size

    denom = float(len(common_cves))
    return Metrics(
        cves=len(common_cves),
        truth_techniques=total_truth_techniques,
        predicted_techniques=total_predicted_techniques,
        recall_at_10=recall_at_10_sum / denom,
        recall_at_20=recall_at_20_sum / denom,
    )


# ---------------------------------------------------------------------------
# V2 extraction from git
# ---------------------------------------------------------------------------
def extract_v2_from_git(output_dir: Path) -> bool:
    """Extract V2 retrieval files from git commit e295530 (output/retrieval/CVE-*.jsonl)."""
    try:
        # List matching files in the commit
        result = subprocess.run(
            [
                "git", "-C", str(PROJECT_ROOT),
                "ls-tree", "-r", "--name-only", V2_GIT_COMMIT, V2_GIT_PATH,
            ],
            capture_output=True, text=True, check=True,
        )
        files = [
            f for f in result.stdout.strip().splitlines()
            if f.startswith(f"{V2_GIT_PATH}/CVE-") and f.endswith(".jsonl")
        ]
        if not files:
            print(f"No CVE-*.jsonl files found in git commit {V2_GIT_COMMIT}:{V2_GIT_PATH}/")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)
        for git_path in files:
            fname = Path(git_path).name  # e.g. CVE-2024.jsonl
            try:
                content = subprocess.run(
                    ["git", "-C", str(PROJECT_ROOT), "show", f"{V2_GIT_COMMIT}:{git_path}"],
                    capture_output=True, text=True, check=True,
                ).stdout
                (output_dir / fname).write_text(content, encoding="utf-8")
            except subprocess.CalledProcessError as exc:
                print(f"Warning: could not extract {git_path}: {exc}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error accessing git: {exc}")
        return False


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_comparison_table(methods: List[MethodSummary]) -> None:
    """Write a markdown comparison table to OUTPUT_MD."""
    lines: List[str] = [
        "# Retrieval Method Comparison on CVE2ATT&CK Evaluation Set",
        "",
        "Ground truth: `cve2attack_result/` (1661 CVEs across 2008-2022).",
        "",
        "| Method | CVEs | Recall@10 | Recall@20 |",
        "|---|---|---|---|",
    ]

    for m in methods:
        if m.metrics is None:
            lines.append(f"| {m.name} | {m.cves} | N/A | N/A |")
        else:
            lines.append(
                f"| {m.name} | {m.metrics.cves} | "
                f"{format_pct(m.metrics.recall_at_10)} | "
                f"{format_pct(m.metrics.recall_at_20)} |"
            )

    lines.extend([
        "",
        "> **Recall@K**: average per-CVE technique recall, i.e. `|pred &cap; truth| / |truth|` averaged over all evaluated CVEs.",
        "",
        f"> V2 extraction from git commit `{V2_GIT_COMMIT}`.",
    ])

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 4 retrieval methods on CVE2ATT&CK eval set.",
    )
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=V1_DEFAULT_DIR,
        help=f"Path to V1 retrieval output directory (default: {V1_DEFAULT_DIR})",
    )
    parser.add_argument(
        "--v2-dir",
        type=Path,
        default=None,
        help="Path to V2 retrieval output directory. If not given, auto-extract from git commit e295530.",
    )
    parser.add_argument(
        "--v3-dir",
        type=Path,
        default=V3_DEFAULT_DIR,
        help=f"Path to V3 retrieval output directory (default: {V3_DEFAULT_DIR})",
    )
    parser.add_argument(
        "--v4-dir",
        type=Path,
        default=V4_DEFAULT_DIR,
        help=f"Path to V4 retrieval output directory (default: {V4_DEFAULT_DIR})",
    )
    parser.add_argument(
        "--no-v2-extract",
        action="store_true",
        help="Skip auto-extraction of V2 from git even if --v2-dir is not given.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Load ground truth
    print("Loading ground truth from cve2attack_result/ ...")
    truth = load_truth(CVE2ATTACK_DIR)
    if not truth:
        print("ERROR: no ground truth found in", CVE2ATTACK_DIR, file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(truth)} CVEs with ground-truth techniques.")

    # --- V1 ---
    v1_name = "V1: baseline (raw desc)"
    v1_pred = load_jsonl_directory(args.v1_dir)
    v1_metrics = compute_metrics(v1_pred, truth)
    print(f"\n{v1_name}: {v1_metrics.cves} CVEs, "
          f"R@10={v1_metrics.recall_at_10:.4f}, R@20={v1_metrics.recall_at_20:.4f}")

    # --- V2 ---
    v2_dir: Optional[Path]
    v2_temp_dir: Optional[str] = None
    if args.v2_dir is not None:
        v2_dir = args.v2_dir
    elif args.no_v2_extract:
        print("\nV2: skipped (--no-v2-extract set and no --v2-dir given)")
        v2_dir = None
    else:
        print("\nExtracting V2 from git commit e295530 ...")
        v2_temp_dir = tempfile.mkdtemp(prefix="v2_retrieval_")
        v2_dir = Path(v2_temp_dir)
        ok = extract_v2_from_git(v2_dir)
        if not ok:
            print("V2 extraction failed; skipping V2.")
            v2_dir = None

    v2_metrics: Optional[Metrics] = None
    v2_pred: Dict[str, List[str]] = {}
    if v2_dir is not None and v2_dir.exists():
        v2_name = "V2: +procedures"
        v2_pred = load_jsonl_directory(v2_dir)
        v2_metrics = compute_metrics(v2_pred, truth)
        print(f"{v2_name}: {v2_metrics.cves} CVEs, "
              f"R@10={v2_metrics.recall_at_10:.4f}, R@20={v2_metrics.recall_at_20:.4f}")

    # --- V3 ---
    v3_name = "V3: +LLM rewrite"
    v3_pred = load_jsonl_directory(args.v3_dir)
    v3_metrics = compute_metrics(v3_pred, truth)
    print(f"\n{v3_name}: {v3_metrics.cves} CVEs, "
          f"R@10={v3_metrics.recall_at_10:.4f}, R@20={v3_metrics.recall_at_20:.4f}")

    # --- V4 ---
    v4_name = "V4: +structured chain"
    v4_pred = load_jsonl_directory(args.v4_dir)
    v4_metrics = compute_metrics(v4_pred, truth)
    print(f"\n{v4_name}: {v4_metrics.cves} CVEs, "
          f"R@10={v4_metrics.recall_at_10:.4f}, R@20={v4_metrics.recall_at_20:.4f}")

    # --- Report ---
    methods: List[MethodSummary] = [
        MethodSummary(name=v1_name, dir_path=args.v1_dir, cves=v1_metrics.cves, pred=v1_pred, metrics=v1_metrics),
    ]
    if v2_metrics is not None and v2_dir is not None:
        methods.append(
            MethodSummary(name=v2_name, dir_path=v2_dir, cves=v2_metrics.cves, pred=v2_pred, metrics=v2_metrics)
        )
    else:
        methods.append(
            MethodSummary(name="V2: +procedures", dir_path=Path("(git e295530)"), cves=0, pred={}, metrics=None)
        )
    methods.append(
        MethodSummary(name=v3_name, dir_path=args.v3_dir, cves=v3_metrics.cves, pred=v3_pred, metrics=v3_metrics)
    )
    methods.append(
        MethodSummary(name=v4_name, dir_path=args.v4_dir, cves=v4_metrics.cves, pred=v4_pred, metrics=v4_metrics)
    )

    write_comparison_table(methods)

    # Cleanup temp dir
    if v2_temp_dir is not None:
        import shutil
        shutil.rmtree(v2_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
