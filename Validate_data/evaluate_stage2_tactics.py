"""Evaluate Stage-2 CVE->Tactics mapping with multi-label metrics.

This script compares predicted tactics from stage2_cve_tactics/result against
ground-truth tactics in Validate_data/cve2technique_full_with_tactics.jsonl.
Outputs per-CVE Precision/Recall/F1 as JSONL.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


CVE_YEAR_RE = re.compile(r"^CVE-(\d{4})-")


def extract_year_from_cve_id(cve_id: str) -> Optional[int]:
    """Extract CVE year from an ID such as CVE-2021-1234."""
    match = CVE_YEAR_RE.match(str(cve_id).strip())
    if not match:
        return None
    return int(match.group(1))


def safe_div(numerator: int, denominator: int) -> float:
    """Return safe floating-point division with zero-denominator guard."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def f1_score(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_gt_tactics_by_cve(gt_jsonl_file: Path) -> Dict[str, Set[str]]:
    """Load GT tactics from JSONL file keyed by cve_id."""
    gt: Dict[str, Set[str]] = {}
    with open(gt_jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            cve_id = str(record.get("cve_id", "")).strip()
            if not cve_id:
                continue

            tactics = record.get("tactics", [])
            if not isinstance(tactics, list):
                tactics = []
            gt[cve_id] = {str(x).strip().upper() for x in tactics if str(x).strip()}

    return gt


def load_predictions_by_cve(result_dir: Path) -> Dict[str, Set[str]]:
    """Load stage-2 prediction files and return per-CVE predicted tactics."""
    pred: Dict[str, Set[str]] = {}
    for year_file in sorted(result_dir.glob("CVE-*.json")):
        with open(year_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        for cve_id, tactics in data.items():
            if not isinstance(cve_id, str):
                continue
            if not isinstance(tactics, list):
                tactics = []
            pred[cve_id] = {str(x).strip().upper() for x in tactics if str(x).strip()}

    return pred


def in_year_range(cve_id: str, start_year: Optional[int], end_year: Optional[int]) -> bool:
    """Check whether a CVE falls inside the requested year range."""
    year = extract_year_from_cve_id(cve_id)
    if year is None:
        return False
    if start_year is not None and year < start_year:
        return False
    if end_year is not None and year > end_year:
        return False
    return True


def evaluate_records(
    gt_by_cve: Dict[str, Set[str]],
    pred_by_cve: Dict[str, Set[str]],
    start_year: Optional[int],
    end_year: Optional[int],
) -> Iterable[Dict[str, object]]:
    """Yield per-CVE metric records for the selected year range."""
    for cve_id in sorted(gt_by_cve.keys()):
        if not in_year_range(cve_id, start_year, end_year):
            continue

        gt_set = gt_by_cve.get(cve_id, set())
        pred_set = pred_by_cve.get(cve_id, set())

        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = f1_score(precision, recall)

        yield {
            "cve_id": cve_id,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": round(precision, 6),
            "Recall": round(recall, 6),
            "F1": round(f1, 6),
        }


def save_jsonl(records: Iterable[Dict[str, object]], output_file: Path) -> int:
    """Write metric records to JSONL and return number of saved lines."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    """CLI entry point for Stage-2 tactics evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 CVE->Tactics mapping")
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("Validate_data") / "cve2technique_full_with_tactics.jsonl",
        help="Ground-truth JSONL file (must include tactics field)",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("stage2_cve_tactics") / "result",
        help="Directory containing yearly prediction JSON files",
    )
    parser.add_argument(
        "-s",
        "--start-year",
        type=int,
        default=None,
        help="Start year (inclusive), e.g. 2021",
    )
    parser.add_argument(
        "-e",
        "--end-year",
        type=int,
        default=None,
        help="End year (inclusive), e.g. 2023",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Validate_data") / "stage2_tactics_eval.jsonl",
        help="Output JSONL file path",
    )
    args = parser.parse_args()

    if args.start_year is not None and args.end_year is not None and args.start_year > args.end_year:
        raise ValueError("--start-year cannot be greater than --end-year")

    if not args.gt.exists():
        raise FileNotFoundError(f"GT file not found: {args.gt}")
    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {args.pred_dir}")

    gt_by_cve = load_gt_tactics_by_cve(args.gt)
    pred_by_cve = load_predictions_by_cve(args.pred_dir)
    records = evaluate_records(gt_by_cve, pred_by_cve, args.start_year, args.end_year)
    count = save_jsonl(records, args.output)

    print(f"[INFO] GT CVEs loaded: {len(gt_by_cve)}")
    print(f"[INFO] Prediction CVEs loaded: {len(pred_by_cve)}")
    print(f"[INFO] Saved evaluation records: {count}")
    print(f"[INFO] Output file: {args.output}")


if __name__ == "__main__":
    main()
