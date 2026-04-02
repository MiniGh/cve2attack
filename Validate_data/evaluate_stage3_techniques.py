"""Evaluate Stage-3 CVE->Techniques mapping with multi-label metrics.

This script compares predicted techniques from stage3_cve_techniques results against
ground-truth techniques in Validate_data/cve2technique_full.jsonl.
Outputs per-CVE Precision/Recall/F1 as JSONL and reports Stage-3 EmptyRate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


CVE_YEAR_RE = re.compile(r"^CVE-(\d{4})-")


def extract_year_from_cve_id(cve_id: str) -> Optional[int]:
    """Extract CVE year from an ID such as CVE-2021-1234."""
    match = CVE_YEAR_RE.match(str(cve_id).strip())
    if not match:
        return None
    return int(match.group(1))


def normalize_main_technique_id(value: str) -> str:
    """Normalize ATT&CK technique ID to main-technique format T####."""
    text = str(value).strip().upper()
    if not text:
        return ""

    if text.startswith("T"):
        body = text[1:]
    else:
        body = text

    if "." in body:
        body = body.split(".", 1)[0]

    if not body.isdigit():
        return ""

    body = body.zfill(4)
    return f"T{body}"


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


def load_gt_techniques_by_cve(gt_jsonl_file: Path) -> Dict[str, Set[str]]:
    """Load GT techniques from JSONL file keyed by cve_id."""
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

            techniques = record.get("techniques", [])
            if not isinstance(techniques, list):
                techniques = []

            normalized: Set[str] = set()
            for technique in techniques:
                technique_id = normalize_main_technique_id(str(technique))
                if technique_id:
                    normalized.add(technique_id)

            gt[cve_id] = normalized

    return gt


def load_predictions_by_cve(result_dir: Path) -> Dict[str, Set[str]]:
    """Load stage-3 prediction files and return per-CVE predicted techniques."""
    pred: Dict[str, Set[str]] = {}
    for year_file in sorted(result_dir.glob("CVE-*.json")):
        with open(year_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        for cve_id, techniques in data.items():
            if not isinstance(cve_id, str):
                continue
            if not isinstance(techniques, list):
                techniques = []

            normalized: Set[str] = set()
            for technique in techniques:
                technique_id = normalize_main_technique_id(str(technique))
                if technique_id:
                    normalized.add(technique_id)
            pred[cve_id] = normalized

    return pred


def load_stage2_non_empty_flags(stage2_dir: Path) -> Dict[str, bool]:
    """Load whether each CVE has non-empty stage-2 tactics."""
    flags: Dict[str, bool] = {}
    for year_file in sorted(stage2_dir.glob("CVE-*.json")):
        with open(year_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        for cve_id, tactics in data.items():
            has_tactics = isinstance(tactics, list) and any(str(x).strip() for x in tactics)
            if isinstance(cve_id, str):
                flags[cve_id] = has_tactics

    return flags


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


def compute_empty_rate(
    pred_by_cve: Dict[str, Set[str]],
    stage2_non_empty_flags: Dict[str, bool],
    start_year: Optional[int],
    end_year: Optional[int],
) -> Tuple[int, int, float]:
    """Compute EmptyRate using denominator=stage2 non-empty CVEs."""
    denominator = 0
    empty_count = 0

    for cve_id, has_stage2 in stage2_non_empty_flags.items():
        if not has_stage2:
            continue
        if not in_year_range(cve_id, start_year, end_year):
            continue

        denominator += 1
        pred_set = pred_by_cve.get(cve_id, set())
        if not pred_set:
            empty_count += 1

    empty_rate = safe_div(empty_count, denominator)
    return empty_count, denominator, empty_rate


def main() -> None:
    """CLI entry point for Stage-3 techniques evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Stage-3 CVE->Techniques mapping")
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("Validate_data") / "cve2technique_full.jsonl",
        help="Ground-truth JSONL file (must include techniques field)",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("stage3_cve_techniques") / "result_sec-i1",
        help="Directory containing yearly prediction JSON files",
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path("stage2_cve_tactics") / "result_sec-i1",
        help="Directory containing yearly stage-2 prediction JSON files",
    )
    parser.add_argument(
        "-s",
        "--start-year",
        type=int,
        default=None,
        help="Start year (inclusive), e.g. 2022",
    )
    parser.add_argument(
        "-e",
        "--end-year",
        type=int,
        default=None,
        help="End year (inclusive), e.g. 2022",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Validate_data") / "stage3_techniques_eval.jsonl",
        help="Output JSONL file path",
    )
    args = parser.parse_args()

    if args.start_year is not None and args.end_year is not None and args.start_year > args.end_year:
        raise ValueError("--start-year cannot be greater than --end-year")

    if not args.gt.exists():
        raise FileNotFoundError(f"GT file not found: {args.gt}")
    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {args.pred_dir}")
    if not args.stage2_dir.exists():
        raise FileNotFoundError(f"Stage-2 directory not found: {args.stage2_dir}")

    gt_by_cve = load_gt_techniques_by_cve(args.gt)
    pred_by_cve = load_predictions_by_cve(args.pred_dir)
    stage2_non_empty_flags = load_stage2_non_empty_flags(args.stage2_dir)

    records = evaluate_records(gt_by_cve, pred_by_cve, args.start_year, args.end_year)
    count = save_jsonl(records, args.output)

    empty_count, denominator, empty_rate = compute_empty_rate(
        pred_by_cve=pred_by_cve,
        stage2_non_empty_flags=stage2_non_empty_flags,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    print(f"[INFO] GT CVEs loaded: {len(gt_by_cve)}")
    print(f"[INFO] Prediction CVEs loaded: {len(pred_by_cve)}")
    print(f"[INFO] Saved evaluation records: {count}")
    print(f"[INFO] Output file: {args.output}")
    print(f"[INFO] Empty predictions on stage2-non-empty CVEs: {empty_count}/{denominator}")
    print(f"[INFO] EmptyRate: {empty_rate:.6f} ({empty_rate * 100:.2f}%)")


if __name__ == "__main__":
    main()
