"""Import the frozen public TRIAGE test split and its mapping semantics."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from cve2attack.schemas import parent_technique_id


TRIAGE_SOURCE_URL = "https://zenodo.org/records/17341504"
TRIAGE_PAPER_URL = "https://arxiv.org/abs/2508.18439"
TRIAGE_PACKAGE_MD5 = "d3d4a603554c3e97f13ba3e6e9dc5832"
TRIAGE_ATTACK_VERSION = "15.1"
TRIAGE_MAPPING_TYPES = (
    "exploitation_technique",
    "primary_impact",
    "secondary_impact",
)
TRIAGE_VIEW_NAMES = (
    "triage_2025_test_all",
    "triage_2025_test_no_secondary",
)


class TRIAGEImportError(ValueError):
    """Raised when selected files from the TRIAGE package are inconsistent."""


def _sorted_ids(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


def _read_split(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "CVE ID" not in (reader.fieldnames or []):
            raise TRIAGEImportError(f"Split file is missing 'CVE ID': {path}")
        values = [str(row["CVE ID"]).strip() for row in reader]
    if not values or any(not value for value in values):
        raise TRIAGEImportError(f"Split file contains empty or no CVE IDs: {path}")
    if len(values) != len(set(values)):
        raise TRIAGEImportError(f"Split file contains duplicate CVE IDs: {path}")
    return values


def _read_expected_counts(source_dir: Path) -> dict[str, int]:
    metadata_path = source_dir / "source.yaml"
    if not metadata_path.exists():
        return {}
    value = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    expected = value.get("expected_counts", {}) if isinstance(value, Mapping) else {}
    return {str(key): int(item) for key, item in expected.items()}


def _read_labels(source_dir: Path) -> tuple[dict[str, list[dict[str, str]]], int]:
    path = source_dir / "labeled_cve_to_attack.csv"
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"CVE ID", "mapping_type", "attack_id", "attack_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise TRIAGEImportError(f"Label file is missing columns: {sorted(missing)}")
        row_count = 0
        for row_number, row in enumerate(reader, start=2):
            cve_id = str(row["CVE ID"]).strip()
            mapping_type = str(row["mapping_type"]).strip()
            technique_id = str(row["attack_id"]).strip().upper()
            if mapping_type not in TRIAGE_MAPPING_TYPES:
                raise TRIAGEImportError(
                    f"Row {row_number}: unsupported mapping type {mapping_type!r}"
                )
            if not cve_id.startswith("CVE-") or not technique_id.startswith("T"):
                raise TRIAGEImportError(f"Row {row_number}: invalid CVE or technique ID")
            row_count += 1
            grouped[cve_id].append(
                {
                    "mapping_type": mapping_type,
                    "technique_id": technique_id,
                    "technique_parent_id": parent_technique_id(technique_id),
                    "technique_name": str(row["attack_name"]).strip(),
                }
            )
    return dict(grouped), row_count


def _record(cve_id: str, labels: list[dict[str, str]], *, exclude_secondary: bool) -> dict[str, Any]:
    selected = [
        item
        for item in labels
        if not (exclude_secondary and item["mapping_type"] == "secondary_impact")
    ]
    by_type: dict[str, dict[str, list[str]]] = {}
    for mapping_type in TRIAGE_MAPPING_TYPES:
        type_labels = [item for item in selected if item["mapping_type"] == mapping_type]
        by_type[mapping_type] = {
            "techniques": _sorted_ids(item["technique_parent_id"] for item in type_labels),
            "techniques_raw": _sorted_ids(item["technique_id"] for item in type_labels),
        }
    return {
        "cve_id": cve_id,
        "techniques": _sorted_ids(item["technique_parent_id"] for item in selected),
        "techniques_raw": _sorted_ids(item["technique_id"] for item in selected),
        "labels_by_mapping_type": by_type,
        "label_metadata": selected,
        "provenance": {
            "dataset": "TRIAGE replication package",
            "split": "test",
            "attack_version": TRIAGE_ATTACK_VERSION,
            "source_url": TRIAGE_SOURCE_URL,
        },
    }


def _write_view(
    directory: Path,
    *,
    name: str,
    records: list[dict[str, Any]],
    view: str,
    source_dir: Path,
) -> None:
    if directory.exists():
        raise FileExistsError(f"TRIAGE benchmark view already exists: {directory}")
    directory.mkdir(parents=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["cve_id"].split("-", 2)[1]].append(record)
    for year, year_records in sorted(grouped.items()):
        path = directory / f"CVE-{year}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in sorted(year_records, key=lambda item: item["cve_id"]):
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    metadata = {
        "name": name,
        "kind": "benchmark",
        "task": "cve_to_attack_technique_candidate_coverage",
        "source": {
            "publisher": "TRIAGE authors",
            "paper": TRIAGE_PAPER_URL,
            "replication_package": TRIAGE_SOURCE_URL,
            "package_md5": TRIAGE_PACKAGE_MD5,
            "selected_files": "data/raw/triage/triage_2025",
        },
        "split": {
            "policy": "Exact public TRIAGE test split; 80/20 split over CVE IDs.",
            "name": "test",
            "annotated_cves": len(records),
        },
        "attack_version": TRIAGE_ATTACK_VERSION,
        "technique_corpus": {
            "path": "data/knowledge/enterprise-attack-15.1.json",
            "version": TRIAGE_ATTACK_VERSION,
        },
        "label_policy": {
            "evaluation_field": "techniques",
            "normalization": "ATT&CK sub-techniques are rolled up to parent techniques.",
            "raw_label_field": "techniques_raw",
            "mapping_type_field": "labels_by_mapping_type",
        },
        "view": view,
        "record_schema": {
            "cve_id": "string",
            "techniques": "list[string] (parent-normalized ATT&CK IDs)",
            "techniques_raw": "list[string] (source ATT&CK IDs)",
            "labels_by_mapping_type": "mapping type to normalized and raw IDs",
            "label_metadata": "selected rows from the public label file",
            "provenance": "source and split metadata",
        },
    }
    (directory / "dataset.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def import_triage_benchmarks(*, source_dir: Path, benchmark_root: Path) -> dict[str, Any]:
    """Create exact public TRIAGE test views from selected replication files."""
    print(f"[triage] reading frozen split and labels from {source_dir}")
    train_ids = _read_split(source_dir / "cves_train.csv")
    test_ids = _read_split(source_dir / "cves_test.csv")
    labels_by_cve, label_rows = _read_labels(source_dir)
    if set(train_ids).intersection(test_ids):
        raise TRIAGEImportError("TRIAGE train and test splits overlap")
    if set(train_ids).union(test_ids) != set(labels_by_cve):
        raise TRIAGEImportError("TRIAGE split CVEs do not match labeled CVEs")

    actual_counts = {
        "train_cves": len(train_ids),
        "test_cves": len(test_ids),
        "labeled_cves": len(labels_by_cve),
        "label_rows": label_rows,
    }
    for key, expected in _read_expected_counts(source_dir).items():
        if key in actual_counts and actual_counts[key] != expected:
            raise TRIAGEImportError(
                f"Expected {key}={expected}, found {actual_counts[key]}"
            )

    all_records = [_record(cve_id, labels_by_cve[cve_id], exclude_secondary=False) for cve_id in test_ids]
    no_secondary_records = [
        _record(cve_id, labels_by_cve[cve_id], exclude_secondary=True) for cve_id in test_ids
    ]
    if any(not record["techniques"] for record in all_records + no_secondary_records):
        raise TRIAGEImportError("A TRIAGE main test view contains a CVE without labels")

    views = {
        TRIAGE_VIEW_NAMES[0]: (
            all_records,
            "All exploitation, primary-impact, and secondary-impact labels on the public test split.",
        ),
        TRIAGE_VIEW_NAMES[1]: (
            no_secondary_records,
            "Public test split with secondary-impact labels excluded, matching the paper ablation.",
        ),
    }
    stats: dict[str, Any] = {"source": actual_counts, "views": {}}
    for name, (records, description) in views.items():
        print(f"[triage] creating {name}: {len(records)} CVEs")
        _write_view(
            benchmark_root / name,
            name=name,
            records=records,
            view=description,
            source_dir=source_dir,
        )
        stats["views"][name] = {
            "cves": len(records),
            "parent_labels": sum(len(record["techniques"]) for record in records),
        }
    return stats
