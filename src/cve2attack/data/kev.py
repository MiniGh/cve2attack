"""Build fixed CTID KEV benchmark views from the published CSV snapshot.

The importer deliberately keeps CTID's mapping semantics instead of flattening
the source into an undocumented CVE-to-technique table.  The active stage-1
pipeline consumes the parent-normalized ``techniques`` field; exact ATT&CK
sub-technique labels and their mapping types remain available in every record
for future sub-technique-aware evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from cve2attack.schemas import parent_technique_id


KEV_SNAPSHOT = "02.13.2025"
KEV_ATTACK_VERSION = "15.1"
KEV_SOURCE_URL = "https://zenodo.org/records/16747173"
KEV_TECHNIQUE_CORPUS = "data/knowledge/enterprise-attack-15.1.json"
KEV_TECHNIQUE_CORPUS_URL = (
    "https://github.com/mitre-attack/attack-stix-data/releases/download/"
    "v15.1/enterprise-attack.json"
)
KEV_TECHNIQUE_CORPUS_SHA256 = "a57988bffe402bb3e19d92dbe80a12143e1970b814e013e080f9df2fa5a3f6bc"
KEV_MAPPING_TYPES = (
    "exploitation_technique",
    "primary_impact",
    "secondary_impact",
)
KEV_VIEW_NAMES = (
    "ctid_kev_2025_02_13_all",
    "ctid_kev_2025_02_13_exploitation",
    "ctid_kev_2025_02_13_nonoverlap",
)

_CVE_ID = re.compile(r"^CVE-\d{4}-\d+$")
_TECHNIQUE_ID = re.compile(r"^T\d{4}(?:\.\d{3})?$")
_REQUIRED_COLUMNS = {
    "mapping_framework",
    "mapping_framework_version",
    "capability_id",
    "mapping_type",
    "attack_object_id",
    "attack_object_name",
    "attack_version",
    "technology_domain",
    "references",
    "comments",
}


class KEVImportError(ValueError):
    """Raised when a supposed KEV snapshot is malformed or incompatible."""


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sorted_ids(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


def _existing_benchmark_ids(directory: Path) -> set[str]:
    identifiers: set[str] = set()
    for path in sorted(directory.glob("CVE-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise KEVImportError(
                        f"Invalid JSON at {path}:{line_number} while reading overlap IDs"
                    ) from exc
                cve_id = str(record.get("cve_id", "")).strip()
                if cve_id:
                    identifiers.add(cve_id)
    return identifiers


def _validate_row(row: dict[str, str], row_number: int) -> tuple[str, str, str]:
    cve_id = row["capability_id"].strip()
    technique_id = row["attack_object_id"].strip()
    mapping_type = row["mapping_type"].strip()
    if row["mapping_framework"].strip().lower() != "kev":
        raise KEVImportError(f"Row {row_number}: expected mapping_framework=kev")
    if row["technology_domain"].strip().lower() != "enterprise":
        raise KEVImportError(f"Row {row_number}: expected technology_domain=enterprise")
    if not _CVE_ID.fullmatch(cve_id):
        raise KEVImportError(f"Row {row_number}: invalid CVE ID: {cve_id!r}")
    if not _TECHNIQUE_ID.fullmatch(technique_id):
        raise KEVImportError(f"Row {row_number}: invalid ATT&CK ID: {technique_id!r}")
    if mapping_type not in KEV_MAPPING_TYPES:
        raise KEVImportError(f"Row {row_number}: unsupported mapping type: {mapping_type!r}")
    return cve_id, technique_id, mapping_type


def _load_records(source: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise KEVImportError(f"KEV CSV is missing columns: {', '.join(missing)}")

        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        attack_versions: set[str] = set()
        framework_versions: set[str] = set()
        row_count = 0
        for row_number, row in enumerate(reader, start=2):
            cve_id, technique_id, mapping_type = _validate_row(row, row_number)
            row_count += 1
            attack_versions.add(row["attack_version"].strip())
            framework_versions.add(row["mapping_framework_version"].strip())
            grouped[cve_id].append(
                {
                    "mapping_type": mapping_type,
                    "technique_id": technique_id,
                    "technique_parent_id": parent_technique_id(technique_id),
                    "technique_name": row["attack_object_name"].strip(),
                    "references": row["references"].strip(),
                    "comments": row["comments"].strip(),
                }
            )

    if not grouped:
        raise KEVImportError("KEV CSV contains no enterprise mappings")
    if attack_versions != {KEV_ATTACK_VERSION}:
        raise KEVImportError(
            f"Expected ATT&CK version {KEV_ATTACK_VERSION}, found {sorted(attack_versions)}"
        )
    if framework_versions != {"02/13/2025"}:
        raise KEVImportError(
            "Expected CTID KEV framework version 02/13/2025, "
            f"found {sorted(framework_versions)}"
        )

    records: dict[str, dict[str, Any]] = {}
    for cve_id, labels in grouped.items():
        by_type: dict[str, dict[str, list[str]]] = {}
        for mapping_type in KEV_MAPPING_TYPES:
            type_labels = [item for item in labels if item["mapping_type"] == mapping_type]
            by_type[mapping_type] = {
                "techniques": _sorted_ids(
                    item["technique_parent_id"] for item in type_labels
                ),
                "techniques_raw": _sorted_ids(item["technique_id"] for item in type_labels),
            }

        metadata_by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
        for item in labels:
            key = (
                item["mapping_type"],
                item["technique_id"],
                item["technique_name"],
                item["references"],
                item["comments"],
            )
            metadata_by_key[key] = item

        records[cve_id] = {
            "cve_id": cve_id,
            "techniques": _sorted_ids(item["technique_parent_id"] for item in labels),
            "techniques_raw": _sorted_ids(item["technique_id"] for item in labels),
            "labels_by_mapping_type": by_type,
            "label_metadata": [
                metadata_by_key[key] for key in sorted(metadata_by_key)
            ],
            "provenance": {
                "dataset": "CTID KEV",
                "snapshot": KEV_SNAPSHOT,
                "attack_version": KEV_ATTACK_VERSION,
                "source_url": KEV_SOURCE_URL,
            },
        }

    return records, {
        "mapping_rows": row_count,
        "mapped_cves": len(records),
        "attack_version": KEV_ATTACK_VERSION,
        "framework_version": "02/13/2025",
    }


def _write_view(
    *,
    directory: Path,
    name: str,
    records: dict[str, dict[str, Any]],
    source: Path,
    source_md5: str,
    source_sha256: str,
    source_stats: dict[str, Any],
    view_description: str,
) -> None:
    directory.mkdir(parents=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cve_id, record in records.items():
        year = cve_id.split("-", 2)[1]
        grouped[year].append(record)
    for year, rows in sorted(grouped.items()):
        path = directory / f"CVE-{year}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in sorted(rows, key=lambda item: item["cve_id"]):
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    metadata = {
        "name": name,
        "kind": "benchmark",
        "task": "cve_to_attack_technique_candidate_coverage",
        "source": {
            "publisher": "MITRE Center for Threat-Informed Defense",
            "dataset": "Known Exploited Vulnerabilities (KEV) ATT&CK mappings",
            "landing_page": KEV_SOURCE_URL,
            "raw_file": source.name,
            "raw_md5": source_md5,
            "raw_sha256": source_sha256,
        },
        "snapshot": KEV_SNAPSHOT,
        "attack_version": KEV_ATTACK_VERSION,
        "technique_corpus": {
            "path": KEV_TECHNIQUE_CORPUS,
            "version": KEV_ATTACK_VERSION,
            "source_url": KEV_TECHNIQUE_CORPUS_URL,
            "sha256": KEV_TECHNIQUE_CORPUS_SHA256,
        },
        "input_text_policy": (
            "Use only the project's raw CVE description as model input. "
            "KEV comments and references are retained as label provenance, not input text."
        ),
        "label_policy": {
            "evaluation_field": "techniques",
            "normalization": "ATT&CK sub-techniques are rolled up to their parent technique.",
            "raw_label_field": "techniques_raw",
            "mapping_type_field": "labels_by_mapping_type",
        },
        "view": view_description,
        "annotated_cves": len(records),
        "record_schema": {
            "cve_id": "string",
            "techniques": "list[string] (parent-normalized ATT&CK IDs)",
            "techniques_raw": "list[string] (source ATT&CK IDs)",
            "labels_by_mapping_type": "mapping type to normalized and raw IDs",
            "label_metadata": "list[mapping provenance]",
            "provenance": "source snapshot metadata",
        },
        "source_stats": source_stats,
    }
    (directory / "dataset.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def import_kev_benchmarks(
    *,
    source: Path,
    benchmark_root: Path,
    cve2attack_benchmark: Path,
) -> dict[str, Any]:
    """Create the fixed ``all``, ``exploitation`` and strict ``nonoverlap`` views.

    Existing target directories are rejected.  This prevents a later import from
    silently changing a frozen benchmark in place.
    """
    if not source.is_file():
        raise FileNotFoundError(f"KEV source CSV does not exist: {source}")
    if not cve2attack_benchmark.is_dir():
        raise FileNotFoundError(
            f"CVE2ATT&CK benchmark directory does not exist: {cve2attack_benchmark}"
        )
    targets = [benchmark_root / name for name in KEV_VIEW_NAMES]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite frozen KEV benchmark directories: " + ", ".join(existing)
        )

    all_records, source_stats = _load_records(source)
    cve2attack_ids = _existing_benchmark_ids(cve2attack_benchmark)
    exploitation_records = {
        cve_id: record
        for cve_id, record in all_records.items()
        if record["labels_by_mapping_type"]["exploitation_technique"]["techniques"]
    }
    nonoverlap_records = {
        cve_id: record for cve_id, record in all_records.items() if cve_id not in cve2attack_ids
    }
    source_md5 = _digest(source, "md5")
    source_sha256 = _digest(source, "sha256")

    views = (
        (
            KEV_VIEW_NAMES[0],
            all_records,
            "All exploitation and impact mappings; use this for the context-aware stage-1 goal.",
        ),
        (
            KEV_VIEW_NAMES[1],
            exploitation_records,
            "Only CVEs with CTID exploitation_technique labels.",
        ),
        (
            KEV_VIEW_NAMES[2],
            nonoverlap_records,
            "All mapping types after excluding CVE IDs present in cve2attack_result.",
        ),
    )
    for name, records, description in views:
        _write_view(
            directory=benchmark_root / name,
            name=name,
            records=records,
            source=source,
            source_md5=source_md5,
            source_sha256=source_sha256,
            source_stats=source_stats,
            view_description=description,
        )

    return {
        "source": str(source),
        "raw_md5": source_md5,
        "raw_sha256": source_sha256,
        "source_stats": source_stats,
        "cve2attack_overlap": len(set(all_records) & cve2attack_ids),
        "views": {
            name: {"annotated_cves": len(records)} for name, records, _ in views
        },
    }
