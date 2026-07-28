"""Create reproducible label-independent subsets of very large benchmarks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from cve2attack.data.loaders import iter_jsonl


def _selection_key(cve_id: str, seed: str) -> tuple[str, str]:
    """Return a stable pseudo-random ordering that never inspects labels."""
    digest = hashlib.sha256(f"{seed}\0{cve_id}".encode("utf-8")).hexdigest()
    return digest, cve_id


def sample_benchmark(
    *,
    source_dir: Path,
    output_dir: Path,
    sample_size: int,
    seed: str,
) -> dict[str, Any]:
    """Write the first ``sample_size`` CVEs in a seeded SHA-256 ordering.

    Sampling uses only the CVE identifier. Technique labels, descriptions and
    model predictions cannot influence membership in the frozen subset.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source benchmark does not exist: {source_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Sample benchmark already exists: {output_dir}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not seed.strip():
        raise ValueError("seed must not be empty")

    records: dict[str, dict[str, Any]] = {}
    for path in sorted(source_dir.glob("CVE-*.jsonl")):
        for record in iter_jsonl(path):
            cve_id = str(record.get("cve_id") or "").strip().upper()
            if not cve_id:
                continue
            if cve_id in records:
                raise ValueError(f"Duplicate CVE in source benchmark: {cve_id}")
            records[cve_id] = dict(record)
    if sample_size > len(records):
        raise ValueError(
            f"sample_size={sample_size} exceeds source size {len(records)}"
        )

    selected_ids = sorted(records, key=lambda cve_id: _selection_key(cve_id, seed))[
        :sample_size
    ]
    selected = [records[cve_id] for cve_id in selected_ids]

    source_metadata_path = source_dir / "dataset.yaml"
    source_metadata: dict[str, Any] = {}
    if source_metadata_path.is_file():
        value = yaml.safe_load(source_metadata_path.read_text(encoding="utf-8")) or {}
        if not isinstance(value, Mapping):
            raise ValueError(f"Benchmark metadata must be a mapping: {source_metadata_path}")
        source_metadata = dict(value)

    metadata = {
        **source_metadata,
        "name": output_dir.name,
        "annotated_cves": len(selected),
        "derived_from": source_dir.name,
        "sampling": {
            "method": "smallest_sha256(seed + NUL + cve_id)",
            "seed": seed,
            "source_records": len(records),
            "sample_size": len(selected),
            "label_independent": True,
        },
        "notes": (
            f"Deterministic supplementary sample of {source_dir.name}; "
            "membership uses CVE IDs only and does not establish label authority."
        ),
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        cve_id = str(record["cve_id"]).upper()
        parts = cve_id.split("-")
        if len(parts) < 3:
            raise ValueError(f"Unexpected CVE ID: {cve_id}")
        grouped[parts[1]].append(record)

    output_dir.mkdir(parents=True)
    for year, year_records in sorted(grouped.items()):
        path = output_dir / f"CVE-{year}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in sorted(year_records, key=lambda item: str(item["cve_id"])):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "dataset.yaml").write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "cohort.json").write_text(
        json.dumps(sorted(selected_ids), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "source": source_dir.name,
        "output": output_dir.name,
        "source_records": len(records),
        "sampled_records": len(selected),
        "years": len(grouped),
        "seed": seed,
    }
