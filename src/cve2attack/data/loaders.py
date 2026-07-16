"""Readers for CVE records, benchmarks, domain mappings and JSONL candidates."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

from cve2attack.schemas import CandidateRecord, parent_technique_id


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(value, dict):
                yield value


def benchmark_truth(directory: Path, roll_up: bool = True) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = defaultdict(set)
    for path in sorted(directory.glob("CVE-*.jsonl")):
        for record in iter_jsonl(path):
            cve_id = str(record.get("cve_id", "")).strip()
            if not cve_id:
                continue
            for raw in record.get("techniques", []) or []:
                if isinstance(raw, Mapping):
                    raw = raw.get("technique_id") or raw.get("tech_id") or raw.get("id")
                if raw is None:
                    continue
                technique_id = parent_technique_id(str(raw)) if roll_up else str(raw).strip()
                if technique_id:
                    truth[cve_id].add(technique_id)
    return dict(truth)


def candidate_records(directory: Path) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    candidate_dir = directory / "candidates" if (directory / "candidates").is_dir() else directory
    for path in sorted(candidate_dir.glob("CVE-*.jsonl")):
        records.extend(CandidateRecord.from_dict(record) for record in iter_jsonl(path))
    return records


def write_candidate_records(records: Sequence[CandidateRecord], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, list[CandidateRecord]] = defaultdict(list)
    for record in records:
        parts = record.cve_id.split("-")
        if len(parts) < 3:
            raise ValueError(f"Unexpected CVE ID: {record.cve_id}")
        grouped[parts[1]].append(record)

    paths: list[Path] = []
    for year, year_records in sorted(grouped.items()):
        path = directory / f"CVE-{year}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in sorted(year_records, key=lambda item: item.cve_id):
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        paths.append(path)
    return paths


def enterprise_cve_ids(domain_dir: Path) -> list[str]:
    seen: set[str] = set()
    identifiers: list[str] = []
    for path in sorted(domain_dir.glob("CVE-*.jsonl")):
        for record in iter_jsonl(path):
            cve_id = str(record.get("cve_id", "")).strip()
            if cve_id and record.get("domain") == "Enterprise" and cve_id not in seen:
                seen.add(cve_id)
                identifiers.append(cve_id)
    return sorted(identifiers)


class CVERepository:
    """Lazy yearly CVE reader; each source file is loaded at most once."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._year_cache: dict[str, dict[str, dict[str, Any]]] = {}

    def _year(self, year: str) -> dict[str, dict[str, Any]]:
        if year not in self._year_cache:
            path = self.directory / f"CVE-{year}.json"
            if not path.exists():
                self._year_cache[year] = {}
            else:
                with path.open("r", encoding="utf-8") as handle:
                    value = json.load(handle)
                self._year_cache[year] = value if isinstance(value, dict) else {}
        return self._year_cache[year]

    def get(self, cve_id: str) -> dict[str, Any] | None:
        parts = cve_id.split("-")
        if len(parts) < 3:
            return None
        record = self._year(parts[1]).get(cve_id)
        return record if isinstance(record, dict) else None

    def description(self, cve_id: str) -> str | None:
        record = self.get(cve_id)
        if not record:
            return None
        value = str(record.get("description") or "").strip()
        return value or None

    def cwes(self, cve_id: str) -> list[str]:
        record = self.get(cve_id) or {}
        return [str(item).removeprefix("CWE-") for item in record.get("cwes", []) or []]


def load_json_mapping(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return {str(key): str(text) for key, text in value.items() if str(text).strip()}
