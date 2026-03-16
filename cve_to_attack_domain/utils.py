"""Utility helpers for loading CVE data and extracting matching tokens."""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Set

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        """Fallback tqdm wrapper when tqdm is not installed."""
        return iterable


def normalize_text(text: str) -> str:
    """Lower-case text and replace punctuation with spaces."""
    lowered = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered)


def tokenize(text: str) -> Set[str]:
    """Tokenize free text into normalized word tokens."""
    normalized = normalize_text(text)
    return {tok for tok in normalized.split() if tok}


def tokenize_source_identifier(source_identifier: str) -> Set[str]:
    """Tokenize NVD sourceIdentifier values with the same normalization rules."""
    return tokenize(source_identifier)


def extract_cpe_tokens(cpes: Iterable[str]) -> Set[str]:
    """Extract vendor and product tokens from CPE 2.3 strings."""
    tokens: Set[str] = set()
    for cpe in cpes or []:
        parts = cpe.split(":")
        if len(parts) >= 5 and parts[0] == "cpe" and parts[1] == "2.3":
            vendor = parts[3].strip().lower()
            product = parts[4].strip().lower()
            if vendor and vendor != "*":
                tokens.add(vendor)
            if product and product != "*":
                tokens.add(product)
    return tokens


def load_cve_records(cve_dir: Path) -> Dict[str, Dict]:
    """Load all CVE records from yearly JSON files under the provided directory."""
    records: Dict[str, Dict] = {}
    files = sorted(cve_dir.glob("CVE-*.json"))

    for file_path in tqdm(files, desc="Loading CVE files", unit="file"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for cve_id, record in data.items():
                if isinstance(record, dict):
                    records[cve_id] = record

    return records


def save_mapping(mapping: Dict[str, str], output_file: Path) -> None:
    """Persist CVE to domain mapping as pretty-printed JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
