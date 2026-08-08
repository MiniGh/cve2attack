"""Shared helpers for Stage 3 CVE to ATT&CK techniques mapping."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class _TqdmFallback:
        """Minimal tqdm-compatible wrapper for iterable progress loops."""

        def __init__(self, iterable, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix_str(self, *_args, **_kwargs) -> None:
            """No-op method to match tqdm API used by callers."""
            return None

        def update(self, _n: int = 1) -> None:
            """No-op progress update for compatibility with tqdm usage."""
            return None

        def close(self) -> None:
            """No-op close method for compatibility with tqdm usage."""
            return None

    def tqdm(iterable=None, **kwargs):
        """Fallback tqdm wrapper when tqdm is not installed."""
        if iterable is None:
            iterable = []
        return _TqdmFallback(iterable, **kwargs)


def setup_logging(verbose: bool = False) -> None:
    """Configure process-wide logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def load_json(file_path: Path) -> Any:
    """Load and return JSON content from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: Path) -> None:
    """Write JSON content using UTF-8 and pretty formatting."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def iter_cve_year_files(cve_dir: Path) -> List[Path]:
    """Return sorted yearly CVE JSON files under the source directory."""
    return sorted(cve_dir.glob("CVE-*.json"))


def iter_cve_records(year_file: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield CVE id and record pairs from a yearly JSON file."""
    data = load_json(year_file)
    if not isinstance(data, dict):
        return

    for cve_id, record in data.items():
        if isinstance(cve_id, str) and isinstance(record, dict):
            yield cve_id, record


def flatten_technique_list(techniques: Iterable[Dict[str, Any]]) -> str:
    """Render techniques as bullet lines for prompt context."""
    lines: List[str] = []
    for technique in techniques:
        technique_id = str(technique.get("id", "")).strip().upper()
        name = str(technique.get("name", "")).strip()
        description = str(technique.get("description", "")).strip()
        lines.append(f"- {technique_id}: {name}")
        if description:
            lines.append(f"  Description: {description}")
    return "\n".join(lines)


def normalize_main_technique_id(value: str) -> str:
    """Normalize an ATT&CK technique id to main-technique format T####."""
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
