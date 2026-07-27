"""Experiment configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULTS: Dict[str, Any] = {
    "input": {"mode": "benchmark", "benchmark": "cve2attack_result"},
    "query": {"strategy": "raw_description"},
    "technique_document": {
        "include_procedures": False,
        "procedure_char_limit": 1500,
    },
    "action_document": {
        "include_descriptions": True,
        "include_procedures": True,
        "min_chars": 20,
        "max_chars": 1200,
        "exclude_query_cve_actions": True,
    },
    "retrieval": {
        "model": "basel/ATTACK-BERT",
        "corpus": "technique",
        "top_k": 20,
        "batch_size": 32,
        "local_files_only": True,
        "aggregation": "max",
        "aggregation_top_m": 3,
        "rank_constant": 60.0,
        "evidence_limit": 3,
        "evidence_text_limit": 400,
    },
    "fusion": {"strategy": "none"},
    "evaluation": {"benchmarks": []},
}


def _merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def load_experiment(path: Path) -> Dict[str, Any]:
    """Load an experiment YAML file and apply stable defaults."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Experiment config must be a mapping: {path}")

    config = _merge(DEFAULTS, raw)
    if not str(config.get("name", "")).strip():
        raise ValueError(f"Experiment config is missing a name: {path}")

    input_mode = config["input"].get("mode")
    if input_mode not in {"benchmark", "full_enterprise"}:
        raise ValueError(f"Unsupported input.mode: {input_mode}")

    query_strategy = config["query"].get("strategy")
    if query_strategy not in {"raw_description", "rewrite_cache"}:
        raise ValueError(f"Unsupported query.strategy: {query_strategy}")

    retrieval_corpus = config["retrieval"].get("corpus")
    if retrieval_corpus not in {"technique", "action"}:
        raise ValueError(f"Unsupported retrieval.corpus: {retrieval_corpus}")
    action_aggregation = config["retrieval"].get("aggregation")
    if action_aggregation not in {"max", "rank_rrf"}:
        raise ValueError(f"Unsupported retrieval.aggregation: {action_aggregation}")
    if int(config["retrieval"].get("aggregation_top_m", 0)) <= 0:
        raise ValueError("retrieval.aggregation_top_m must be positive")
    if float(config["retrieval"].get("rank_constant", 0)) <= 0:
        raise ValueError("retrieval.rank_constant must be positive")

    fusion_strategy = config["fusion"].get("strategy")
    if fusion_strategy not in {"none", "structured_chain"}:
        raise ValueError(f"Unsupported fusion.strategy: {fusion_strategy}")
    return config


def project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else project_root / path
