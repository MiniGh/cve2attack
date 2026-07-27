"""Candidate-source complementarity diagnostics for the public TRIAGE test split.

This module deliberately keeps diagnostic metrics separate from the compact
paper-reproduction metrics in :mod:`cve2attack.evaluation.ranking`.  The public
TRIAGE histories are truncated at rank 20, while project and SMET histories can
be longer; representing unavailable points as ``None`` prevents a truncated
list from being mistaken for a real Recall@30/50 result.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence, Set

from cve2attack.data.loaders import CVERepository, benchmark_truth, candidate_records, iter_jsonl
from cve2attack.evaluation.ranking import rankings_from_records
from cve2attack.evaluation.triage import (
    REFERENCE_FILES,
    TRIAGE_ALL_BENCHMARK,
    load_reference_history,
)
from cve2attack.schemas import parent_technique_id


DEFAULT_CUTOFFS = (1, 3, 5, 10, 20, 30, 50)
RANK_BINS = (
    ("rank_1", 1, 1),
    ("rank_2_3", 2, 3),
    ("rank_4_5", 4, 5),
    ("rank_6_10", 6, 10),
    ("rank_11_20", 11, 20),
    ("rank_21_30", 21, 30),
    ("rank_31_50", 31, 50),
    ("rank_over_50", 51, None),
)
VALID_PARENT_TECHNIQUE = re.compile(r"^T\d{4}$")


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _run_name(run_dir: Path) -> str:
    """Use the experiment name as the stable display label when possible."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return run_dir.name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(manifest.get("experiment") or manifest.get("run_id") or run_dir.name)


def _require_complete_coverage(
    predictions: Mapping[str, Sequence[str]],
    cohort: Iterable[str],
    *,
    source_name: str,
) -> None:
    """Fail early when a full-ranking diagnostic source omits cohort CVEs."""
    missing = [cve_id for cve_id in cohort if not predictions.get(cve_id)]
    if not missing:
        return
    examples = ", ".join(missing[:3])
    raise ValueError(
        f"Diagnostic source {source_name!r} does not fully cover the TRIAGE test "
        f"cohort: missing {len(missing)} CVEs (examples: {examples})"
    )


def _canonical_candidates(ranked: Sequence[str], cutoff: int) -> list[str]:
    """Return unique valid parent IDs from the first ``cutoff`` stored ranks.

    Public reference histories intentionally contain empty/non-technique slots
    for paper-metric reproduction.  Candidate-budget overlap, however, should
    count usable ATT&CK candidates rather than those formatting placeholders.
    """
    values: list[str] = []
    seen: set[str] = set()
    for raw in ranked[:cutoff]:
        technique_id = parent_technique_id(str(raw))
        if not VALID_PARENT_TECHNIQUE.fullmatch(technique_id) or technique_id in seen:
            continue
        seen.add(technique_id)
        values.append(technique_id)
    return values


def recall_curve(
    predictions: Mapping[str, Sequence[str]],
    truth: Mapping[str, Set[str]],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    observable_through: int | None = None,
) -> dict[str, Any]:
    """Compute macro/micro recall and hit rate at arbitrary cutoffs.

    ``observable_through`` is a source-level publication limit.  A result above
    that limit is represented by ``None`` instead of treating missing ranks as
    misses.  This is essential for the public TRIAGE Top-20 history.
    """
    cve_ids = sorted(truth)
    total_labels = sum(len(truth[cve_id]) for cve_id in cve_ids)
    denominator_cves = len(cve_ids) or 1
    denominator_labels = total_labels or 1
    points: dict[str, Any] = {}
    for cutoff in cutoffs:
        if observable_through is not None and cutoff > observable_through:
            points[str(cutoff)] = {
                "observable": False,
                "hit_rate": None,
                "macro_recall": None,
                "micro_recall": None,
                "matched_labels": None,
            }
            continue
        hits = 0
        macro_sum = 0.0
        matched = 0
        for cve_id in cve_ids:
            expected = set(truth[cve_id])
            found = len(expected.intersection(predictions.get(cve_id, ())[:cutoff]))
            hits += int(found > 0)
            macro_sum += found / len(expected) if expected else 0.0
            matched += found
        points[str(cutoff)] = {
            "observable": True,
            "hit_rate": hits / denominator_cves,
            "macro_recall": macro_sum / denominator_cves,
            "micro_recall": matched / denominator_labels,
            "matched_labels": matched,
        }
    return {
        "cohort_cves": len(cve_ids),
        "relevant_labels": total_labels,
        "observable_through": observable_through,
        "points": points,
    }


def _truth_by_mapping_type(directory: Path) -> dict[str, dict[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for path in sorted(directory.glob("CVE-*.jsonl")):
        for record in iter_jsonl(path):
            cve_id = str(record["cve_id"])
            for mapping_type, value in (record.get("labels_by_mapping_type") or {}).items():
                techniques = {
                    parent_technique_id(str(raw))
                    for raw in (value.get("techniques", []) if isinstance(value, Mapping) else [])
                }
                if techniques:
                    result[str(mapping_type)][cve_id] = techniques
    return dict(result)


def _label_mapping_types(directory: Path) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = defaultdict(list)
    for mapping_type, typed_truth in _truth_by_mapping_type(directory).items():
        for cve_id, techniques in typed_truth.items():
            for technique_id in techniques:
                result[(cve_id, technique_id)].append(mapping_type)
    return {key: sorted(set(values)) for key, values in result.items()}


def _train_label_frequency(source_dir: Path) -> Counter[str]:
    """Count how many distinct training CVEs contain each parent technique."""
    with (source_dir / "cves_train.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        train_ids = {str(row["CVE ID"]).strip() for row in csv.DictReader(handle)}
    labels_by_cve: dict[str, set[str]] = defaultdict(set)
    with (source_dir / "labeled_cve_to_attack.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            cve_id = str(row["CVE ID"]).strip()
            if cve_id in train_ids:
                labels_by_cve[cve_id].add(parent_technique_id(str(row["attack_id"])))
    return Counter(technique for values in labels_by_cve.values() for technique in values)


def _frequency_bin(count: int) -> str:
    if count == 0:
        return "unseen_0"
    if count <= 2:
        return "rare_1_2"
    if count <= 9:
        return "medium_3_9"
    return "frequent_10_plus"


def _description_length_bin(word_count: int) -> str:
    if word_count <= 25:
        return "short_0_25_words"
    if word_count <= 50:
        return "medium_26_50_words"
    return "long_51_plus_words"


def build_label_rows(
    *,
    truth: Mapping[str, Set[str]],
    mapping_types: Mapping[tuple[str, str], Sequence[str]],
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    train_frequency: Mapping[str, int],
    repository: CVERepository,
) -> list[dict[str, Any]]:
    """Create one auditable row per parent-normalized relevant label occurrence."""
    rows: list[dict[str, Any]] = []
    for cve_id in sorted(truth):
        description = repository.description(cve_id) or ""
        word_count = len(description.split())
        cwes = sorted(set(repository.cwes(cve_id)))
        cwe_group = "no_cwe" if not cwes else ("one_cwe" if len(cwes) == 1 else "multiple_cwes")
        for technique_id in sorted(truth[cve_id]):
            source_ranks: dict[str, int | None] = {}
            for source_name, source_rankings in rankings.items():
                ranked = list(source_rankings.get(cve_id, ()))
                source_ranks[source_name] = ranked.index(technique_id) + 1 if technique_id in ranked else None
            frequency = int(train_frequency.get(technique_id, 0))
            rows.append(
                {
                    "cve_id": cve_id,
                    "year": cve_id.split("-", 2)[1],
                    "technique_id": technique_id,
                    "mapping_types": list(mapping_types.get((cve_id, technique_id), ())),
                    "train_cve_frequency": frequency,
                    "label_frequency_bin": _frequency_bin(frequency),
                    "cwes": cwes,
                    "cwe_group": cwe_group,
                    "description_word_count": word_count,
                    "description_length_bin": _description_length_bin(word_count),
                    "ranks": source_ranks,
                }
            )
    return rows


def rank_distribution(
    label_rows: Sequence[Mapping[str, Any]], source_name: str
) -> dict[str, int]:
    counts = {name: 0 for name, _, _ in RANK_BINS}
    counts["unranked"] = 0
    for row in label_rows:
        rank = row["ranks"].get(source_name)
        if rank is None:
            counts["unranked"] += 1
            continue
        for name, lower, upper in RANK_BINS:
            if rank >= lower and (upper is None or rank <= upper):
                counts[name] += 1
                break
    return counts


def practical_failure_diagnosis(
    label_rows: Sequence[Mapping[str, Any]], source_names: Sequence[str]
) -> dict[str, Any]:
    """Classify labels by their best rank across a specified source family."""
    counts = {"top_20": 0, "rank_21_50": 0, "rank_over_50": 0, "unranked": 0}
    detail: list[dict[str, Any]] = []
    for row in label_rows:
        ranks = [row["ranks"].get(name) for name in source_names]
        observed = [int(rank) for rank in ranks if rank is not None]
        best_rank = min(observed) if observed else None
        if best_rank is None:
            category = "unranked"
        elif best_rank <= 20:
            category = "top_20"
        elif best_rank <= 50:
            category = "rank_21_50"
        else:
            category = "rank_over_50"
        counts[category] += 1
        detail.append(
            {
                "cve_id": row["cve_id"],
                "technique_id": row["technique_id"],
                "best_rank": best_rank,
                "category": category,
            }
        )
    total = len(label_rows) or 1
    return {
        "sources": list(source_names),
        "counts": counts,
        "rates": {name: value / total for name, value in counts.items()},
        "detail": detail,
    }


def _source_hits(
    label_rows: Sequence[Mapping[str, Any]], source_name: str, cutoff: int
) -> set[tuple[str, str]]:
    return {
        (str(row["cve_id"]), str(row["technique_id"]))
        for row in label_rows
        if row["ranks"].get(source_name) is not None
        and int(row["ranks"][source_name]) <= cutoff
    }


def pairwise_diagnostics(
    *,
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    truth: Mapping[str, Set[str]],
    label_rows: Sequence[Mapping[str, Any]],
    source_limits: Mapping[str, int],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    """Measure candidate overlap and correct-label complementarity source by source."""
    result: dict[str, Any] = {}
    source_names = list(rankings)
    all_labels = {(str(row["cve_id"]), str(row["technique_id"])) for row in label_rows}
    for cutoff in cutoffs:
        available = [name for name in source_names if source_limits[name] >= cutoff]
        cutoff_rows: dict[str, Any] = {}
        for index, left in enumerate(available):
            for right in available[index + 1 :]:
                intersections: list[int] = []
                jaccards: list[float] = []
                for cve_id in sorted(truth):
                    left_set = set(_canonical_candidates(rankings[left].get(cve_id, ()), cutoff))
                    right_set = set(_canonical_candidates(rankings[right].get(cve_id, ()), cutoff))
                    intersection = len(left_set.intersection(right_set))
                    union = left_set.union(right_set)
                    intersections.append(intersection)
                    jaccards.append(intersection / len(union) if union else 1.0)
                left_hits = _source_hits(label_rows, left, cutoff)
                right_hits = _source_hits(label_rows, right, cutoff)
                key = f"{left}__vs__{right}"
                cutoff_rows[key] = {
                    "left": left,
                    "right": right,
                    "mean_candidate_intersection": mean(intersections) if intersections else 0.0,
                    "mean_candidate_jaccard": mean(jaccards) if jaccards else 0.0,
                    "correct_both": len(left_hits.intersection(right_hits)),
                    "correct_left_only": len(left_hits - right_hits),
                    "correct_right_only": len(right_hits - left_hits),
                    "correct_neither": len(all_labels - left_hits - right_hits),
                }
        result[str(cutoff)] = cutoff_rows
    return result


def unique_correct_hits(
    label_rows: Sequence[Mapping[str, Any]],
    source_names: Sequence[str],
    *,
    cutoff: int,
) -> dict[str, int]:
    hit_sets = {name: _source_hits(label_rows, name, cutoff) for name in source_names}
    result: dict[str, int] = {}
    for name, hits in hit_sets.items():
        others: set[tuple[str, str]] = set()
        for other_name, other_hits in hit_sets.items():
            if other_name != name:
                others.update(other_hits)
        result[name] = len(hits - others)
    return result


def union_oracle(
    *,
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    truth: Mapping[str, Set[str]],
    source_names: Sequence[str],
    cutoffs: Sequence[int],
    source_limits: Mapping[str, int],
) -> dict[str, Any]:
    """Upper bound from the union of every source's Top-K usable candidates.

    The union can contain substantially more than K candidates.  Its measured
    mean/max size is therefore reported explicitly and it must not be presented
    as a controlled Top-K method.
    """
    total_labels = sum(len(values) for values in truth.values()) or 1
    result: dict[str, Any] = {"sources": list(source_names), "points": {}}
    for cutoff in cutoffs:
        if any(source_limits[name] < cutoff for name in source_names):
            result["points"][str(cutoff)] = {"observable": False}
            continue
        matched = 0
        candidate_counts: list[int] = []
        for cve_id, expected in truth.items():
            union: set[str] = set()
            for name in source_names:
                union.update(_canonical_candidates(rankings[name].get(cve_id, ()), cutoff))
            matched += len(expected.intersection(union))
            candidate_counts.append(len(union))
        result["points"][str(cutoff)] = {
            "observable": True,
            "micro_recall": matched / total_labels,
            "matched_labels": matched,
            "mean_union_candidates": mean(candidate_counts) if candidate_counts else 0.0,
            "max_union_candidates": max(candidate_counts, default=0),
        }
    return result


def _group_memberships(row: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        "mapping_type": list(row.get("mapping_types", ())) or ["unknown"],
        "year": [str(row["year"])],
        "label_frequency": [str(row["label_frequency_bin"])],
        "cwe_count": [str(row["cwe_group"])],
        "cwe_id": [f"CWE-{value}" for value in row.get("cwes", ())] or ["no_cwe"],
        "description_length": [str(row["description_length_bin"])],
    }


def grouped_recall(
    *,
    label_rows: Sequence[Mapping[str, Any]],
    source_names: Sequence[str],
    source_limits: Mapping[str, int],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    """Compute micro recall curves over transparent label-occurrence groups."""
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in label_rows:
        for dimension, values in _group_memberships(row).items():
            for value in values:
                grouped[dimension][value].append(row)

    payload: dict[str, Any] = {}
    for dimension, groups in grouped.items():
        payload[dimension] = {}
        for group_name, rows in sorted(groups.items()):
            source_values: dict[str, Any] = {}
            for source_name in source_names:
                points: dict[str, float | None] = {}
                for cutoff in cutoffs:
                    if source_limits[source_name] < cutoff:
                        points[str(cutoff)] = None
                    else:
                        matched = sum(
                            row["ranks"].get(source_name) is not None
                            and int(row["ranks"][source_name]) <= cutoff
                            for row in rows
                        )
                        points[str(cutoff)] = matched / len(rows) if rows else 0.0
                source_values[source_name] = points
            payload[dimension][group_name] = {
                "label_occurrences": len(rows),
                "cves": len({str(row["cve_id"]) for row in rows}),
                "micro_recall": source_values,
            }
    return payload


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _curve_table(
    lines: list[str], curves: Mapping[str, Mapping[str, Any]], cutoffs: Sequence[int]
) -> None:
    lines.extend(
        [
            "| Source | " + " | ".join(f"Micro R@{cutoff}" for cutoff in cutoffs) + " |",
            "|---|" + "---:|" * len(cutoffs),
        ]
    )
    for name, curve in curves.items():
        values = [
            _pct(curve["points"][str(cutoff)]["micro_recall"])
            for cutoff in cutoffs
        ]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines.append("")


def _group_table(
    lines: list[str], title: str, groups: Mapping[str, Any], source_names: Sequence[str]
) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| Group | Labels | " + " | ".join(f"{name} R@20" for name in source_names) + " |",
            "|---|---:|" + "---:|" * len(source_names),
        ]
    )
    for group_name, item in groups.items():
        values = [
            _pct(item["micro_recall"][name].get("20")) for name in source_names
        ]
        lines.append(
            f"| {group_name} | {item['label_occurrences']} | " + " | ".join(values) + " |"
        )
    lines.append("")


def diagnose_triage_candidates(
    run_dirs: Sequence[Path],
    *,
    output_dir: Path,
    project_root: Path,
    source_dir: Path,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> Path:
    """Run the complete 60-CVE candidate-complementarity diagnostic package."""
    if output_dir.exists():
        raise FileExistsError(f"Diagnostic directory already exists: {output_dir}")
    benchmark_dir = project_root / "data" / "benchmarks" / TRIAGE_ALL_BENCHMARK
    truth = benchmark_truth(benchmark_dir)
    if len(truth) != 60:
        raise RuntimeError(f"Expected exact 60-CVE TRIAGE test cohort, found {len(truth)}")

    print("[diagnose] loading project candidate rankings", flush=True)
    rankings: dict[str, dict[str, list[str]]] = {}
    project_sources: list[str] = []
    for index, raw_path in enumerate(run_dirs, start=1):
        run_dir = raw_path if raw_path.is_absolute() else project_root / raw_path
        name = _run_name(run_dir)
        if name in rankings:
            raise ValueError(f"Duplicate diagnostic source name: {name}")
        rankings[name] = rankings_from_records(candidate_records(run_dir))
        project_sources.append(name)
        stored_lengths = [len(rankings[name].get(cve_id, ())) for cve_id in truth]
        print(
            f"[diagnose] project source {index}/{len(run_dirs)}: {name}; "
            f"coverage={sum(length > 0 for length in stored_lengths)}/60; "
            f"stored_rank_range={min(stored_lengths)}-{max(stored_lengths)}",
            flush=True,
        )
        _require_complete_coverage(rankings[name], truth, source_name=name)

    print("[diagnose] loading and validating public SMET/TRIAGE histories", flush=True)
    reference_dir = source_dir / "reference_predictions"
    reference_sources: list[str] = []
    for name, filename in REFERENCE_FILES[TRIAGE_ALL_BENCHMARK].items():
        rankings[name] = load_reference_history(
            reference_dir / filename,
            expected_truth=truth,
        )
        reference_sources.append(name)

    # Project full-ranking runs and SMET expose their stored depth.  TRIAGE's
    # replication package explicitly publishes only Top-20 output even though
    # some individual response lists terminate earlier.
    source_limits = {
        name: min(len(rankings[name].get(cve_id, ())) for cve_id in truth)
        for name in project_sources
    }
    source_limits["SMET (paper)"] = max(len(values) for values in rankings["SMET (paper)"].values())
    source_limits["TRIAGE (paper)"] = 20

    print("[diagnose] computing recall curves and per-label ranks", flush=True)
    train_frequency = _train_label_frequency(source_dir)
    label_rows = build_label_rows(
        truth=truth,
        mapping_types=_label_mapping_types(benchmark_dir),
        rankings=rankings,
        train_frequency=train_frequency,
        repository=CVERepository(project_root / "data" / "raw" / "cve"),
    )
    if len(label_rows) != 143:
        raise RuntimeError(f"Expected 143 parent-label occurrences, found {len(label_rows)}")
    curves = {
        name: recall_curve(
            source_rankings,
            truth,
            cutoffs=cutoffs,
            observable_through=source_limits[name],
        )
        for name, source_rankings in rankings.items()
    }
    distributions = {name: rank_distribution(label_rows, name) for name in rankings}

    print("[diagnose] computing overlap, unique hits, union oracles and groups", flush=True)
    pairwise = pairwise_diagnostics(
        rankings=rankings,
        truth=truth,
        label_rows=label_rows,
        source_limits=source_limits,
        cutoffs=cutoffs,
    )
    unique_hits = {
        str(cutoff): unique_correct_hits(
            label_rows,
            [name for name in rankings if source_limits[name] >= cutoff],
            cutoff=cutoff,
        )
        for cutoff in cutoffs
    }
    union_families = {
        "project_methods": project_sources,
        "project_plus_smet": [*project_sources, "SMET (paper)"],
        "all_including_supervised_triage": [*project_sources, *reference_sources],
    }
    unique_hits_by_family: dict[str, dict[str, Any]] = {}
    for cutoff in cutoffs:
        unique_hits_by_family[str(cutoff)] = {}
        for family, members in union_families.items():
            if any(source_limits[name] < cutoff for name in members):
                unique_hits_by_family[str(cutoff)][family] = {"observable": False}
            else:
                unique_hits_by_family[str(cutoff)][family] = {
                    "observable": True,
                    "hits": unique_correct_hits(label_rows, members, cutoff=cutoff),
                }
    union_oracles = {
        family: union_oracle(
            rankings=rankings,
            truth=truth,
            source_names=members,
            cutoffs=cutoffs,
            source_limits=source_limits,
        )
        for family, members in union_families.items()
    }
    diagnoses = {
        "project_methods": practical_failure_diagnosis(label_rows, project_sources),
        "project_plus_smet": practical_failure_diagnosis(
            label_rows, [*project_sources, "SMET (paper)"]
        ),
    }
    groups = grouped_recall(
        label_rows=label_rows,
        source_names=list(rankings),
        source_limits=source_limits,
        cutoffs=cutoffs,
    )

    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "label_ranks.jsonl", label_rows)
    _write_jsonl(
        output_dir / "practical_failure_labels.jsonl",
        diagnoses["project_methods"].pop("detail"),
    )
    payload = {
        "cohort": {
            "benchmark": TRIAGE_ALL_BENCHMARK,
            "cves": len(truth),
            "parent_label_occurrences": len(label_rows),
            "cutoffs": list(cutoffs),
        },
        "semantics": {
            "recall_curve": "Raw stored rank semantics, matching the public paper metric at 5/10/20.",
            "candidate_overlap": "Unique valid parent ATT&CK IDs; empty public-history slots do not consume candidate budget.",
            "union_oracle": "Union of every member source's Top-K; mean/max union size is reported and is usually greater than K.",
            "triage_truncation": "TRIAGE public predictions end at Top-20, so Recall@30/50 is unavailable rather than zero.",
        },
        "sources": {
            name: {
                "kind": "project_label_free" if name in project_sources else (
                    "public_label_free_floor" if name == "SMET (paper)" else "public_supervised_upper_bound"
                ),
                "observable_through": source_limits[name],
            }
            for name in rankings
        },
        "curves": curves,
        "rank_distributions": distributions,
        "pairwise": pairwise,
        "unique_correct_hits": unique_hits,
        "unique_correct_hits_by_family": unique_hits_by_family,
        "union_oracles": union_oracles,
        "practical_failure_diagnosis": diagnoses,
        "groups": groups,
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    project_counts = diagnoses["project_methods"]["counts"]
    project_oracle_20 = union_oracles["project_methods"]["points"]["20"]
    plus_smet_20 = union_oracles["project_plus_smet"]["points"]["20"]
    best_project_20 = max(
        curves[name]["points"]["20"]["micro_recall"] for name in project_sources
    )
    lines = [
        "# Stage-1 candidate complementarity diagnosis",
        "",
        "Exact public TRIAGE test view: 60 CVEs and 143 parent-normalized relevant label occurrences.",
        "",
        "## Recall curves",
        "",
    ]
    _curve_table(lines, curves, cutoffs)
    lines.extend(
        [
            "TRIAGE Recall@30/@50 is `N/A`: the public package publishes no ranks beyond 20. SMET and the diagnostic project runs expose longer rankings.",
            "",
            "## Main failure diagnosis",
            "",
            f"Across project methods, {project_counts['top_20']}/143 labels appear in at least one Top-20, "
            f"{project_counts['rank_21_50']}/143 first appear at ranks 21–50, and "
            f"{project_counts['rank_over_50'] + project_counts['unranked']}/143 remain outside every practical Top-50.",
            "",
            f"The project Top-20 union oracle reaches {_pct(project_oracle_20['micro_recall'])} with "
            f"{project_oracle_20['mean_union_candidates']:.1f} candidates per CVE on average; the best single project source is {_pct(best_project_20)}.",
            "",
            f"Adding public SMET output raises the source-wise Top-20 union oracle to {_pct(plus_smet_20['micro_recall'])} "
            f"at {plus_smet_20['mean_union_candidates']:.1f} candidates per CVE on average. This is an upper bound, not a controlled Top-20 result.",
            "",
            "Interpretation rule: many labels at ranks 21–50 indicate a ranking/reranking problem; many labels outside every Top-50 indicate missing practical candidate-source coverage and motivate an additional retrieval view before reranking.",
            "",
            "## Correct-label rank distribution",
            "",
            "| Source | 1 | 2–3 | 4–5 | 6–10 | 11–20 | 21–30 | 31–50 | >50 | Unranked |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in distributions.items():
        lines.append(
            f"| {name} | {item['rank_1']} | {item['rank_2_3']} | {item['rank_4_5']} | "
            f"{item['rank_6_10']} | {item['rank_11_20']} | {item['rank_21_30']} | "
            f"{item['rank_31_50']} | {item['rank_over_50']} | {item['unranked']} |"
        )
    lines.append("")
    project_unique_20 = unique_hits_by_family["20"]["project_methods"]["hits"]
    lines.extend(
        [
            "For TRIAGE, `Unranked` means absent from the published Top-20 history; it does not establish a rank beyond 20.",
            "",
            "## Top-20 complementarity",
            "",
            "Unique correct hits below are labels found by one project source and by none of the other project sources.",
            "",
            "| Project source | Unique correct labels |",
            "|---|---:|",
        ]
    )
    for name in project_sources:
        lines.append(f"| {name} | {project_unique_20[name]} |")
    lines.extend(
        [
            "",
            "| Source pair | Mean candidate Jaccard | Left-only correct | Right-only correct |",
            "|---|---:|---:|---:|",
        ]
    )
    top20_pairs = pairwise["20"]
    complementarity_sources = set([*project_sources, "SMET (paper)"])
    for item in top20_pairs.values():
        if item["left"] not in complementarity_sources or item["right"] not in complementarity_sources:
            continue
        lines.append(
            f"| {item['left']} vs {item['right']} | {item['mean_candidate_jaccard']:.3f} | "
            f"{item['correct_left_only']} | {item['correct_right_only']} |"
        )
    lines.append("")
    _group_table(lines, "Mapping type at Recall@20", groups["mapping_type"], list(rankings))
    _group_table(lines, "Training-label frequency at Recall@20", groups["label_frequency"], list(rankings))
    _group_table(lines, "CWE availability at Recall@20", groups["cwe_count"], list(rankings))
    _group_table(lines, "Description length at Recall@20", groups["description_length"], list(rankings))
    _group_table(lines, "CVE year at Recall@20", groups["year"], list(rankings))
    lines.extend(
        [
            "## Output files",
            "",
            "- `diagnostics.json`: all curves, source limits, overlaps, unique hits, union oracles, and grouped metrics.",
            "- `label_ranks.jsonl`: one row per relevant parent label with all source ranks and grouping features.",
            "- `practical_failure_labels.jsonl`: project-family best-rank category for case inspection.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[diagnose] report written to {output_dir}", flush=True)
    return output_dir
