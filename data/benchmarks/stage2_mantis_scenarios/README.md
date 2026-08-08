# M&NTIS stage-2 case-study labels

This small benchmark contains only the labels attached to the trace-derived
M&NTIS case studies under `data/stage2_scenarios/mantis/`. It is intentionally
kept separate from the KEV, TRIAGE, CVE2ATT&CK and `data_result` benchmarks.

It is a fixed three-scenario handoff cohort between Stage 1 and Stage 2, not a
population-level benchmark. Sub-techniques are rolled up to parent Technique IDs
by the standard benchmark loader. Structured metadata, including the frozen
ATT&CK corpus hash and the source dataset identifiers, lives in `dataset.yaml`.

## Stage 1 obligations (candidate generation)

Stage 1 uses only the CVE IDs to select records and only the original
descriptions in `data/raw/cve/` to construct queries. The M&NTIS Technique
labels are evaluation-only and their ranks are inspected only after the
candidate files have been written.

The formal generation condition is V5c with ATT&CK Enterprise 15.1, strict
query-specific leave-one-CVE-out filtering, and parent-Technique Top-20. Use
`experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml`; do not use the
root V5c config, whose default benchmark can fall back to ATT&CK 18.1.

## Stage 2 obligations (graph context reranking)

The labels are used only after reranking. They must never be read while a
scenario is converted into `AttackGraph.xml`, nor while topology rules run. The
graph renderer copies only the scenario `context` section and never its
`evaluation` section, so a label cannot become a graph fact.
