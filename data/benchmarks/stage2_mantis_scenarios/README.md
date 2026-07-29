# M&NTIS Stage 2 case-study labels

This three-CVE cohort mirrors the frozen case-study labels in the Stage 2
worktree at `data/benchmarks/stage2_mantis_scenarios/`. It is intentionally
separate from KEV, TRIAGE, CVE2ATT&CK, and `data_result` benchmarks.

Stage 1 candidate generation uses only the CVE IDs to select records and only
the original descriptions in `data/raw/cve/` to construct queries. The M&NTIS
Technique labels are evaluation-only and are inspected for ranks only after the
candidate files have been written. Sub-techniques are rolled up to parent
Technique IDs by the standard benchmark loader.

This is a three-scenario handoff baseline for Stage 2, not a population-level
benchmark. The formal generation condition is V5c with ATT&CK Enterprise 15.1,
strict query-specific leave-one-CVE-out filtering, and parent-Technique Top-20.
Use `experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml`; do not use
the root V5c config whose default benchmark can fall back to ATT&CK 18.1.
