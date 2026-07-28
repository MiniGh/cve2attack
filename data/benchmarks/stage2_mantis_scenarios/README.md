# M&NTIS stage-2 scenario labels

This small benchmark contains only labels attached to the trace-derived M&NTIS
case studies under `data/stage2_scenarios/mantis/`. It is intentionally kept
separate from KEV, TRIAGE and CVE2ATT&CK benchmarks.

The labels are used only after reranking. They must never be read while a
scenario is converted into `AttackGraph.xml` or while topology rules run.
Sub-techniques are rolled up to parent Technique IDs by the standard benchmark
loader. This is a case-study evaluation set, not a population-level benchmark.
