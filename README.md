# CVE → ATT&CK Mapping Pipeline

This repository contains the two-stage research pipeline. Stage 1 generates a
ranked ATT&CK Technique candidate set for each CVE. Stage 2 extracts MulVAL
attack-graph context that will be used to rerank those candidates. The current
integration branch is based on the refactored `new_method` stage-1 approach;
the alternative layered implementation remains on `main` until final integration.

Start with `STAGE2_PLAN.md` for the thesis-scope roadmap, work-package acceptance
criteria, leakage safeguards and Git/worktree rules. The lower-level graph JSON
contract is documented separately in `docs/stage2_graph_context.md`.

V3a was the method selected during the initial refactor, but the frozen TRIAGE comparison showed V1 as the strongest pre-V5 single-source Top-20 baseline. V3a remains an important query-view ablation rather than a preselected final method. The current research roadmap and acceptance gates are maintained in `STAGE1_PLAN.md`.

The frozen label-free Stage-1 method is V5c: it embeds individual ATT&CK descriptions, sub-technique descriptions and procedure actions, then rolls the Top-3 action ranks back to parent Techniques. Its strict leave-one-CVE-out evaluation reaches Micro Recall@20 60.14% on the public 60-CVE TRIAGE test view (V1 37.76%, SMET 52.45%, supervised TRIAGE 76.92%). The same frozen method improves over V1 on `cve2attack_result`, all three KEV views and a label-independent 2,000-CVE `data_result` sample. Corpus ablations show that procedures provide nearly all of the gain; procedure-count exposure bias is documented as a limitation. Stage 1 is frozen and complete for handoff to the graph-context Stage 2.

## Layout

- `STAGE1_PLAN.md`: current Stage-1 roadmap, evidence, work packages and acceptance criteria.
- `experiments/`: versioned method definitions; no generated results.
- `models/ollama/`: versioned Ollama templates and runtime parameters; no model weights.
- `data/benchmarks/`: paper datasets with ground-truth CVE → technique mappings.
- `data/knowledge/`: ATT&CK, CWE and CAPEC knowledge sources.
- `data/raw/`: raw CVE records.
- `data/raw/triage/triage_2025/`: selected public TRIAGE split, labels and reference predictions; the 773.7 MB archive is not stored in Git.
- `data/derived/`: reproducible intermediate data and expensive rewrite caches.
- `data/stage2_sources/`: external scenario packages and source inventories; raw and extracted archives are ignored by Git.
- `data/stage2_scenarios/`: versioned normalized scenario descriptors, generated graph inputs and frozen single-case inputs.
- `src/cve2attack/`: reusable pipeline, strategies and evaluation code.
- `src/cve2attack/stage2/`: MulVAL parsing and versioned graph-context extraction.
- `tests/fixtures/mulval/`: self-contained attack-graph regression input.
- `STAGE2_PLAN.md`: stage-2 roadmap, acceptance checklist and next task.
- `runs/<run_id>/`: one isolated execution, ignored by Git.
- `comparisons/<comparison_id>/`: comparison of multiple runs, ignored by Git.
- `archive/`: inactive TF-IDF and pre-refactor code.

## Install

```bash
.venv/bin/pip install -e .
```

The V3/V4 Ollama tag can be recreated on the remote Ollama service without copying weights into this repository:

```bash
OLLAMA_HOST=http://172.23.216.73:11434 \
  ollama create sec-i1-cve-rewrite:v1 \
  -f models/ollama/sec-i1-cve-rewrite-v1.Modelfile
```

## Run an experiment

```bash
.venv/bin/python -m cve2attack rewrite experiments/v3a_llm_rewrite.yaml --workers 2 --max-cves 20
.venv/bin/python -m cve2attack run experiments/v3a_llm_rewrite.yaml --max-cves 20
```

Run the strict action-level Top-20 method (the first action embedding-cache build is expensive; later runs reuse it):

```bash
.venv/bin/python -m cve2attack inspect experiments/v5c_raw_action_rank_rrf.yaml
.venv/bin/python -m cve2attack run experiments/v5c_raw_action_rank_rrf.yaml
```

`exclude_query_cve_actions: true` is mandatory in formal V5 runs: a procedure that originally names the query CVE is excluded even though CVE/CAN identifiers are also masked in the indexed text. Direct corpus/benchmark overlap can be audited separately:

```bash
.venv/bin/python -m cve2attack audit-action-overlap \
  --benchmark triage_2025_test_all \
  --comparison-id triage_action_procedure_overlap_audit
```

Run the same method on either independent paper benchmark without copying the experiment definition:

```bash
.venv/bin/python -m cve2attack run experiments/v1_raw_attackbert.yaml --benchmark data_result
.venv/bin/python -m cve2attack run experiments/v1_raw_attackbert.yaml --benchmark cve2attack_result
```

Cross-benchmark validation uses the same frozen ATT&CK 15.1 corpus for V1 and V5c:

```bash
.venv/bin/python -m cve2attack run \
  experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml \
  --benchmark ctid_kev_2025_02_13_nonoverlap
```

The committed `data_result_hash_sample_2000` cohort was selected from the 286,461-record
`data_result` benchmark using only a seeded SHA-256 ordering of CVE IDs. It is a reproducibility
and scale check, not a replacement for an independently curated authoritative benchmark.

The final frozen-method audit is reproducible with `diagnose-action-final`. Its ignored runtime
artifacts live under `comparisons/triage_stage1_v5c_final_audit/`; the permanent result summary and
method decision are recorded in `STAGE1_PLAN.md` and `docs/experiment_history.md`.

Rewrite caches use the selected benchmark name and prompt-template version. For example, `--benchmark data_result` with V3a reads or generates `data/derived/rewrite_cache/data_result_sec_i1_llama3_chat_v1.json`. Legacy `*_sec_i1.json` caches were generated with an invalid raw-prompt template and must not be mixed with the v1 cache.

Check paths, query coverage and the Technique knowledge base without loading the embedding model:

```bash
.venv/bin/python -m cve2attack inspect experiments/v3a_llm_rewrite.yaml
```

Each run writes a resolved `manifest.json`, schema-versioned candidates, metrics and a report under a new directory in `runs/`. The default evaluation target is the selected input benchmark; the two paper datasets are never implicitly merged.

## Extract stage-2 graph context

M&NTIS and similar execution packages are not MulVAL XML. Convert their
versioned, normalized scenario descriptor before running context extraction:

```bash
.venv/bin/python -m cve2attack build-stage2-graph \
  --scenario data/stage2_scenarios/mantis/zerologon/scenario.yaml \
  --output data/stage2_scenarios/mantis/zerologon/AttackGraph.xml \
  --force
```

The converter reads only the `context` section when constructing vertices and
edges. `evaluation.expected_techniques` remains separate and is used only by
the benchmark loader after reranking.

```bash
.venv/bin/python -m cve2attack extract-graph-context \
  --attack-graph tests/fixtures/mulval/AttackGraph.xml \
  --output stage2_runs/example/contexts.json \
  --max-graph-depth 2
```

The graph producer remains external. Only `AttackGraph.xml` crosses the project
boundary, so the package does not depend on the old parent directory or on a
specific MulVAL installation. See `docs/stage2_graph_context.md` for the JSON
contract and module responsibilities.

Run the first stage-1 -> graph context -> deterministic reranking smoke loop:

```bash
PYTHONPATH=src ../cve2attack/.venv/bin/python -m cve2attack run-stage2 \
  --stage1-run /home/ghdemi/Code/cve2attack/runs/triage_rrf_v1_v3a_d50_k60_top20 \
  --attack-graph tests/fixtures/stage2/public_facing/AttackGraph.xml \
  --benchmark triage_2025_test_all \
  --run-id cve_2023_20887_public_facing_smoke \
  --scenario-kind synthetic_public_facing_smoke
```

This fixture uses a real CVE, real stage-1 Top-20 candidates and public TRIAGE
labels, but its network topology is synthetic. It validates the engineering
loop and must not be presented as independent aggregate accuracy evidence.
Each run writes contexts, joined records, reranked records, metrics, a manifest
and a readable report under `stage2_runs/<run_id>/`.

Run the trace-derived Zerologon case from committed inputs:

```bash
.venv/bin/python -m cve2attack run-stage2 \
  --stage1-run data/stage2_scenarios/mantis/zerologon/stage1_snapshot \
  --attack-graph data/stage2_scenarios/mantis/zerologon/AttackGraph.xml \
  --benchmark stage2_mantis_scenarios \
  --run-id mantis_zerologon_v1 \
  --scenario-kind trace_derived_mantis_lateral_movement
```

## Compare runs

```bash
.venv/bin/python -m cve2attack compare \
  --benchmark cve2attack_result \
  runs/<run-a> runs/<run-b>
```

Every method is evaluated on the benchmark's complete fixed cohort. Missing predictions count as misses and coverage is reported separately.
When exactly two runs are compared, the report additionally includes their paired per-CVE
Recall@10/@20 deltas, 10,000-sample bootstrap 95% confidence intervals, and counts of improved,
unchanged and worse CVEs.

Import and compare on the exact 60-CVE public TRIAGE test split:

```bash
.venv/bin/python -m cve2attack import-triage

.venv/bin/python -m cve2attack compare-triage \
  --comparison-id kev_methods_vs_triage \
  runs/kev_v1_raw_attackbert_15_1 \
  runs/kev_v2_raw_procedures_15_1 \
  runs/kev_v3a_llm_rewrite_15_1 \
  runs/kev_v3b_llm_rewrite_procedures_15_1
```

The comparison reports both per-CVE macro Recall@K and TRIAGE's pooled-label
micro Recall@K. The names stay explicit because the two values are not
interchangeable.

## Candidate schema

```json
{
  "schema_version": "1.0",
  "cve_id": "CVE-2022-0014",
  "domain": "Enterprise",
  "candidates": [
    {
      "technique_id": "T1574",
      "score": 0.6575,
      "sources": ["embedding"]
    }
  ]
}
```

Legacy records using `techniques: ["T..."]` or `techniques: [{"id": ..., "score": ...}]` are accepted by the shared reader, but all new runs use the schema above.
