# CVE → ATT&CK Mapping Pipeline

This repository contains the two-stage research pipeline. Stage 1 generates a
ranked ATT&CK Technique candidate set for each CVE. Stage 2 extracts MulVAL
attack-graph context that will be used to rerank those candidates. The current
integration branch is based on the refactored `new_method` stage-1 approach;
the alternative layered implementation remains on `main` until final integration.

Start with `STAGE2_PLAN.md` for the thesis-scope roadmap, work-package acceptance
criteria, leakage safeguards and Git/worktree rules. The lower-level graph JSON
contract is documented separately in `docs/stage2_graph_context.md`.

The selected method is **V3a**: use the corrected Llama 3-templated Ollama tag `sec-i1-cve-rewrite:v1` to rewrite a CVE description into attacker-action language, then retrieve top-level ATT&CK techniques with `basel/ATTACK-BERT` using technique name + description.

## Layout

- `experiments/`: versioned method definitions; no generated results.
- `models/ollama/`: versioned Ollama templates and runtime parameters; no model weights.
- `data/benchmarks/`: paper datasets with ground-truth CVE → technique mappings.
- `data/knowledge/`: ATT&CK, CWE and CAPEC knowledge sources.
- `data/raw/`: raw CVE records.
- `data/raw/triage/triage_2025/`: selected public TRIAGE split, labels and reference predictions; the 773.7 MB archive is not stored in Git.
- `data/derived/`: reproducible intermediate data and expensive rewrite caches.
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

Run the same method on either independent paper benchmark without copying the experiment definition:

```bash
.venv/bin/python -m cve2attack run experiments/v1_raw_attackbert.yaml --benchmark data_result
.venv/bin/python -m cve2attack run experiments/v1_raw_attackbert.yaml --benchmark cve2attack_result
```

Rewrite caches use the selected benchmark name and prompt-template version. For example, `--benchmark data_result` with V3a reads or generates `data/derived/rewrite_cache/data_result_sec_i1_llama3_chat_v1.json`. Legacy `*_sec_i1.json` caches were generated with an invalid raw-prompt template and must not be mixed with the v1 cache.

Check paths, query coverage and the Technique knowledge base without loading the embedding model:

```bash
.venv/bin/python -m cve2attack inspect experiments/v3a_llm_rewrite.yaml
```

Each run writes a resolved `manifest.json`, schema-versioned candidates, metrics and a report under a new directory in `runs/`. The default evaluation target is the selected input benchmark; the two paper datasets are never implicitly merged.

## Extract stage-2 graph context

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

## Compare runs

```bash
.venv/bin/python -m cve2attack compare \
  --benchmark cve2attack_result \
  runs/<run-a> runs/<run-b>
```

Every method is evaluated on the benchmark's complete fixed cohort. Missing predictions count as misses and coverage is reported separately.

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
