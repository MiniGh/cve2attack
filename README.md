# CVE → ATT&CK Stage-1 Candidate Generation (`new_method`)

This branch contains the embedding-retrieval approach for the first mapping stage: generate a ranked ATT&CK technique candidate set for each CVE. It is independent from the alternative layered-LLM implementation on `main`.

The selected method is **V3a**: use the corrected Llama 3-templated Ollama tag `sec-i1-cve-rewrite:v1` to rewrite a CVE description into attacker-action language, then retrieve top-level ATT&CK techniques with `basel/ATTACK-BERT` using technique name + description.

## Layout

- `experiments/`: versioned method definitions; no generated results.
- `models/ollama/`: versioned Ollama templates and runtime parameters; no model weights.
- `data/benchmarks/`: paper datasets with ground-truth CVE → technique mappings.
- `data/knowledge/`: ATT&CK, CWE and CAPEC knowledge sources.
- `data/raw/`: raw CVE records.
- `data/derived/`: reproducible intermediate data and expensive rewrite caches.
- `src/cve2attack/`: reusable pipeline, strategies and evaluation code.
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

## Compare runs

```bash
.venv/bin/python -m cve2attack compare \
  --benchmark cve2attack_result \
  runs/<run-a> runs/<run-b>
```

Every method is evaluated on the benchmark's complete fixed cohort. Missing predictions count as misses and coverage is reported separately.

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
