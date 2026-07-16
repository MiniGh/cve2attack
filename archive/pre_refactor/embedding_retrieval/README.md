# Embedding Retrieval Prototype

This folder contains a phase-1 embedding-only retrieval pipeline for CVE -> ATT&CK candidate generation.

## Script

- `run_embedding_retrieval.py`

## What it does

- Loads Enterprise-only CVE IDs from `cve_to_attack_domain/result/CVE-*.jsonl`.
- Joins CVE descriptions from `og_data/cve/CVE-{year}.json`.
- Extracts top-level (non-sub-technique) ATT&CK techniques from `og_data/enterprise-attack.json`.
- Builds technique docs using `name + description + relationship uses descriptions`.
- Embeds techniques with `BAAI/bge-m3` via SiliconFlow API.
- Caches technique embeddings to `output/retrieval/tech_embeddings_cache.npz`.
- Embeds each CVE description and retrieves top-k technique IDs by cosine similarity.
- Writes yearly JSONL files under `output/retrieval/` such as `CVE-1999.jsonl` and `CVE-2024.jsonl`.
- Writes `output/retrieval/inspect_sample.md`.

Yearly JSONL output format:

```json
{
  "cve_id": "CVE-2016-1499",
  "techniques": ["T1190", "T1083"]
}
```

- Output records only keep `techniques`.
- `inspect_sample.md` lists the top techniques without scores.

## Dependencies

Install in project venv:

```bash
.venv/bin/pip install openai numpy
```

## Run

```bash
export SILICONFLOW_API_KEY="<your_key>"
.venv/bin/python embedding_retrieval/run_embedding_retrieval.py
```

Quick smoke run (small subset):

```bash
export SILICONFLOW_API_KEY="<your_key>"
.venv/bin/python embedding_retrieval/run_embedding_retrieval.py --max-cves 20
```

## Notes

- Enterprise only.
- Parent techniques only (sub-techniques are filtered out).
- No structured chain, no LLM reranking.
- Output contains only technique IDs, grouped by year.
