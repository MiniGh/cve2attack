# Derived data

This directory contains rebuildable intermediate artifacts used by the `new_method` pipeline:

- `domain_mapping/`: rule-based CVE → ATT&CK domain classifications.
- `structured_chain/`: historical CWE → CAPEC → ATT&CK chain data used only by V4.
- `rewrite_cache/`: expensive LLM rewrites, keyed by benchmark/model.
- `embedding_cache/`: local model caches ignored by Git.
