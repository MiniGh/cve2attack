# Retrieval Method Comparison on CVE2ATT&CK Evaluation Set

Ground truth: `cve2attack_result/` (1661 CVEs across 2008-2022).

| Method | CVEs | Recall@10 | Recall@20 |
|---|---|---|---|
| V1: baseline (raw desc) | 1534 | 33.31% | 45.59% |
| V2: +procedures | 1534 | 33.31% | 45.59% |
| V3: +LLM rewrite | 1660 | 38.95% | 52.90% |

> **Recall@K**: average per-CVE technique recall, i.e. `|pred &cap; truth| / |truth|` averaged over all evaluated CVEs.

> V2 extraction from git commit `e295530`.