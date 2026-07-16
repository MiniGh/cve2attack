# `new_method` experiment history

The figures below are historical results preserved from the original branch. They must be regenerated with the unified schema and fixed benchmark cohort before being used as final paper results.

| Method | Query | Technique document | Fusion | Historical result |
|---|---|---|---|---|
| V1 | raw CVE description | name + description | none | R@10 33.31%, R@20 45.59% (1534 CVEs) |
| V2 | raw CVE description | name + description + procedures | none | same as V1 |
| V3a (selected) | LLM rewrite | name + description | none | R@10 38.95%, R@20 52.90% (1660 CVEs) |
| V3b | LLM rewrite | name + description + procedures | none | R@10 39.49%, R@20 52.87% |
| V4 | LLM rewrite | name + description + procedures | structured-chain fusion | R@10 39.61%, R@20 52.87% |

The alternative layered-LLM method on `main` is another first-stage method and is outside this refactor.
