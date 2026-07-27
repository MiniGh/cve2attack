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

## 2026-07-23: fixed-parameter RRF baseline on the TRIAGE public test split

This experiment uses the exact 60-CVE / 143-parent-label `triage_2025_test_all` view.  No labels were used to choose weights or tune the RRF constant: every source has weight 1, the standard fixed `rank_constant=60` is used, and the final candidate budget is always Top-20.  Source depths 20 and 50 are reported as a diagnostic of whether ranks 21-50 can be recovered.

| Method | Source depth | Micro R@5 | Micro R@10 | Micro R@20 |
|---|---:|---:|---:|---:|
| V1 | full ranking | 18.18% | 23.78% | 37.76% |
| V3a | full ranking | 13.99% | 20.28% | 35.66% |
| V3b | full ranking | 15.38% | 20.98% | 34.97% |
| RRF V1 + V3a | 20 | 15.38% | 20.98% | 38.46% |
| RRF V1 + V3a | 50 | 14.69% | 20.28% | **39.86%** |
| RRF V1 + V3b | 20 | 14.69% | 21.68% | 37.06% |
| RRF V1 + V3b | 50 | 14.69% | 21.68% | 39.16% |
| RRF V1 + V2 + V3a + V3b | 20 | 15.38% | 21.68% | 37.76% |
| RRF V1 + V2 + V3a + V3b | 50 | 13.99% | 21.68% | **39.86%** |
| SMET public prediction | public history | 23.78% | 37.76% | 52.45% |

The simplest selected RRF baseline is `V1 + V3a`, source depth 50.  It raises Micro Recall@20 from V1's 37.76% (54/143 labels) to 39.86% (57/143), while Micro Recall@5 and @10 decline.  Relative to V1 Top-20 it gains nine labels and loses six: RRF promotes candidates that have moderate ranks in both sources, but can demote a correct V1-only candidate when the rewrite source ranks it poorly.

At Recall@20, the selected baseline changes mapping-type coverage as follows:

| Mapping type | V1 | RRF V1 + V3a depth 50 |
|---|---:|---:|
| exploitation technique | 37.70% | 42.62% |
| primary impact | 50.00% | 48.15% |
| secondary impact | 13.79% | 17.24% |

The four-source depth-50 variant reaches the same overall 39.86%, with exploitation/primary/secondary Recall@20 of 40.98%/48.15%/20.69%.  Because V1/V2 and V3a/V3b are highly redundant, the extra sources do not improve the overall score over the two-source baseline.

Interpretation: RRF is retained as the label-free controlled-budget fusion baseline, but it realizes only 3 of the 15 additional label hits available in the project Top-20 union oracle (48.25%).  It therefore does not remove the need for a genuinely different action-level retrieval source.  The next method should first raise practical candidate coverage; RRF can then fuse that new source into a final Top-20.

Artifacts:

- selected run: `runs/triage_rrf_v1_v3a_d50_k60_top20/`
- four-source ablation: `runs/triage_rrf_all4_d50_k60_top20/`
- unified comparison: `comparisons/triage_rrf_baseline_k60_equal_weights/`

## 2026-07-27: action-level ATT&CK retrieval (V5)

V5 indexes every active parent/sub-technique description and every ATT&CK `uses` relationship as an independent action. Sub-techniques are rolled up to their parent Technique only after retrieval. The ATT&CK 15.1 corpus contains 14,121 actions: 13,484 procedures, 435 sub-technique descriptions and 202 parent descriptions over 202 parent Techniques.

All exact CVE/CAN identifiers are replaced by `[VULNERABILITY]`. Formal results additionally use query-specific leave-one-CVE-out (LOO): every action whose original procedure named the query CVE is excluded before Technique aggregation. The public-overlap audit found only 8/60 test CVEs in procedure text and 5/143 direct true-label pairs (3.50%). LOO therefore prevents a real shortcut, but it changes the overall result by less than one percentage point.

| Method | Query | Action corpus | Aggregation | Micro R@5 | Micro R@10 | Micro R@20 |
|---|---|---|---|---:|---:|---:|
| V5e | raw | descriptions only | max similarity | 16.08% | 24.48% | 40.56% |
| V5i | raw | descriptions + procedures, strict LOO | max similarity | 31.47% | 39.86% | 53.85% |
| **V5k / formal V5c** | **raw** | **descriptions + procedures, strict LOO** | **Top-3 action rank-RRF** | **32.87%** | **44.06%** | **60.14%** |
| V5l | rewrite | descriptions + procedures, strict LOO | Top-3 action rank-RRF | 31.47% | 39.86% | 59.44% |
| SMET public prediction | public history | — | — | 23.78% | 37.76% | 52.45% |
| TRIAGE public prediction | supervised public history | — | — | 61.54% | 69.93% | 76.92% |

V5k contributes 21 correct Top-20 labels missed by all five other project sources. The project union oracle rises to 67.83%, but requires 39 candidates per CVE on average and is not a controlled Top-20 result. V5k performs best on frequent training labels (70.91% R@20), while medium-frequency labels remain weak (17.65%); procedure-count/frequency bias must therefore be diagnosed before making a final paper claim.

The predeclared equal-weight fusion baseline did not convert the union into a better controlled result:

| Controlled Top-20 output | Micro R@10 | Micro R@20 |
|---|---:|---:|
| V5k action source alone | **44.06%** | **60.14%** |
| RRF V1 + V5k, depth 50, k=60 | 34.27% | 48.25% |
| RRF V1 + V5e + V5k, depth 50, k=60 | 34.97% | 46.15% |

Simple source-level RRF dilutes an already stronger action ranking, so it is retained as a negative result rather than selected. No weights or aggregation parameters were searched on the frozen 60-CVE test split.

Artifacts:

- strict comparison: `comparisons/triage_action_v5_leave_one_cve_out/`
- complementarity: `comparisons/triage_candidate_complementarity_actions_loo_final/`
- controlled fusion: `comparisons/triage_action_v5_controlled_rrf_fusion/`
- overlap audit: `comparisons/triage_action_procedure_overlap_audit/`
