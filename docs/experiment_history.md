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

## 2026-07-28: frozen-parameter cross-benchmark validation

This validation compares V1 and formal V5c without changing any method parameter between
datasets. Both use ATT&CK Enterprise 15.1 and `basel/ATTACK-BERT`; V5c uses raw CVE text,
strict query-specific leave-one-CVE-out, all three action types, Top-3 action rank-RRF with
`rank_constant=60`, and a fixed Top-20 output. Every run has 100% query and prediction coverage.

The table reports the project's per-CVE Macro Recall. The TRIAGE Micro Recall values reported in
the preceding section (V1 37.76%, V5c 60.14% at Top-20) use pooled labels and must not be mixed
with these Macro values.

| Benchmark | CVEs | V1 R@10 | V5c R@10 | ΔR@10 (95% CI) | V1 R@20 | V5c R@20 | ΔR@20 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRIAGE test all | 60 | 26.40% | 45.49% | +19.09 [11.73, 26.73] pp | 39.43% | 61.98% | +22.55 [14.10, 31.32] pp |
| `cve2attack_result` | 1,661 | 35.44% | 44.83% | +9.40 [7.74, 11.09] pp | 47.83% | 58.53% | +10.70 [8.96, 12.47] pp |
| KEV all | 296 | 22.77% | 45.91% | +23.14 [19.26, 27.17] pp | 35.41% | 60.51% | +25.10 [21.12, 29.05] pp |
| KEV exploitation | 284 | 22.50% | 46.09% | +23.59 [19.72, 27.72] pp | 35.14% | 60.42% | +25.28 [21.34, 29.35] pp |
| KEV nonoverlap | 251 | 24.91% | 46.10% | +21.20 [17.07, 25.39] pp | 36.52% | 61.43% | +24.90 [20.59, 29.24] pp |
| `data_result` hash sample | 2,000 | 61.30% | 77.07% | +15.77 [13.97, 17.57] pp | 73.75% | 85.11% | +11.36 [9.79, 12.96] pp |

Intervals use 10,000 paired CVE-level bootstrap samples with seed `20260728`. Every Recall@10
and Recall@20 interval is entirely above zero. At Top-20, V5c improves/worsens 27/3 TRIAGE CVEs,
422/121 `cve2attack_result` CVEs, 143/10 KEV-all CVEs, 139/10 KEV-exploitation CVEs, 122/9
KEV-nonoverlap CVEs, and 381/69 sampled `data_result` CVEs; all remaining CVEs are unchanged.

The KEV nonoverlap result is especially important: its +24.90-point Top-20 gain remains after
removing every CVE shared with `cve2attack_result`, so the effect is not explained by benchmark
overlap. The `data_result` V1 score is unusually high and the dataset has no currently verifiable
citation; its 2,000-CVE cohort was therefore selected only by a committed seeded SHA-256 ordering
of CVE IDs and is treated as supplementary scale/consistency evidence, not authoritative ground
truth.

Conclusion: frozen V5c is the current primary label-free Stage-1 candidate generator. This closes
the question of whether its improvement is unique to the 60-CVE TRIAGE test slice, but it does not
yet close Stage 1. Procedure-count bias, parent/sub-technique/procedure corpus ablations and
case-level error analysis remain before freezing the paper method and final table.

Runs:

- `runs/multibench_cve2attack_v1_attack15_1_retry/`
- `runs/multibench_cve2attack_v5c_action_attack15_1/`
- `runs/kev_v1_raw_attackbert_15_1/`
- `runs/multibench_kev_all_v5c_action_attack15_1/`
- `runs/multibench_data_sample2000_v1_attack15_1/`
- `runs/multibench_data_sample2000_v5c_action_attack15_1/`

Paired reports:

- `comparisons/multibench_final_triage_v1_vs_v5c_paired/`
- `comparisons/multibench_final_cve2attack_v1_vs_v5c_paired/`
- `comparisons/multibench_final_kev_all_v1_vs_v5c_paired_retry/`
- `comparisons/multibench_final_kev_exploitation_v1_vs_v5c_paired/`
- `comparisons/multibench_final_kev_nonoverlap_v1_vs_v5c_paired/`
- `comparisons/multibench_final_data_sample2000_v1_vs_v5c_paired/`

## 2026-07-28: final V5c corpus, bias and case audit

The final audit keeps the frozen TRIAGE cohort and all retrieval parameters unchanged. The only
ablation variable is which ATT&CK action source types are present. All procedure conditions retain
query-specific leave-one-CVE-out exclusion.

| Corpus | Micro R@5 | Micro R@10 | Micro R@20 | Macro R@20 |
|---|---:|---:|---:|---:|
| V1 parent Technique document | 18.18% | 23.78% | 37.76% | 39.43% |
| V5 parent descriptions only | 11.89% | 18.88% | 34.97% | 36.74% |
| V5 sub-technique descriptions only | 11.19% | 21.68% | 31.47% | 30.67% |
| V5 parent + sub-technique descriptions | 11.89% | 22.38% | 30.77% | 30.43% |
| V5 procedures only, strict LOO | **34.97%** | **46.15%** | **61.54%** | **64.06%** |
| **Formal V5c all actions, strict LOO** | 32.87% | 44.06% | 60.14% | 61.98% |

Procedures provide nearly all of the action-level improvement; splitting parent and sub-technique
descriptions is not sufficient. Procedure-only is slightly stronger on this frozen test, but this
was learned from a post-selection ablation. It therefore remains an explanatory result and does
not replace formal V5c, whose complete corpus was selected before and validated across all other
benchmarks.

Procedure coverage is a material limitation. Across 202 parent Techniques, procedure count has
Spearman rho 0.596 with V5c Top-20 exposure and rho 0.593 with false-positive exposure. Its rho
with per-Technique label Recall@20 is 0.314 among labeled Techniques. These are descriptive rather
than causal estimates, but they show that procedure-rich Techniques are systematically easier to
surface and must be disclosed in the paper.

The 143 true labels split into 37 V5c-only Top-20 gains, 49 retained V1/V5c hits, 5 V1 hits lost by
V5c, 19 unresolved labels at ranks 21-50 and 33 unresolved labels beyond rank 50. Every row records
the exact V1/V5c rank and V5c action evidence.

Artifacts:

- parent descriptions: `runs/triage_final_v5c_parent_description_fullranking/`
- sub-technique descriptions: `runs/triage_final_v5c_subtechnique_description_fullranking/`
- procedures: `runs/triage_final_v5c_procedure_fullranking/`
- final audit: `comparisons/triage_stage1_v5c_final_audit/`

Decision: Stage 1 is frozen and complete. The formal downstream input remains strict-LOO V5c
Top-20. Further V3 prompt tuning, test-set weight search, post-hoc promotion of procedure-only, and
automatic supervised reranking are out of scope unless a new independent benchmark or Stage-2
end-to-end bottleneck analysis justifies reopening Stage 1.
