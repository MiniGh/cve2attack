# Extended local privilege escalation stage-2 labels

This benchmark holds only the pre-registered evaluation labels for the extended
local privilege escalation cohort under `data/stage2_scenarios/extended_lpe/`
plus the existing PwnKit bridging control. It is deliberately separate from the
pooled `ctid_kev_2025_02_13_*` benchmarks.

## Why a dedicated label set

The pooled CTID benchmarks merge every mapping type into one truth set per CVE.
Under that pooling CVE-2021-40449 carries eight techniques (including T1566
Phishing, T1071 Web Protocols and T1082 System Information Discovery) and
CVE-2022-21999 carries five, while CVE-2020-0787 carries two. Evaluating the
local privilege escalation research question against those pooled sets would
(a) let a hit on an unrelated technique such as T1566 count as success for a
privilege escalation graph, and (b) give each case a different number of correct
answers, making the per-case comparison incoherent.

`docs/stage2_extended_lpe_evaluation_protocol.md` section 2 froze one label per
case: T1068. This directory encodes exactly that pre-registration, recording for
each case the CTID mapping type and CSV row the label came from.

## Case roles

- `main`: CVE-2020-0787, CVE-2021-40449, CVE-2022-21999, CVE-2022-26904.
  These four form the new main aggregate.
- `bridging_control`: CVE-2021-4034 (PwnKit), already evaluated earlier and kept
  only to connect the new cohort to existing results.
- `diagnostic_only`: CVE-2010-3856, which has no independent ATT&CK mapping. Its
  T1068 annotation is authored inside the same AttackMate trace and therefore
  cannot serve as a gold label. Reported separately, never in the main aggregate.

## Leakage boundary

Labels are read only after reranking, by the benchmark loader. They never enter
scenario conversion, graph facts, topology rule detection or candidate scores.
Sub-techniques would be rolled up to parent Technique IDs by the standard
loader; every label here is already a parent technique.

This is a case-study evaluation set, not a population-level benchmark.
