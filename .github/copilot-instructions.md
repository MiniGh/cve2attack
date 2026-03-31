# Project Guidelines

## Code Style
- Target Python 3.10+ and keep type hints plus concise docstrings, matching existing modules.
- Keep imports and path handling consistent with current scripts: compute PROJECT_ROOT with Path(__file__).resolve().parents[1] where needed.
- Preserve JSON output conventions: UTF-8, pretty JSON for files intended for inspection, JSONL for record-wise evaluation outputs.

## Architecture
- Pipeline is split into two stages:
  - Stage 1 domain classification: scripts/run_classifier.py writes CVE -> domain mapping to cve_to_attack_domain/result/cve_domain_mapping.json.
  - Stage 2 tactic mapping: stage2_cve_tactics/run_mapping.py reads Stage 1 mapping and predicts ATT&CK tactics per CVE year file.
- Ground truth and evaluation utilities live under Validate_data/.
- Source datasets are under og_data/ and include large yearly CVE JSON files plus ATT&CK JSON sources.

## Build and Test
- Use repo root as working directory.
- Run Stage 1 mapping:
  - python scripts/run_classifier.py
- Run Stage 2 mapping (example):
  - python stage2_cve_tactics/run_mapping.py --model qwen3:32b --start-year 2021 --end-year 2021 --workers 4
- Run Stage 2 evaluation:
  - python Validate_data/evaluate_stage2_tactics.py --pred-dir stage2_cve_tactics/result --output Validate_data/stage2_tactics_eval.jsonl
- Build full-chain GT tactics labels (when needed):
  - python Validate_data/build_cve2tactics_from_full_chain.py

## Conventions
- Domain labels are canonicalized to Enterprise, ICS, Mobile (stage-2 normalization uses lowercase internally).
- Tactic IDs must be uppercase ATT&CK tactic IDs in TA#### format.
- Prefer minimal, local changes; do not refactor unrelated pipeline stages in one change.
- Keep robust fallbacks for optional dependencies (for example, tqdm fallback wrappers) unless explicitly removing them.

## Pitfalls
- Stage 2 requires cve_to_attack_domain/result/cve_domain_mapping.json; run Stage 1 first if missing.
- stage2_cve_tactics/llm_client.py defaults to a specific local network endpoint; confirm it is reachable before running large batches.
- Running across full CVE ranges is expensive; prefer --start-year/--end-year and optional --limit while iterating.
