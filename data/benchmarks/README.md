# Benchmarks

Each child directory is an independent paper dataset with its own annotation policy. Evaluators must report them separately. Combining labels from multiple benchmarks requires an explicit, versioned dataset definition; no implicit union is allowed.

## CTID KEV benchmark views

The three `ctid_kev_2025_02_13_*` directories are generated from one fixed,
public CTID Known Exploited Vulnerabilities (KEV) snapshot:

- Source landing page: <https://zenodo.org/records/16747173>
- Raw file: `data/raw/kev/kev-02.13.2025_attack-15.1-enterprise.csv`
- CTID snapshot: `02.13.2025`; ATT&CK Enterprise version: `15.1`
- Raw MD5: `21338a56761278482dc5f169638414ca`
- Retrieval corpus: `data/knowledge/enterprise-attack-15.1.json`, from the
  official MITRE ATT&CK STIX 15.1 Release
- Retrieval corpus SHA-256:
  `a57988bffe402bb3e19d92dbe80a12143e1970b814e013e080f9df2fa5a3f6bc`

Generate the views once after obtaining the raw CSV:

```bash
.venv/bin/python -m cve2attack import-kev
```

The command refuses to overwrite an existing directory, so a benchmark cannot
be silently changed in place.  Every generated `dataset.yaml` records the raw
file's MD5 and SHA-256 checksums.

| Benchmark | Gold labels | Intended use |
|---|---|---|
| `ctid_kev_2025_02_13_all` | Union of `exploitation_technique`, `primary_impact`, and `secondary_impact` | Primary KEV result for context-aware stage 1 |
| `ctid_kev_2025_02_13_exploitation` | Only `exploitation_technique` labels | Exploit-action diagnostic |
| `ctid_kev_2025_02_13_nonoverlap` | All mapping types after removing CVEs in `cve2attack_result` | Stricter external result |

The active pipeline uses the parent-normalized `techniques` field, because it
retrieves top-level ATT&CK techniques.  The source-exact IDs, including
sub-techniques, remain in `techniques_raw`; `labels_by_mapping_type` retains
the CTID semantic role and `label_metadata` retains the evidence references.

Every KEV `dataset.yaml` declares its frozen 15.1 technique corpus.  `inspect`
and `run` select that corpus automatically, creating a separate embedding cache
from the current `enterprise-attack.json` used by the other benchmarks.  An
experiment can set `technique_document.attack_bundle` only for an explicit
ATT&CK-version migration experiment.

For a fair candidate-generation evaluation, models must use only the project
raw CVE description.  KEV comments and references are label provenance, not
model input.
