# Stage-2 external scenario sources

This directory records public scenario sources considered for evaluating the
attack-graph context stage. It deliberately separates reproducible third-party
downloads from project-owned converters, manifests, and reports.

## Directory policy

- `source_inventory.yaml` is the authoritative provenance and suitability
  record. Update it whenever a source version or assessment changes.
- Third-party repositories, archives, extracted files, PCAPs, and logs are
  ignored by Git. They remain under the corresponding source directory on the
  experiment machine.
- A source is not an evaluation benchmark merely because it contains an attack
  or an ATT&CK identifier. A full-pipeline case must identify a CVE, preserve an
  external ATT&CK label, and provide enough host/reachability evidence to build
  or convert an attack graph.
- Synthetic or manually completed topology must be marked as controlled data
  and reported separately from externally authored scenarios.

## Current layout

```text
data/stage2_sources/
├── README.md
├── source_inventory.yaml
├── attackmate/
│   ├── raw/          # Zenodo ZIP files when network access is available
│   ├── extracted/    # extracted Zenodo files
│   └── repository/   # pinned shallow clone of the official GitHub repository
└── mantis/
    └── downloads/    # user-authorized M&NTIS exports after account login
```

## First inventory result

The pinned AttackMate repository contains 13 YAML examples and 35
ATT&CK-annotated steps, covering 30 distinct technique or sub-technique IDs.
Four examples can be linked to real CVEs through an explicit CVE reference or
the named official Metasploit module:

| CVE | AttackMate example | AttackMate labels | Existing `data_result` labels |
| --- | --- | --- | --- |
| CVE-2012-1823 | `webserv.yml` | T1190, T1059.004 | T1059 |
| CVE-2010-3856 | `http-put_example.yml` | T1068 | T1068, T1548, T1574 |
| CVE-2010-2075 | `upgrade_to_meterpreter.yml` | T1190, T1059.004, T1195.002 | T1059 |
| CVE-2011-2523 | `include.yml` | T1190, T1195.002, T1059.004 | T1205 |

This is enough to seed controlled conversion and disagreement analysis, but not
enough for a final external benchmark: the examples target one Metasploitable2
host, contain no MulVAL graph, do not provide a vulnerability-driven lateral
movement case, and are absent from the current 60-CVE stage-1 RRF run.

M&NTIS is closer to the required external format because its documented export
contains topology, assets, an ATT&CK-labelled attack report, and attack graph.
The data portal requires account registration, so its CVE coverage remains
unverified until a smallest available dataset is downloaded by an authenticated
user and placed under `mantis/downloads/`.

## Claim boundary

AttackMate playbook metadata is an externally authored annotation, not an
authoritative universal CVE label. Differences from `data_result` must be
reported as annotation disagreement rather than silently merged into a new
ground truth.
