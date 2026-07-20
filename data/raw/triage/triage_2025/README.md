# TRIAGE public replication subset

This directory contains only the small, evaluation-relevant files selected
from the 773.7 MB public `TRIAGE.zip` archive.  The full archive is deliberately
not copied into the repository.

The selected files preserve:

- the exact 236/60 train/test CVE split;
- all 806 labels and their exploitation/primary/secondary mapping types;
- public SMET predictions with and without secondary-impact labels;
- public TRIAGE predictions with and without secondary-impact labels.

`source.yaml` records the original archive digest, paths and individual file
digests.  Run `import-triage` to build the two normalized benchmark views.  Run
`compare-triage` to validate the reference metrics and compare project runs on
the exact 60-CVE public test split.

These reference predictions are experimental outputs from the paper.  They are
not used as model input or training labels by this project.
