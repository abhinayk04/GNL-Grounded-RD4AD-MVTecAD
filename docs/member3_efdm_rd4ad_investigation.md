# Member 3 — EFDM and RD4AD Investigation

Generated UTC: 2026-11-19T05:44:17.528035+00:00

## EFDM grid outlier

The standard-versus-EFDM comparison reports a large grid-category change.
The grid metrics and metadata were inspected from the existing evaluation
JSON, aligned summary, integrity audit, and category result directory.

Do not treat the grid improvement as validated until its evaluation
protocol and metric provenance are confirmed against the original maps
and masks. This investigation does not recalculate AUROC from maps.

## RD4AD source recovery

- Likely source files found: 4
- Archive files with matching members: 0
- Notebooks with matching source cells: 2

## Checkpoint inspection

- Checkpoint files inspected: 15
- Inspection was limited to serialization format and metadata.
- No tensor payloads were loaded and no checkpoint was modified.

A checkpoint's presence and size alone do not prove its architecture.
RD4AD must not be claimed as validated until its source architecture,
checkpoint keys, and evaluation script are matched.

## Search evidence

- Source candidate: `/kaggle/working/ADShift-GNL-working/__pycache__/de_resnet.cpython-313.pyc` (18,136 bytes)
- Source candidate: `/kaggle/working/ADShift-GNL-working/de_resnet.py` (16,803 bytes)
- Source candidate: `/kaggle/working/project_clean/experiments/gnl/__pycache__/de_resnet.cpython-313.pyc` (18,146 bytes)
- Source candidate: `/kaggle/working/project_clean/experiments/gnl/de_resnet.py` (16,410 bytes)
- Notebook candidate: `/kaggle/working/GNL_Project_Backup/GNL_Baseline_recovery.ipynb`
- Notebook candidate: `/kaggle/working/EFDM_COMPLETE_EVIDENCE/implementation_candidates/GNL_Baseline_recovery.ipynb`