# Member 3 — Final Research Handoff

Generated: 2026-11-19T05:40:45.834996+00:00

## 1. Repository state

- Branch: `member3/pca-method`
- HEAD at report generation: `c2f1ffd`
- Working directory: `/kaggle/working/project_clean`
- The final report and evidence index are generated from files present in this workspace.
- GitHub push status must be checked separately; the previous push attempt failed because GitHub credentials were unavailable in Kaggle.

## 2. PCA nuisance-subspace filtering

The repository contains the PCA implementation under `pca_method/`.

- PCA basis files discovered (`.pth`): **45**
- Category comparison rows: **15**
- PCA numeric column means:
- Mean `baseline_pixel_auroc`: 0.966469
- Mean `pca_pixel_auroc`: 0.964577
- Mean `pixel_delta`: -0.001893
- Mean `baseline_image_auroc`: 0.959216
- Mean `pca_image_auroc`: 0.961515
- Mean `image_delta`: 0.002299

### PCA plot status

- Available: `results/pca/image_auroc_improvement.png`
- Available: `results/pca/macro_auroc_comparison.png`
- Available: `results/pca/pixel_auroc_improvement.png`

The three plots were restored from their committed Git blobs and verified by byte-for-byte hash comparison in the preceding recovery task.

## 3. GNL baseline

GNL result CSV inventory and numeric summaries:

#### `results/gnl/GNL_all_15_categories.csv`
- Rows: 15
- Mean `checkpoint_epoch`: 18.933333
- Mean `test_images`: 115.000000
- Mean `normal_images`: 31.133333
- Mean `anomalous_images`: 83.866667
- Mean `image_auroc`: 0.959216
- Mean `pixel_auroc`: 0.966469
#### `results/gnl/GNL_AUPRO_recalculated_15.csv`
- Rows: 15
- Mean `images`: 115.000000
- Mean `regions`: 125.600000
- Mean `max_fpr`: 0.300000
- Mean `aupro`: 0.903178

## 4. EFDM evidence

- Files discovered in the EFDM backup/evidence roots: **53**
- File extensions: `{".csv": 4, ".ipynb": 1, ".json": 30, ".md": 1, ".npy": 16, ".txt": 1}`
- Small CSV/JSON evidence files are enumerated in `member3_final_evidence_index.json`.

The bottle and grid anomaly-map evidence copies in `EFDM_COMPLETE_EVIDENCE/category_results/` were removed only after their SHA-256 hashes were verified identical to the corresponding backup files. The backup originals remain preserved. The wood evidence copies were different and were preserved.

## 5. RD4AD

- Training script present: **False**
- Evaluation script present: **False**
- `latest.pth` checkpoint files discovered in the backup: **15**

Checkpoint presence does not by itself establish successful evaluation or a complete RD4AD comparison. Checkpoint evaluation should be reported only when matching evaluation outputs are available.

## 6. Reproducibility and interpretation

- Existing results were inventoried; this task did not retrain models.
- No new model metrics were fabricated or inferred from missing files.
- The evidence index records file discovery, row counts, numeric column means, plot availability, and checkpoint inventory.
- For scientific comparison, use identical test splits, image preprocessing, scoring rules, and metric implementations across methods.
- Report image AUROC, pixel AUROC, and AUPRO separately. Do not treat them as interchangeable.

## 7. Remaining delivery item

The documentation commit is local. The earlier push failed with:

`fatal: could not read Username for 'https://github.com': No such device or address`

Therefore the local commit must be pushed from an authenticated Git environment, or after configuring a GitHub credential in Kaggle Secrets and an appropriate credential helper. Do not paste a personal access token directly into a notebook cell or commit it to the repository.

## 8. Files created by this task

- `docs/member3_final_research_summary.md`
- `docs/member3_final_evidence_index.json`
