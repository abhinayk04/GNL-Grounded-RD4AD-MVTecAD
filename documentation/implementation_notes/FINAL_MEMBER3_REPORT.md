# Member 3 - Final PCA-GNL Experimental Report

Generated: 2026-11-18T21:09:04.925982

## Completion

- Categories evaluated: 15/15
- PCA bases validated: 45/45
- PCA components: 16
- GNL retraining: No
- Original baseline checkpoints modified: No

## Macro results

| Metric | GNL | PCA-GNL | Difference |
|---|---:|---:|---:|
| Pixel AUROC | 0.966469 | 0.964577 | -0.001893 |
| Image AUROC | 0.959216 | 0.961515 | +0.002299 |

## Per-category findings

- Pixel AUROC improved in 6 of 15 categories.
- Image AUROC improved in 5 categories.
- Image AUROC was unchanged in 5 categories.
- Image AUROC decreased in 5 categories.

## Method

Category-specific PCA bases were generated from normal MVTec training images.
The feature difference was teacher(original) minus teacher(AugMix).
PCA was calculated independently at the three feature levels with 16 components.

At inference, teacher and student features were projected onto the orthogonal
complement of the nuisance PCA subspace. The original GNL cosine-distance
scoring, bilinear resizing to 256x256, Gaussian smoothing with sigma 4,
image score maximum and mask threshold 0.5 were retained.

## Interpretation

PCA-GNL improved macro image-level AUROC by +0.002299
but changed macro pixel-level AUROC by -0.001893.

The effect is category-dependent. The method should not be described as
universally improving GNL. Both positive and negative category-level results
must be reported.

## Package contents

- pca_bases/: category-specific PCA bases for 15 categories and 3 levels
- evaluation_results/: combined JSON, CSV and comparison graphs
- implementation/: PCA source implementation
- member3_pca_implementation.md
- member3_integration_handoff.md

## Reproduction note

The original GNL inference script and checkpoints were preserved.
Bottle/Cable detailed results were reused from their previously validated run.
The remaining 13 categories were evaluated against their original stored
baseline summaries. See the combined results JSON for per-category values.
