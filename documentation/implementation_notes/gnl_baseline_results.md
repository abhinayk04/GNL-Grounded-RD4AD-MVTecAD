# GNL Baseline — Implementation and Results Report

## 1. Project Overview

This report documents the completed Generalized Normality Learning (GNL)
baseline experiments on the MVTec Anomaly Detection (MVTec AD) dataset.

The experiments cover all 15 MVTec AD categories.

## 2. Training Status

GNL training has been completed for all 15 categories.

| Category | Checkpoint Epoch |
|---|---:|
| bottle | 4 |
| cable | 20 |
| capsule | 20 |
| hazelnut | 20 |
| metal_nut | 20 |
| pill | 20 |
| screw | 20 |
| toothbrush | 20 |
| transistor | 20 |
| zipper | 20 |
| carpet | 20 |
| grid | 20 |
| leather | 20 |
| tile | 20 |
| wood | 20 |

The trained checkpoints are preserved separately and are not included
in the Git repository.

## 3. Image-Level and Pixel-Level Results

The following metrics were obtained from the completed GNL evaluations.

| Category | Image AUROC | Pixel AUROC |
|---|---:|---:|
| bottle | 0.995238 | 0.983326 |
| cable | 0.957834 | 0.979222 |
| capsule | 0.937774 | 0.985056 |
| hazelnut | 1.000000 | 0.990227 |
| metal_nut | 1.000000 | 0.974899 |
| pill | 0.971904 | 0.975906 |
| screw | 0.924985 | 0.992548 |
| toothbrush | 0.986111 | 0.989066 |
| transistor | 0.974167 | 0.901828 |
| zipper | 0.887342 | 0.976140 |
| carpet | 0.991974 | 0.991042 |
| grid | 0.777778 | 0.852005 |
| leather | 1.000000 | 0.994252 |
| tile | 0.992785 | 0.956159 |
| wood | 0.990351 | 0.955364 |
| **Macro average** | **0.9445** | **0.9683** |

## 4. AUPRO Evaluation

AUPRO was recalculated using the saved anomaly maps and ground-truth
masks. The calculation used:

- 256 x 256 anomaly maps.
- The evaluation mask preprocessing used by the inference pipeline.
- 8-connected connected components.
- Maximum false-positive rate of 0.30.
- The project's evaluation metric implementation.

The resulting macro AUPRO is **0.903178** (90.32%).

Per-category results are stored in:

    results/gnl/GNL_AUPRO_recalculated_15.csv

The recalculated file uses 8-connected regions to match the current
project metric implementation. The earlier AUPRO CSV is retained
separately as a historical result.

## 5. Result Files

The following CSV files are included in the repository:

- results/gnl/GNL_all_15_categories.csv
- results/gnl/GNL_AUPRO_recalculated_15.csv

These files preserve the category-level evaluation results.

## 6. Validation and Data Integrity

The saved anomaly-map audit covered all 15 categories.

The audit verified:
- 1,725 anomaly maps in total.
- Image and mask counts reconcile with the evaluation records.
- No missing image/mask pairs were detected.
- No duplicate map names were detected.
- Anomaly maps have 256 x 256 dimensions.
- No invalid map shapes or non-finite values were detected.

The results were copied into the clean project repository and
verified against their source files using SHA256 hashes.

## 7. Current Scope and Limitations

The GNL baseline has completed training and evaluation across all
15 MVTec AD categories.

The following work remains separate:
1. Integrate and evaluate the RD4AD baseline.
2. Integrate the Member 3 nuisance PCA method with actual model features.
3. Evaluate the PCA-enhanced method.
4. Compare GNL, RD4AD, and PCA-enhanced results under a consistent
   evaluation protocol.

The reported GNL metrics are baseline results. They do not represent
RD4AD or PCA-enhanced performance.

## 8. Conclusion

The GNL baseline training and evaluation have been completed for all
15 MVTec AD categories. The category-level image AUROC, pixel AUROC,
and recalculated AUPRO results are preserved in the project results
directory.

RD4AD integration and PCA-enhanced evaluation remain pending.
