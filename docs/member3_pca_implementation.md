# Member 3 - PCA Nuisance Subspace Implementation

## Objective
Reduce nuisance feature variation in GNL teacher/student representations
using category-specific Principal Component Analysis.

## Nuisance feature differences
For each normal MVTec training image, the frozen pretrained teacher extracts
features from the original image and its AugMix-transformed counterpart.

Delta F = F_teacher(original) - F_teacher(AugMix)

## PCA
PCA is performed independently for each category and each of the three
teacher feature levels. Spatial positions are treated as observations and
channels as features. The covariance is centered before eigendecomposition.

The retained basis contains K=16 principal directions.

## Inference
At each feature level, teacher and student features are projected onto the
orthogonal complement of the nuisance basis:

F_projected = F - V(V^T F)

Cosine distance is then calculated between projected teacher and student
features. The three distance maps are resized to 256x256, summed and
Gaussian-smoothed with sigma=4.

## Evaluation
Image-level score: maximum anomaly-map value.
Pixel-level evaluation: flattened anomaly map against ground-truth masks
thresholded at 0.5.

## Experimental safeguards
- PCA bases are category-specific.
- PCA bases are generated from train/good images only.
- GNL checkpoints are not retrained or modified.
- Original baseline inference files are preserved.
- Baseline and PCA are evaluated on identical test batches.
- Existing Bottle/Cable PCA results are reused.

## Limitations
PCA performance can vary by category. A gain in one metric does not imply
a gain in every metric. Test-set results are comparative experimental
results and should not be used to claim universal improvement.
