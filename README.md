# GNL-Grounded RD4AD for MVTec AD

## Project Overview

Implementation and evaluation of three anomaly detection methods:

1. RD4AD - Reverse Distillation baseline
2. GNL - Generalized Normality Learning
3. PCA-Gated RD4AD/GNL - Proposed nuisance filtering method

Dataset: MVTec Anomaly Detection (MVTec AD).

## Team Responsibilities

- Member 1: Environment setup, RD4AD and GNL implementation.
- Member 2: Dataset preparation and evaluation.
- Member 3: PCA nuisance filtering and integration.

## Project Structure

- ADShift/: Original implementation.
- datasets/: MVTec AD dataset.
- checkpoints/: Trained model checkpoints.
- pca_method/: PCA implementation.
- experiments/: Experiment logs.
- results/: Evaluation metrics.
- documentation/: Project documentation.

## Experimental Rules

- Train using normal images only.
- Use official test images for evaluation.
- Preserve original ADShift code.
- Record actual experimental results.
- Do not upload large datasets or checkpoints to GitHub.

## Initial Target

Start with the MVTec AD bottle category.
