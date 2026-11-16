
# Member 3 — PCA Integration Handoff

## 1. Purpose

This document explains how Member 1 can integrate the
Member 3 nuisance PCA implementation into the RD4AD/GNL pipeline.

## 2. Files Provided by Member 3

- pca_method/compute_nuisance_pca.py
- pca_method/inference_pca_gating.py
- pca_method/pca_config.py
- pca_method/integration_interface.py

## 3. Required Inputs

The integration interface expects:

1. Teacher feature tensor.
2. Student feature tensor.
3. A previously computed nuisance PCA basis.

Expected dimensions:

- Teacher features: [B, C, H, W]
- Student features: [B, C, H, W]
- PCA basis: [C, K]

Teacher and student feature shapes must match.
The PCA basis channel dimension must match C.

## 4. Integration Procedure

### Step 1: Extract Teacher Features

Use the frozen teacher model to obtain feature representations.

### Step 2: Extract Student Features

Use the student model to obtain corresponding feature representations.

### Step 3: Load the PCA Basis

Load the category-specific PCA basis generated from actual
teacher feature differences.

Example:

    checkpoint = torch.load(
        "pca_method/saved_bases/bottle.pth",
        map_location="cpu"
    )

    basis = checkpoint["basis"]

### Step 4: Apply PCA Filtering

Example:

    from pca_method.integration_interface import (
        pca_filtered_residual
    )

    result = pca_filtered_residual(
        teacher_features,
        student_features,
        basis
    )

    raw_residual = result["raw_residual"]
    filtered_residual = result["filtered_residual"]

## 5. Important Requirements

- Teacher and student features must correspond to the same
  feature stage.
- Their tensor shapes must match.
- The PCA basis must be computed using actual teacher feature
  differences, not random synthetic features.
- The PCA basis must have the same channel dimension as the
  selected feature tensors.
- PCA component count must be selected through validation.
- Keep the original RD4AD/GNL anomaly-scoring method available
  for baseline comparison.

## 6. Current Validation Status

The standalone PCA implementation has passed:

- Centered covariance computation tests.
- GPU-input and CPU-PCA compatibility tests.
- Twelve synthetic regression tests.
- Inference projection tests.
- Integration interface tests.
- Error-handling tests.

These tests validate implementation behavior, not anomaly
detection performance.

## 7. Pending Integration Work

1. Connect the actual RD4AD/GNL teacher and student models.
2. Identify the correct feature extraction stage.
3. Generate actual MVTec AD teacher feature differences.
4. Compute the category-specific nuisance PCA basis.
5. Apply PCA filtering during inference.
6. Evaluate the baseline and PCA-enhanced model.
7. Compare anomaly detection metrics.

## 8. Expected Team Responsibilities

Member 1:
- Provide teacher/student feature extraction.
- Confirm model architecture and feature dimensions.
- Integrate the PCA interface into the model pipeline.

Member 2:
- Prepare MVTec AD dataset.
- Support evaluation and anomaly metrics.
- Record baseline and PCA-enhanced results.

Member 3:
- Maintain PCA computation and projection modules.
- Generate nuisance bases from actual teacher features.
- Validate PCA integration.
- Support experimental comparisons.
