
# Member 3 — PCA Implementation and Validation Report

## 1. Project Overview

This work is part of the GNL-Grounded RD4AD project for MVTec AD.

The objective of Member 3 is to implement a nuisance PCA method that removes
feature variations associated with content-preserving transformations from
the teacher-student feature residual.

The implementation is designed to work with feature tensors in BCHW format.

## 2. Implementation Responsibilities

Member 3 is responsible for:

- Nuisance PCA basis computation.
- PCA basis configuration and serialization.
- Inference-time PCA projection.
- Numerical validation of the PCA implementation.
- Documentation of implementation and testing.

## 3. Methodology

### 3.1 Nuisance Feature Differences

For an input image x and a content-preserving transformation T(x), the
teacher feature difference is:

    Delta F = F_T(x) - F_T(T(x))

These feature differences are used to estimate the nuisance subspace.

### 3.2 Centered PCA

Feature tensors are reshaped so that channels form the feature dimension
and batch-spatial positions form the observation dimension.

The implementation computes the global channel mean, centers the
observations, and calculates the sample covariance matrix:

    C = X_centered X_centered^T / (N - 1)

The eigenvectors corresponding to the largest eigenvalues form the
nuisance PCA basis.

### 3.3 PCA Basis

The basis has shape:

    [C, K]

where C is the number of feature channels and K is the number of
selected principal components.

The current configuration uses K = 16 as a provisional setting.
The final value must be selected using real MVTec AD validation.

## 4. Inference-Time PCA Projection

The teacher-student residual is:

    R = F_T(x) - F_S(x)

The nuisance component is projected onto the PCA basis:

    R_nuisance = V(V^T R)

The filtered residual is:

    R_filtered = R - V(V^T R)

This removes the component of the residual lying in the nuisance
subspace, assuming V has orthonormal columns.

The implementation is provided in:

    pca_method/inference_pca_gating.py

## 5. Implemented Files

### compute_nuisance_pca.py

Implements:
- Feature-difference processing.
- Global channel mean calculation.
- Centered covariance estimation.
- Symmetric eigendecomposition.
- Eigenvalue sorting.
- Explained variance calculation.
- PCA basis saving.

### inference_pca_gating.py

Implements:
- BCHW residual validation.
- Basis dimension validation.
- Nuisance projection.
- Filtered residual reconstruction.

### pca_config.py

Contains the preliminary PCA configuration, including:
- Target category: bottle.
- Number of components: 16.
- Candidate components: 8, 16 and 32.
- Category-specific basis setting.
- Teacher feature-difference definition.
- Teacher-student inference residual definition.

## 6. Synthetic Validation Results

The implementation was tested using synthetic feature tensors.

The following results were obtained after the centered covariance
correction:

| Components | Cumulative explained variance | Orthogonality error |
|---|---:|---:|
| 8 | 0.9230 | 2.38e-7 |
| 16 | 0.9422 | 2.38e-7 |
| 32 | 0.9709 | 2.38e-7 |

These are synthetic validation results, not MVTec AD benchmark results.

The tests also verified feature-channel configurations with 32, 64,
128 and 256 channels.

All tested configurations produced PCA bases with the expected shape
[C, 16].

## 7. Validation Scope and Limitations

The synthetic tests validate the mathematical and tensor-shape
behavior of the PCA implementation.

They do not establish anomaly-detection performance.

The following work remains:

1. Obtain actual teacher feature differences from the RD4AD/GNL pipeline.
2. Compute and save a real MVTec AD category-specific nuisance basis.
3. Integrate PCA projection into the actual inference pipeline.
4. Evaluate anomaly detection on MVTec AD.
5. Compare baseline and PCA-enhanced results.
6. Select the final PCA component count using validation results.

The current K = 16 setting is provisional.

## 8. Conclusion

Member 3 has implemented the core nuisance PCA computation and
inference-time projection modules.

The implementation has passed synthetic numerical and tensor-shape
validation. Integration with actual RD4AD/GNL model features and
evaluation on MVTec AD are still required before reporting final
experimental performance.
