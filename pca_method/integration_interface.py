
import torch
from pca_method.inference_pca_gating import apply_pca_gating


def pca_filtered_residual(
    teacher_features,
    student_features,
    nuisance_basis
):
    """
    Apply nuisance PCA filtering to teacher-student feature residuals.

    Args:
        teacher_features: Teacher output [B, C, H, W].
        student_features: Student output [B, C, H, W].
        nuisance_basis: PCA basis [C, K].

    Returns:
        Dictionary containing:
            raw_residual: Original teacher-student residual.
            filtered_residual: PCA-filtered residual.
    """

    if teacher_features.ndim != 4:
        raise ValueError("Teacher features must have shape [B,C,H,W].")

    if student_features.ndim != 4:
        raise ValueError("Student features must have shape [B,C,H,W].")

    if teacher_features.shape != student_features.shape:
        raise ValueError("Teacher and student feature shapes must match.")

    if nuisance_basis.ndim != 2:
        raise ValueError("Nuisance basis must have shape [C,K].")

    if teacher_features.shape[1] != nuisance_basis.shape[0]:
        raise ValueError("Feature channels and PCA basis channels do not match.")

    # Compute teacher-student residual
    raw_residual = teacher_features - student_features

    # Apply PCA nuisance filtering
    filtered_residual = apply_pca_gating(
        raw_residual,
        nuisance_basis
    )

    return {
        "raw_residual": raw_residual,
        "filtered_residual": filtered_residual
    }
