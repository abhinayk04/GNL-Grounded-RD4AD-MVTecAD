
import torch


def apply_pca_gating(residual, basis):
    """
    Remove the nuisance subspace from teacher-student residuals.

    Args:
        residual: [B,C,H,W]
        basis: [C,K]

    Returns:
        filtered_residual: [B,C,H,W]
    """

    if residual.ndim != 4:
        raise ValueError("Residual must have shape [B,C,H,W].")

    if basis.ndim != 2:
        raise ValueError("Basis must have shape [C,K].")

    batch, channels, height, width = residual.shape

    if basis.shape[0] != channels:
        raise ValueError(
            f"Channel mismatch: residual has {channels}, "
            f"basis has {basis.shape[0]}."
        )

    original_dtype = residual.dtype
    device = residual.device

    basis = basis.to(device=device, dtype=residual.dtype)
    residual_flat = residual.reshape(batch, channels, -1)

    # Orthogonal projection onto nuisance subspace.
    nuisance_component = basis @ (
        basis.transpose(0, 1) @ residual_flat
    )

    filtered = residual_flat - nuisance_component

    return filtered.reshape(batch, channels, height, width).to(original_dtype)
