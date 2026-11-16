
import torch
from pathlib import Path


def compute_nuisance_basis(feature_differences, num_components=16):
    """
    Compute a channel-wise nuisance PCA basis.

    Args:
        feature_differences:
            List of tensors shaped [B,C,H,W] or [C,H,W].
            Each tensor contains teacher feature differences:
            teacher(x) - teacher(T(x)).

        num_components:
            Number of principal directions to retain.

    Returns:
        basis: [C,K] orthonormal nuisance basis.
        eigenvalues: [K] selected eigenvalues.
        explained_variance_ratio: [K] variance ratios.
    """

    if not feature_differences:
        raise ValueError("No feature differences provided.")

    covariance = None
    total_columns = 0

    for difference in feature_differences:

        difference = torch.as_tensor(
            difference, dtype=torch.float64
        )

        if difference.ndim == 3:
            difference = difference.unsqueeze(0)

        if difference.ndim != 4:
            raise ValueError(
                "Expected feature differences shaped [B,C,H,W] "
                "or [C,H,W]."
            )

        batch, channels, height, width = difference.shape

        # Flatten spatial positions, preserving channels.
        flattened = difference.reshape(batch, channels, -1)

        # Combine batch and spatial positions.
        flattened = flattened.permute(1, 0, 2).reshape(channels, -1)

        if covariance is None:
            covariance = torch.zeros(
                channels, channels, dtype=torch.float64
            )

        covariance += flattened @ flattened.T
        total_columns += flattened.shape[1]

    covariance /= total_columns

    # Symmetric eigendecomposition.
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    # Sort eigenvalues in descending order.
    indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    if num_components < 1 or num_components > eigenvectors.shape[1]:
        raise ValueError("Invalid number of PCA components.")

    basis = eigenvectors[:, :num_components].contiguous()

    total_variance = eigenvalues.clamp(min=0).sum()
    explained_variance_ratio = (
        eigenvalues[:num_components].clamp(min=0)
        / total_variance.clamp(min=1e-12)
    )

    return basis.float(), eigenvalues[:num_components].float(), explained_variance_ratio.float()


def save_nuisance_basis(
    basis,
    eigenvalues,
    explained_variance_ratio,
    category,
    save_dir="pca_method/saved_bases"
):
    """Save the PCA basis and its metadata."""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / f"{category}.pth"

    torch.save(
        {
            "basis": basis.cpu(),
            "eigenvalues": eigenvalues.cpu(),
            "explained_variance_ratio": explained_variance_ratio.cpu(),
            "category": category,
        },
        output_path
    )

    print(f"Saved PCA basis: {output_path}")
    print(f"Basis shape: {tuple(basis.shape)}")
    print(
        "Explained variance:",
        explained_variance_ratio.sum().item()
    )

    return output_path
