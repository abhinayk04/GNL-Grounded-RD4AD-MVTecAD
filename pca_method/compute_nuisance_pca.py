
import torch
from pathlib import Path


def compute_nuisance_basis(feature_differences, num_components=16):

    if feature_differences is None or len(feature_differences) == 0:
        raise ValueError("No feature differences provided.")

    channel_sum = None
    total_columns = 0
    channels = None

    # All PCA calculations use CPU float64 for consistency.
    for difference in feature_differences:

        difference = torch.as_tensor(difference).detach().to(
            device="cpu", dtype=torch.float64
        )

        if difference.ndim == 3:
            difference = difference.unsqueeze(0)

        if difference.ndim != 4:
            raise ValueError("Expected [B,C,H,W] or [C,H,W].")

        batch, current_channels, height, width = difference.shape

        if channels is None:
            channels = current_channels
            channel_sum = torch.zeros(channels, dtype=torch.float64)
        elif current_channels != channels:
            raise ValueError("Inconsistent channel dimensions.")

        flattened = difference.reshape(batch, channels, -1)
        flattened = flattened.permute(1, 0, 2).reshape(channels, -1)

        channel_sum += flattened.sum(dim=1)
        total_columns += flattened.shape[1]

    if total_columns < 2:
        raise ValueError("At least two observations are required.")

    if num_components < 1 or num_components > channels:
        raise ValueError("Invalid number of PCA components.")

    mean = channel_sum / total_columns

    covariance = torch.zeros(
        channels, channels, dtype=torch.float64
    )

    for difference in feature_differences:

        difference = torch.as_tensor(difference).detach().to(
            device="cpu", dtype=torch.float64
        )

        if difference.ndim == 3:
            difference = difference.unsqueeze(0)

        batch = difference.shape[0]

        flattened = difference.reshape(batch, channels, -1)
        flattened = flattened.permute(1, 0, 2).reshape(channels, -1)

        centered = flattened - mean.unsqueeze(1)
        covariance += centered @ centered.T

    covariance /= (total_columns - 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[indices].clamp(min=0)
    eigenvectors = eigenvectors[:, indices]

    basis = eigenvectors[:, :num_components].contiguous()

    total_variance = eigenvalues.sum()

    explained_variance_ratio = (
        eigenvalues[:num_components]
        / total_variance.clamp(min=1e-12)
    )

    return (
        basis.float(),
        eigenvalues[:num_components].float(),
        explained_variance_ratio.float()
    )


def save_nuisance_basis(
    basis,
    eigenvalues,
    explained_variance_ratio,
    category,
    save_dir="pca_method/saved_bases"
):

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
    print("Explained variance:", explained_variance_ratio.sum().item())

    return output_path
