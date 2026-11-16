
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import label as connected_components


def _to_numpy(data):
    """Convert NumPy arrays or PyTorch tensors to NumPy."""
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.asarray(data)


def _prepare_maps(masks, anomaly_maps):
    """Convert masks and anomaly maps to [N, H, W]."""
    masks = _to_numpy(masks)
    anomaly_maps = _to_numpy(anomaly_maps)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if anomaly_maps.ndim == 4 and anomaly_maps.shape[1] == 1:
        anomaly_maps = anomaly_maps[:, 0]

    if masks.ndim != 3 or anomaly_maps.ndim != 3:
        raise ValueError("Masks and anomaly maps must have shape [N,H,W]")

    if masks.shape != anomaly_maps.shape:
        raise ValueError("Masks and anomaly maps must have matching shapes")

    masks = masks > 0.5

    if not np.all(np.isfinite(anomaly_maps)):
        raise ValueError("Anomaly maps contain NaN or infinite values")

    return masks, anomaly_maps.astype(np.float64)


def compute_image_auroc(image_labels, image_scores):
    """Calculate image-level AUROC with input validation."""

    image_labels = _to_numpy(image_labels).reshape(-1)
    image_scores = _to_numpy(image_scores).reshape(-1)

    if len(image_labels) == 0:
        raise ValueError("Image labels cannot be empty")

    if len(image_labels) != len(image_scores):
        raise ValueError("Image labels and scores must have equal lengths")

    if not np.all(np.isfinite(image_scores)):
        raise ValueError("Image scores contain NaN or infinite values")

    if not np.all(np.isin(image_labels, [0, 1])):
        raise ValueError("Image labels must contain only 0 and 1")

    if len(np.unique(image_labels)) != 2:
        raise ValueError(
            "Image AUROC requires normal and defective samples"
        )

    return float(
        roc_auc_score(image_labels, image_scores)
    )

def compute_pixel_auroc(masks, anomaly_maps):
    """Calculate pixel-level AUROC."""
    masks, anomaly_maps = _prepare_maps(masks, anomaly_maps)

    labels = masks.reshape(-1).astype(np.uint8)
    scores = anomaly_maps.reshape(-1)

    if len(np.unique(labels)) != 2:
        raise ValueError("Pixel AUROC requires both normal and defect pixels")

    return float(roc_auc_score(labels, scores))




def compute_aupro(masks, anomaly_maps, max_fpr=0.30):
    """
    Calculate AUPRO using the official MVTec AD curve-construction
    and integration procedure.

    Uses all distinct anomaly-score thresholds, 8-connected
    components, and equal weighting of defect regions.
    """
    from bisect import bisect

    masks, anomaly_maps = _prepare_maps(masks, anomaly_maps)

    if masks.shape[0] == 0:
        raise ValueError("AUPRO requires at least one image")

    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be between 0 and 1")

    structure = np.ones((3, 3), dtype=np.uint8)

    num_gt_regions = 0
    labeled_masks = []

    for mask in masks:
        labeled, count = connected_components(
            mask.astype(np.uint8),
            structure=structure
        )
        labeled_masks.append(labeled)
        num_gt_regions += count

    if num_gt_regions == 0:
        raise ValueError("AUPRO requires at least one defect region")

    num_ok_pixels = int((~masks).sum())

    if num_ok_pixels == 0:
        raise ValueError("AUPRO requires normal background pixels")

    # Flatten scores and prepare official PRO/FPR contributions.
    anomaly_scores = anomaly_maps.ravel().copy()

    fp_changes = (~masks).astype(np.uint8).ravel()
    pro_changes = np.zeros(masks.shape, dtype=np.float64)

    for image_index, labeled in enumerate(labeled_masks):
        for region_id in range(1, int(labeled.max()) + 1):
            region = labeled == region_id
            region_size = int(region.sum())

            pro_changes[image_index][region] = 1.0 / region_size

    pro_changes = pro_changes.ravel()

    # Sort all anomaly scores in descending order.
    sort_indices = np.argsort(anomaly_scores)[::-1]

    anomaly_scores = anomaly_scores[sort_indices]
    fp_changes = fp_changes[sort_indices]
    pro_changes = pro_changes[sort_indices]

    # Calculate cumulative FPR and PRO.
    fprs = np.cumsum(fp_changes, dtype=np.uint64).astype(np.float64)
    fprs /= num_ok_pixels

    pros = np.cumsum(pro_changes)
    pros /= num_gt_regions

    # Keep only the final point for each equal anomaly score.
    keep_mask = np.append(
        np.diff(anomaly_scores) != 0,
        True
    )

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]

    # Match the official numerical clipping.
    fprs = np.clip(fprs, 0.0, 1.0)
    pros = np.clip(pros, 0.0, 1.0)

    # Add the curve's starting and ending points.
    fprs = np.concatenate(([0.0], fprs, [1.0]))
    pros = np.concatenate(([0.0], pros, [1.0]))

    # Official MVTec trapezoidal integration with interpolation
    # at max_fpr when it is not already a curve point.
    correction = 0.0

    if max_fpr not in fprs:
        insertion_index = bisect(fprs, max_fpr)

        if not 0 < insertion_index < len(fprs):
            raise ValueError("Integration limit is outside the curve")

        interpolated_pro = (
            pros[insertion_index - 1]
            + (pros[insertion_index] - pros[insertion_index - 1])
            * (max_fpr - fprs[insertion_index - 1])
            / (fprs[insertion_index] - fprs[insertion_index - 1])
        )

        correction = (
            0.5
            * (interpolated_pro + pros[insertion_index - 1])
            * (max_fpr - fprs[insertion_index - 1])
        )

    inside = fprs <= max_fpr

    x = fprs[inside]
    y = pros[inside]

    area = (
        np.sum(
            0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])
        )
        + correction
    )

    result = float(area / max_fpr)

    if not np.isfinite(result) or not 0.0 <= result <= 1.0 + 1e-8:
        raise RuntimeError("Invalid AUPRO value: {}".format(result))

    return float(np.clip(result, 0.0, 1.0))

def compute_all_metrics(
    image_labels,
    image_scores,
    masks,
    anomaly_maps
):
    """Calculate all three anomaly detection metrics."""
    return {
        "image_auroc": compute_image_auroc(
            image_labels, image_scores
        ),
        "pixel_auroc": compute_pixel_auroc(
            masks, anomaly_maps
        ),
        "aupro": compute_aupro(
            masks, anomaly_maps
        )
    }
