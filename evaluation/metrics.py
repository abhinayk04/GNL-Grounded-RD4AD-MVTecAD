
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
    """Calculate image-level AUROC."""
    image_labels = _to_numpy(image_labels).reshape(-1)
    image_scores = _to_numpy(image_scores).reshape(-1)

    if len(image_labels) != len(image_scores):
        raise ValueError("Image labels and scores must have equal lengths")

    if len(np.unique(image_labels)) != 2:
        raise ValueError("Image AUROC requires normal and defective samples")

    return float(roc_auc_score(image_labels, image_scores))


def compute_pixel_auroc(masks, anomaly_maps):
    """Calculate pixel-level AUROC."""
    masks, anomaly_maps = _prepare_maps(masks, anomaly_maps)

    labels = masks.reshape(-1).astype(np.uint8)
    scores = anomaly_maps.reshape(-1)

    if len(np.unique(labels)) != 2:
        raise ValueError("Pixel AUROC requires both normal and defect pixels")

    return float(roc_auc_score(labels, scores))



def compute_aupro(masks, anomaly_maps, num_thresholds=200, max_fpr=0.30):
    """
    Calculate normalized area under the per-region overlap curve.

    Each connected defect region is evaluated only against
    predictions from its own image.
    """
    masks, anomaly_maps = _prepare_maps(masks, anomaly_maps)

    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")

    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be between 0 and 1")

    structure = np.ones((3, 3), dtype=np.uint8)

    # Store each region together with its image index.
    components = []

    total_background = int((~masks).sum())

    for image_index, mask in enumerate(masks):
        labeled, count = connected_components(
            mask.astype(np.uint8),
            structure=structure
        )

        for component_id in range(1, count + 1):
            region = labeled == component_id
            components.append((image_index, region))

    if not components:
        raise ValueError("AUPRO requires at least one defect region")

    if total_background == 0:
        raise ValueError("AUPRO requires normal background pixels")

    min_score = float(anomaly_maps.min())
    max_score = float(anomaly_maps.max())

    thresholds = np.linspace(
        max_score + 1e-8,
        min_score - 1e-8,
        num_thresholds
    )

    fpr_values = []
    pro_values = []

    for threshold in thresholds:
        predictions = anomaly_maps >= threshold

        false_positives = np.logical_and(
            predictions, ~masks
        ).sum()

        fpr = float(false_positives) / total_background

        region_overlaps = []

        for image_index, region in components:
            image_prediction = predictions[image_index]

            overlap = np.logical_and(
                image_prediction, region
            ).sum()

            region_overlaps.append(
                float(overlap) / float(region.sum())
            )

        pro = float(np.mean(region_overlaps))

        fpr_values.append(fpr)
        pro_values.append(pro)

    fpr_values = np.asarray(fpr_values)
    pro_values = np.asarray(pro_values)

    # Sort by false-positive rate.
    order = np.argsort(fpr_values)
    fpr_values = fpr_values[order]
    pro_values = pro_values[order]

    # Keep the maximum PRO at each duplicate FPR.
    unique_fpr = np.unique(fpr_values)

    unique_pro = np.array([
        np.max(pro_values[fpr_values == value])
        for value in unique_fpr
    ])

    # Include the origin if it is missing.
    if unique_fpr[0] > 0:
        unique_fpr = np.insert(unique_fpr, 0, 0.0)
        unique_pro = np.insert(unique_pro, 0, 0.0)

    # Clip the curve to the requested FPR limit.
    inside = unique_fpr < max_fpr

    clipped_fpr = unique_fpr[inside]
    clipped_pro = unique_pro[inside]

    endpoint_pro = np.interp(
        max_fpr, unique_fpr, unique_pro
    )

    clipped_fpr = np.append(clipped_fpr, max_fpr)
    clipped_pro = np.append(clipped_pro, endpoint_pro)

    area = np.trapz(clipped_pro, clipped_fpr)

    result = float(area / max_fpr)

    # Numerical safety check, not a replacement for the calculation.
    if not np.isfinite(result) or result < -1e-8 or result > 1.0 + 1e-8:
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
