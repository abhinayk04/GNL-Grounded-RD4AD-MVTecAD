
import numpy as np

from evaluation.metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_aupro,
    compute_all_metrics
)


# ------------------------------------------------
# 1. Image-level AUROC test
# ------------------------------------------------

image_labels = np.array([0, 1, 0, 1])
image_scores = np.array([0.1, 0.9, 0.2, 0.8])

image_auc = compute_image_auroc(
    image_labels, image_scores
)

print("Image AUROC:", image_auc)
assert abs(image_auc - 1.0) < 1e-6


# ------------------------------------------------
# 2. Pixel-level AUROC and AUPRO test
# ------------------------------------------------

masks = np.zeros((2, 1, 32, 32), dtype=np.float32)

# Insert a 4x4 defect region into the second image.
masks[1, 0, 10:14, 10:14] = 1.0

anomaly_maps = np.zeros((2, 1, 32, 32), dtype=np.float32)

# Give defect pixels high anomaly scores.
anomaly_maps[masks == 1] = 0.9

# Give normal pixels low anomaly scores.
anomaly_maps[masks == 0] = 0.1

pixel_auc = compute_pixel_auroc(masks, anomaly_maps)
aupro = compute_aupro(
    masks, anomaly_maps, num_thresholds=200
)

print("Pixel AUROC:", pixel_auc)
print("AUPRO:", aupro)

assert abs(pixel_auc - 1.0) < 1e-6
assert 0.0 <= aupro <= 1.0
assert aupro > 0.9


# ------------------------------------------------
# 3. Combined evaluation test
# ------------------------------------------------

results = compute_all_metrics(
    image_labels,
    image_scores,
    masks,
    anomaly_maps
)

print("\nCombined metric results:")
for name, value in results.items():
    print(name, ":", value)

print("\nALL METRIC TESTS PASSED")
