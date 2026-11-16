
import numpy as np

from evaluation.metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_aupro,
    compute_all_metrics
)


def expect_value_error(function, description):
    try:
        function()
    except ValueError:
        print("PASSED:", description)
        return

    raise AssertionError("Expected ValueError: " + description)


# ------------------------------------------------
# 1. Image-level AUROC
# ------------------------------------------------

labels = np.array([0, 1, 0, 1])
scores = np.array([0.1, 0.9, 0.2, 0.8])

assert abs(compute_image_auroc(labels, scores) - 1.0) < 1e-6

print("PASSED: Image AUROC")


# Invalid image inputs
expect_value_error(
    lambda: compute_image_auroc([], []),
    "Empty image inputs"
)

expect_value_error(
    lambda: compute_image_auroc([0, 1], [0.2]),
    "Mismatched image lengths"
)

expect_value_error(
    lambda: compute_image_auroc([0, 1], [0.2, np.nan]),
    "NaN image scores"
)

expect_value_error(
    lambda: compute_image_auroc([0, 2], [0.2, 0.8]),
    "Invalid image labels"
)

expect_value_error(
    lambda: compute_image_auroc([0, 0], [0.2, 0.3]),
    "Missing defective image class"
)


# ------------------------------------------------
# 2. Pixel AUROC
# ------------------------------------------------

masks = np.zeros((2, 1, 32, 32), dtype=np.float32)
anomaly_maps = np.zeros_like(masks)

masks[1, 0, 10:14, 10:14] = 1

anomaly_maps[masks == 1] = 0.9
anomaly_maps[masks == 0] = 0.1

pixel_auc = compute_pixel_auroc(masks, anomaly_maps)

assert abs(pixel_auc - 1.0) < 1e-6

print("PASSED: Pixel AUROC")


expect_value_error(
    lambda: compute_pixel_auroc(
        np.zeros((2, 1, 32, 32)),
        np.zeros((2, 1, 32, 32))
    ),
    "Pixel AUROC with no defect pixels"
)


# ------------------------------------------------
# 3. Multiple defect regions and multiple images
# ------------------------------------------------

multi_masks = np.zeros((3, 1, 32, 32), dtype=np.float32)

# Two separate regions in image 1.
multi_masks[1, 0, 3:7, 3:7] = 1
multi_masks[1, 0, 20:24, 20:24] = 1

# One region in image 2.
multi_masks[2, 0, 12:17, 12:17] = 1

multi_maps = np.zeros_like(multi_masks)

multi_maps[multi_masks == 1] = 0.9
multi_maps[multi_masks == 0] = 0.1

multi_auc = compute_pixel_auroc(multi_masks, multi_maps)

multi_aupro = compute_aupro(
    multi_masks,
    multi_maps
)

assert abs(multi_auc - 1.0) < 1e-6
assert 0.0 <= multi_aupro <= 1.0

print("PASSED: Multiple images and defect regions")
print("Multiple-region AUPRO:", multi_aupro)


# ------------------------------------------------
# 4. Imperfect anomaly predictions
# ------------------------------------------------

poor_maps = np.full_like(multi_masks, 0.5)

poor_aupro = compute_aupro(
    multi_masks,
    poor_maps
)

assert np.isfinite(poor_aupro)
assert 0.0 <= poor_aupro <= 1.0

print("PASSED: Imperfect prediction map")
print("Imperfect AUPRO:", poor_aupro)


# ------------------------------------------------
# 5. Invalid AUPRO inputs
# ------------------------------------------------

expect_value_error(
    lambda: compute_aupro(
        np.zeros((2, 1, 32, 32)),
        np.zeros((2, 1, 32, 32))
    ),
    "AUPRO without defect regions"
)

expect_value_error(
    lambda: compute_aupro(
        multi_masks,
        np.full_like(multi_masks, np.nan)
    ),
    "AUPRO with NaN scores"
)

expect_value_error(
    lambda: compute_aupro(
        multi_masks,
        multi_maps,
        max_fpr=0.0
    ),
    "Invalid AUPRO integration limit"
)


# ------------------------------------------------
# 6. Combined evaluation
# ------------------------------------------------

results = compute_all_metrics(
    labels,
    scores,
    masks,
    anomaly_maps
)

for name, value in results.items():
    assert np.isfinite(value)
    print(name, ":", value)

print("\nALL EXPANDED METRIC TESTS PASSED")
