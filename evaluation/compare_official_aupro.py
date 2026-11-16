
import os
import sys
import numpy as np

from evaluation.metrics import compute_aupro

# Locate the downloaded official MVTec evaluation package.
official_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "mvtec_ad_evaluation_reference",
        "mvtec_ad_evaluation"
    )
)

sys.path.insert(0, official_path)

from pro_curve_util import compute_pro
from generic_util import trapezoid


def compare_case(name, masks, anomaly_maps):
    # Our implementation.
    our_result = compute_aupro(masks, anomaly_maps, max_fpr=0.30)

    # Official MVTec implementation.
    official_fprs, official_pros = compute_pro(
        anomaly_maps=list(anomaly_maps),
        ground_truth_maps=list(masks.astype(np.uint8))
    )

    official_result = trapezoid(
        official_fprs,
        official_pros,
        x_max=0.30
    ) / 0.30

    difference = abs(our_result - official_result)

    print("\nTest:", name)
    print("Our AUPRO:      ", our_result)
    print("Official AUPRO: ", official_result)
    print("Difference:     ", difference)

    assert np.isclose(
    our_result,
    official_result,
    rtol=1e-8,
    atol=1e-10
), "AUPRO mismatch with official implementation"

    print("PASSED:", name)


# Reproducible test data.
rng = np.random.RandomState(42)

masks = np.zeros((3, 32, 32), dtype=np.uint8)

# Two separate defect regions in image 1.
masks[0, 3:8, 3:8] = 1
masks[0, 20:25, 20:25] = 1

# One defect region in image 2.
masks[1, 12:18, 12:18] = 1

# Image 3 remains normal.

# Test 1: Random continuous anomaly scores.
maps_random = rng.random_sample((3, 32, 32))

# Increase scores in defect regions.
maps_random[masks == 1] += 1.0

compare_case(
    "Random scores with multiple defect regions",
    masks,
    maps_random
)

# Test 2: Scores with many ties.
maps_tied = rng.randint(
    0, 6,
    size=(3, 32, 32)
).astype(np.float64)

maps_tied[masks == 1] += 5.0

compare_case(
    "Repeated anomaly scores",
    masks,
    maps_tied
)

print("\nALL OFFICIAL AUPRO COMPARISONS PASSED")
