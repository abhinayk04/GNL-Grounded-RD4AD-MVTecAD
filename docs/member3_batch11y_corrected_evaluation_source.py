# ================================================================
# BATCH 11Y — CORRECTED ALL 15 CATEGORY EFDM-GNL EVALUATION
# Evaluation only. No training and no checkpoint modification.
# ================================================================

import os
import sys
import gc
import json
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

# -------------------- PATHS AND SETTINGS --------------------

ROOT = Path("/kaggle/working/project_clean")
DATA_ROOT = Path("/kaggle/working/mvtec_dataset")
BACKUP = Path("/kaggle/working/GNL_Project_Backup")

REPORT_DIR = ROOT / "docs"
RESULT_DIR = ROOT / "results/member3/efdm_gnl_all15"

JSON_REPORT = REPORT_DIR / "member3_batch11y_efdm_gnl_corrected_all15.json"
CSV_REPORT = RESULT_DIR / "efdm_gnl_corrected_all15_comparison.csv"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA = 0.5
GAUSSIAN_SIGMA = 4

assert DEVICE.type == "cuda", "CUDA is unavailable."
assert ROOT.is_dir(), f"Project root missing: {ROOT}"
assert DATA_ROOT.is_dir(), f"Dataset root missing: {DATA_ROOT}"

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from experiments.gnl.resnet import wide_resnet50_2
from experiments.gnl.de_resnet import de_wide_resnet50_2

# -------------------- IMAGE PREPROCESSING --------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_image(path):
    with Image.open(path) as im:
        image = im.convert("RGB")
        return transform(image).unsqueeze(0).to(DEVICE)

def load_mask(category, defect, image_stem):
    if defect == "good":
        return np.zeros((256, 256), dtype=np.uint8)

    path = (
        DATA_ROOT / category / "ground_truth" /
        defect / f"{image_stem}_mask.png"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth mask missing: {path}")

    with Image.open(path) as im:
        mask = im.convert("L").resize(
            (256, 256), Image.Resampling.NEAREST
        )
        return (np.asarray(mask) > 0).astype(np.uint8)

# -------------------- EFDM IMPLEMENTATION --------------------

def efdm_feature(x, reference, lamda=LAMBDA):
    blend = 1.0 - lamda
    b, c, h, w = x.shape

    x_flat = x.reshape(b, c, -1)
    ref_flat = reference.reshape(b, c, -1)

    sorted_x, indices = torch.sort(x_flat, dim=-1)
    sorted_ref, _ = torch.sort(ref_flat, dim=-1)

    mixed = sorted_x + (sorted_ref - sorted_x) * blend
    inverse_indices = torch.argsort(indices, dim=-1)

    return mixed.gather(
        -1, inverse_indices
    ).reshape(b, c, h, w).contiguous()

@torch.no_grad()
def extract_efdm_features(encoder, x, reference):
    def stem(z):
        z = encoder.conv1(z)
        z = encoder.bn1(z)
        z = encoder.relu(z)
        return encoder.maxpool(z)

    x = stem(x)
    reference = stem(reference)

    # Level A
    feature_a = encoder.layer1(x)
    reference_a = encoder.layer1(reference)
    feature_a = efdm_feature(feature_a, reference_a)

    # Level B
    feature_b = encoder.layer2(feature_a)
    reference_b = encoder.layer2(reference_a)
    feature_b = efdm_feature(feature_b, reference_b)

    # Level C: calculated from transformed B, no direct EFDM.
    feature_c = encoder.layer3(feature_b)

    return [feature_a, feature_b, feature_c]

# -------------------- ANOMALY MAP --------------------

def calculate_anomaly_map(teacher_features, student_features):
    assert len(teacher_features) == 3
    assert len(student_features) == 3

    maps = []

    # Both implementations return corresponding A, B, C levels.
    for level, (teacher, student) in enumerate(
        zip(teacher_features, student_features)
    ):
        if teacher.shape != student.shape:
            raise RuntimeError(
                f"Feature mismatch at level {level}: "
                f"{tuple(teacher.shape)} vs {tuple(student.shape)}"
            )

        distance = 1.0 - F.cosine_similarity(
            teacher, student, dim=1
        )

        distance = F.interpolate(
            distance.unsqueeze(1),
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        maps.append(distance[0, 0].cpu().numpy())

    anomaly_map = np.sum(maps, axis=0)
    return gaussian_filter(anomaly_map, sigma=GAUSSIAN_SIGMA)

# -------------------- CHECKPOINT DISCOVERY --------------------

def find_checkpoint(category):
    candidates = [
        BACKUP / "checkpoints" / f"{category}_full" / "latest.pth",
        BACKUP / "checkpoints" / category / "latest.pth"
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No trained checkpoint found for {category}. "
        f"Checked: {[str(p) for p in candidates]}"
    )

def find_reference(category):
    reference = DATA_ROOT / category / "train" / "good" / "000.png"

    if not reference.is_file():
        raise FileNotFoundError(
            f"Original EFDM reference missing: {reference}"
        )

    return reference


# -------------------- PER-CATEGORY EVALUATION --------------------

def evaluate_category(category):
    category_root = DATA_ROOT / category
    test_root = category_root / "test"

    checkpoint_path = find_checkpoint(category)
    reference_path = find_reference(category)

    if not test_root.is_dir():
        raise FileNotFoundError(f"Test directory missing: {test_root}")

    # Restore the category-specific trained GNL model.
    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu"
    )

    if checkpoint.get("category") != category:
        raise RuntimeError(
            f"Checkpoint category mismatch: "
            f"expected {category}, got {checkpoint.get('category')}"
        )

    bn.load_state_dict(checkpoint["bn"], strict=True)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)

    encoder = encoder.to(DEVICE).eval()
    bn = bn.to(DEVICE).eval()
    decoder = decoder.to(DEVICE).eval()

    reference = load_image(reference_path)

    baseline_scores = []
    efdm_scores = []
    image_labels = []

    baseline_pixel_maps = []
    efdm_pixel_maps = []
    ground_truth_masks = []

    changed_counts = [0, 0, 0]
    image_records = []

    defect_dirs = sorted(
        p for p in test_root.iterdir() if p.is_dir()
    )

    if not defect_dirs:
        raise RuntimeError(f"No defect folders in {test_root}")

    total_images = 0

    with torch.inference_mode():
        for defect_dir in defect_dirs:
            defect = defect_dir.name

            image_paths = sorted(
                p for p in defect_dir.iterdir()
                if p.suffix.lower() in (
                    ".png", ".jpg", ".jpeg", ".bmp"
                )
            )

            print(
                f"  {category}/{defect}: "
                f"{len(image_paths)} images"
            )

            for image_path in image_paths:
                x = load_image(image_path)

                # Original GNL baseline.
                baseline_teacher = encoder(x)
                baseline_student = decoder(bn(baseline_teacher))

                baseline_map = calculate_anomaly_map(
                    baseline_teacher, baseline_student
                )

                # EFDM-GNL.
                efdm_teacher = extract_efdm_features(
                    encoder, x, reference
                )
                efdm_student = decoder(bn(efdm_teacher))

                efdm_map = calculate_anomaly_map(
                    efdm_teacher, efdm_student
                )

                changed = [
                    not torch.allclose(
                        a, b, rtol=1e-5, atol=1e-6
                    )
                    for a, b in zip(
                        baseline_teacher, efdm_teacher
                    )
                ]

                changed_counts = [
                    old + int(new)
                    for old, new in zip(changed_counts, changed)
                ]

                mask = load_mask(
                    category, defect, image_path.stem
                )

                label = int(defect != "good")

                baseline_score = float(np.max(baseline_map))
                efdm_score = float(np.max(efdm_map))

                baseline_scores.append(baseline_score)
                efdm_scores.append(efdm_score)
                image_labels.append(label)

                baseline_pixel_maps.append(
                    baseline_map.reshape(-1)
                )
                efdm_pixel_maps.append(
                    efdm_map.reshape(-1)
                )
                ground_truth_masks.append(mask.reshape(-1))

                maps_finite = bool(
                    np.isfinite(baseline_map).all()
                    and np.isfinite(efdm_map).all()
                )

                image_records.append({
                    "image": image_path.name,
                    "defect": defect,
                    "label": label,
                    "baseline_score": baseline_score,
                    "efdm_score": efdm_score,
                    "feature_changed_A_B_C": changed,
                    "maps_finite": maps_finite
                })

                total_images += 1

    y_image = np.asarray(image_labels)
    y_pixel = np.concatenate(ground_truth_masks)

    baseline_pixels = np.concatenate(baseline_pixel_maps)
    efdm_pixels = np.concatenate(efdm_pixel_maps)

    if len(np.unique(y_image)) != 2:
        raise RuntimeError(
            f"{category}: image AUROC requires good and anomalous images"
        )

    if len(np.unique(y_pixel)) != 2:
        raise RuntimeError(
            f"{category}: pixel AUROC requires normal and anomalous pixels"
        )

    metrics = {
        "baseline_image_auroc": float(
            roc_auc_score(y_image, baseline_scores)
        ),
        "efdm_gnl_image_auroc": float(
            roc_auc_score(y_image, efdm_scores)
        ),
        "baseline_pixel_auroc": float(
            roc_auc_score(y_pixel, baseline_pixels)
        ),
        "efdm_gnl_pixel_auroc": float(
            roc_auc_score(y_pixel, efdm_pixels)
        )
    }

    metrics["image_auroc_delta"] = (
        metrics["efdm_gnl_image_auroc"]
        - metrics["baseline_image_auroc"]
    )

    metrics["pixel_auroc_delta"] = (
        metrics["efdm_gnl_pixel_auroc"]
        - metrics["baseline_pixel_auroc"]
    )

    return {
        "category": category,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "reference_image": str(reference_path),
        "image_count": total_images,
        "feature_changed_image_counts_A_B_C": changed_counts,
        "all_maps_finite": all(
            r["maps_finite"] for r in image_records
        ),
        "metrics": metrics,
        "per_image": image_records
    }

# -------------------- RUN ALL 15 CATEGORIES --------------------

REPORT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

all_results = []
failures = []

print("=" * 78)
print("BATCH 11Y — CORRECTED ALL 15 CATEGORY EFDM-GNL EVALUATION")
print("Device:", DEVICE)
print("Categories:", len(CATEGORIES))
print("EFDM lambda:", LAMBDA)
print("Gaussian sigma:", GAUSSIAN_SIGMA)
print("=" * 78)

for index, category in enumerate(CATEGORIES, start=1):
    print(f"\n[{index}/{len(CATEGORIES)}] CATEGORY: {category}")
    print("-" * 65)

    try:
        result = evaluate_category(category)
        all_results.append(result)

        m = result["metrics"]

        print(
            f"Images: {result['image_count']} | "
            f"Feature changes A/B/C: "
            f"{result['feature_changed_image_counts_A_B_C']}"
        )

        print(
            f"Image AUROC: "
            f"{m['baseline_image_auroc']:.6f} -> "
            f"{m['efdm_gnl_image_auroc']:.6f}"
        )

        print(
            f"Pixel AUROC: "
            f"{m['baseline_pixel_auroc']:.6f} -> "
            f"{m['efdm_gnl_pixel_auroc']:.6f}"
        )

    except Exception as exc:
        failures.append({
            "category": category,
            "error": repr(exc),
            "traceback": traceback.format_exc()
        })

        print("CATEGORY FAILED:", repr(exc))

    finally:
        for name in ("encoder", "bn", "decoder", "reference"):
            if name in globals():
                del globals()[name]

        gc.collect()
        torch.cuda.empty_cache()

# -------------------- AGGREGATE RESULTS --------------------

summary_rows = []

for result in all_results:
    row = {
        "category": result["category"],
        "checkpoint_epoch": result["checkpoint_epoch"],
        "image_count": result["image_count"],
        "feature_changed_A": result["feature_changed_image_counts_A_B_C"][0],
        "feature_changed_B": result["feature_changed_image_counts_A_B_C"][1],
        "feature_changed_C": result["feature_changed_image_counts_A_B_C"][2],
        "all_maps_finite": result["all_maps_finite"]
    }
    row.update(result["metrics"])
    summary_rows.append(row)

df = pd.DataFrame(summary_rows)

if not df.empty:
    df.to_csv(CSV_REPORT, index=False)

metric_columns = [
    "baseline_image_auroc",
    "efdm_gnl_image_auroc",
    "baseline_pixel_auroc",
    "efdm_gnl_pixel_auroc",
    "image_auroc_delta",
    "pixel_auroc_delta"
]

macro_metrics = {}

if not df.empty:
    for column in metric_columns:
        macro_metrics[column] = float(df[column].mean())

report = {
    "experiment": "Member 3 Batch 11O all-category EFDM-GNL evaluation",
    "categories_requested": CATEGORIES,
    "categories_completed": len(all_results),
    "categories_failed": len(failures),
    "device": str(DEVICE),
    "lambda": LAMBDA,
    "efdm_levels": ["A", "B"],
    "feature_order": ["A", "B", "C"],
    "gaussian_sigma": GAUSSIAN_SIGMA,
    "macro_category_metrics": macro_metrics,
    "category_results": all_results,
    "failures": failures,
    "csv_report": str(CSV_REPORT)
}

JSON_REPORT.write_text(json.dumps(report, indent=2))

# -------------------- FINAL DISPLAY --------------------

print("\n" + "=" * 100)
print("BATCH 11O — FINAL CATEGORY COMPARISON")
print("=" * 100)

if not df.empty:
    display_columns = [
        "category",
        "baseline_image_auroc",
        "efdm_gnl_image_auroc",
        "baseline_pixel_auroc",
        "efdm_gnl_pixel_auroc",
        "image_auroc_delta",
        "pixel_auroc_delta"
    ]

    print(df[display_columns].to_string(index=False))

    print("\nMACRO AVERAGE ACROSS SUCCESSFUL CATEGORIES")
    for key, value in macro_metrics.items():
        print(f"{key}: {value:.6f}")

print("\nCategories completed:", len(all_results))
print("Categories failed:", len(failures))

if failures:
    print("\nFAILURE DETAILS:")
    for failure in failures:
        print(failure["category"], "->", failure["error"])

print("\nJSON report:", JSON_REPORT)
print("CSV report:", CSV_REPORT)
print("=" * 100)