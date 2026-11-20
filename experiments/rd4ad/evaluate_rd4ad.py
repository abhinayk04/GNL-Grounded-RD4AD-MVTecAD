
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "ADShift" / "rd4ad_baseline" / "source"

# Add project root so shared evaluation package can be imported
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SOURCE_DIR))

# Shared Member 2 evaluation utilities
from evaluation.mvtec_loader import MVTecDataset
from evaluation.metrics import compute_all_metrics

# Official RD4AD architecture
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2


def cal_anomaly_map(features, outputs, out_size):
    """Combine feature-level cosine distances into one anomaly map."""

    anomaly_map = np.zeros((out_size, out_size), dtype=np.float32)

    for teacher_feature, decoder_feature in zip(features, outputs):
        distance = 1 - F.cosine_similarity(
            teacher_feature, decoder_feature, dim=1
        )

        distance = distance.unsqueeze(1)

        distance = F.interpolate(
            distance,
            size=out_size,
            mode="bilinear",
            align_corners=True
        )

        anomaly_map += distance[0, 0].detach().cpu().numpy()

    return anomaly_map


def load_checkpoint(checkpoint_path, device):
    """Load a trusted RD4AD checkpoint."""

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

    return checkpoint


def evaluate(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Category: {args.category}")

    # Check dataset directory
    data_root = Path(args.data_root)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_root}"
        )

    # Load shared MVTec test dataset
    test_dataset = MVTecDataset(
        root=data_root,
        category=args.category,
        phase="test",
        image_size=args.image_size
    )

    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"Test images: {len(test_dataset)}")

    # Initialize official RD4AD models
    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)

    # Load trained checkpoint
    checkpoint_path = (
        Path(args.checkpoint_dir)
        / f"{args.category}_final.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = load_checkpoint(checkpoint_path, device)

    bn.load_state_dict(checkpoint["bn"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    bn.eval()
    decoder.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    # Store predictions and ground truth
    image_labels = []
    image_scores = []
    all_masks = []
    all_anomaly_maps = []
    defect_types = []

    # Inference
    with torch.inference_mode():

        for index, batch in enumerate(test_loader):

            images, masks, labels, defect_type = batch

            images = images.to(device)

            # RD4AD forward pass
            features = encoder(images)
            outputs = decoder(bn(features))

            # Generate anomaly map
            anomaly_map = cal_anomaly_map(
                features,
                outputs,
                args.image_size
            )

            # Gaussian smoothing
            anomaly_map = gaussian_filter(
                anomaly_map,
                sigma=4
            ).astype(np.float32)

            # Image-level score
            image_score = float(anomaly_map.max())

            image_labels.append(int(labels.item()))
            image_scores.append(image_score)

            all_masks.append(masks.cpu().numpy())
            all_anomaly_maps.append(anomaly_map)

            defect_types.extend(list(defect_type))

            if (index + 1) % 50 == 0:
                print(
                    f"Processed {index + 1}/{len(test_dataset)} images"
                )

    # Convert collected data into NumPy arrays
    image_labels = np.asarray(image_labels, dtype=np.int64)
    image_scores = np.asarray(image_scores, dtype=np.float32)

    all_masks = np.concatenate(all_masks, axis=0)
    all_anomaly_maps = np.stack(all_anomaly_maps, axis=0)

    defect_types = np.asarray(defect_types, dtype=str)

    # Calculate shared evaluation metrics
    metrics = compute_all_metrics(
        image_labels,
        image_scores,
        all_masks,
        all_anomaly_maps
    )

    # Create results directory
    results_dir = (
        Path(args.results_dir) / "rd4ad"
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = (
        results_dir / f"{args.category}_metrics.txt"
    )

    with open(metrics_file, "w", encoding="utf-8") as file:

        file.write(f"RD4AD Results - {args.category}\n")
        file.write("=" * 40 + "\n")

        for metric_name, metric_value in metrics.items():
            file.write(
                f"{metric_name}: {metric_value:.6f}\n"
            )

    # Export per-image anomaly maps and scores
    output_file = (
        results_dir
        / f"{args.category}_anomaly_outputs.npz"
    )

    np.savez_compressed(
        output_file,
        anomaly_maps=all_anomaly_maps,
        image_scores=image_scores,
        image_labels=image_labels,
        masks=all_masks,
        defect_types=defect_types
    )

    print("\nRD4AD Evaluation Results")
    print("=" * 40)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")

    print("\nSaved files:")
    print(metrics_file)
    print(output_file)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="RD4AD evaluation on MVTec AD"
    )

    parser.add_argument(
        "--category",
        type=str,
        default="bottle"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "mvtec")
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "rd4ad")
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "results")
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0
    )

    args = parser.parse_args()

    evaluate(args)
