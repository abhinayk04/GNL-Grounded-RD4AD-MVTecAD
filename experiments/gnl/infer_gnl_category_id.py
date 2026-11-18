import os
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from PIL import Image

from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from dataset import MVTecDataset, get_data_transforms


def calculate_anomaly_map(teacher_features, student_features, out_size=256):
    anomaly_map = np.zeros((out_size, out_size), dtype=np.float32)

    for teacher, student in zip(teacher_features, student_features):
        distance = 1 - F.cosine_similarity(teacher, student)
        distance = distance.unsqueeze(1)

        distance = F.interpolate(
            distance,
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=True
        )

        anomaly_map += distance[0, 0].cpu().numpy()

    return gaussian_filter(anomaly_map, sigma=4)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    map_dir = os.path.join(args.output_dir, "anomaly_maps")
    os.makedirs(map_dir, exist_ok=True)

    print("Device:", device)
    print("Category:", args.category)
    print("Dataset:", args.dataset_root)
    print("Checkpoint:", args.checkpoint)

    image_transform, mask_transform = get_data_transforms(256, 256)

    test_dataset = MVTecDataset(
        root=os.path.join(args.dataset_root, args.category),
        transform=image_transform,
        gt_transform=mask_transform,
        phase="test"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    print("Test images:", len(test_dataset))

    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )

    if checkpoint.get("category") != args.category:
        raise ValueError(
            "Checkpoint category mismatch: expected {}, got {}".format(
                args.category, checkpoint.get("category")
            )
        )

    bn.load_state_dict(checkpoint["bn"], strict=True)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)

    encoder = encoder.to(device).eval()
    bn = bn.to(device).eval()
    decoder = decoder.to(device).eval()

    print("Loaded checkpoint epoch:", checkpoint["epoch"])

    image_labels, image_scores = [], []
    pixel_labels, pixel_scores = [], []
    records = []

    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            images, masks, labels, defect_types = batch
            images = images.to(device)

            teacher_features = encoder(images)
            student_features = decoder(bn(teacher_features))

            anomaly_map = calculate_anomaly_map(
                teacher_features, student_features, out_size=256
            )

            label = int(labels.item())
            defect_type = defect_types[0]
            filename = os.path.basename(test_dataset.img_paths[index])
            image_score = float(np.max(anomaly_map))

            image_labels.append(label)
            image_scores.append(image_score)

            mask = masks[0, 0].cpu().numpy()
            mask_binary = (mask > 0.5).astype(np.uint8)

            pixel_labels.extend(mask_binary.reshape(-1).tolist())
            pixel_scores.extend(anomaly_map.reshape(-1).tolist())

            map_name = "{:03d}_{}_{}".format(
                index, defect_type, filename
            )

            np.save(
                os.path.join(map_dir, map_name.replace(".png", ".npy")),
                anomaly_map
            )

            map_min, map_max = anomaly_map.min(), anomaly_map.max()
            normalized_map = (
                (anomaly_map - map_min) / (map_max - map_min + 1e-8)
            )

            heatmap = Image.fromarray(
                (normalized_map * 255).astype(np.uint8)
            )
            heatmap.save(
                os.path.join(
                    map_dir, map_name.replace(".png", "_heatmap.png")
                )
            )

            records.append({
                "filename": filename,
                "defect_type": defect_type,
                "label": label,
                "image_score": image_score
            })

            if (index + 1) % 10 == 0 or index + 1 == len(test_dataset):
                print("Processed: {}/{}".format(
                    index + 1, len(test_dataset)
                ))

    image_auroc = roc_auc_score(image_labels, image_scores)
    pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)

    with open(os.path.join(args.output_dir, "image_scores.csv"), "w",
              newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "filename", "defect_type", "label", "image_score"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    with open(os.path.join(args.output_dir, "evaluation_summary.txt"), "w") as f:
        f.write("GNL {} ID Evaluation\n".format(args.category))
        f.write("========================\n")
        f.write("Checkpoint epoch: {}\n".format(checkpoint["epoch"]))
        f.write("Test images: {}\n".format(len(test_dataset)))
        f.write("Normal images: {}\n".format(image_labels.count(0)))
        f.write("Anomalous images: {}\n".format(image_labels.count(1)))
        f.write("Image-level AUROC: {:.6f}\n".format(image_auroc))
        f.write("Pixel-level AUROC: {:.6f}\n".format(pixel_auroc))

    print("\n===== EVALUATION RESULTS =====")
    print("Category:", args.category)
    print("Test images:", len(test_dataset))
    print("Normal:", image_labels.count(0))
    print("Anomalous:", image_labels.count(1))
    print("Image-level AUROC:", round(image_auroc, 6))
    print("Pixel-level AUROC:", round(pixel_auroc, 6))
    print("Results saved:", args.output_dir)


if __name__ == "__main__":
    main()
