


import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Locate the official RD4AD source without modifying it
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "ADShift" / "rd4ad_baseline" / "source"

sys.path.insert(0, str(SOURCE_DIR))

from dataset import get_data_transforms
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rd4ad_loss(features, outputs):
    loss = 0.0

    for teacher_feature, decoder_feature in zip(features, outputs):
        loss += torch.mean(
            1 - torch.nn.functional.cosine_similarity(
                teacher_feature,
                decoder_feature,
                dim=1
            )
        )

    return loss

def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cpu":
        print("WARNING: CPU detected. Actual training is planned for Colab GPU.")

    train_dir = (
        Path(args.data_root) / args.category / "train"
    )

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training folder not found: {train_dir}"
        )

    data_transform, _ = get_data_transforms(
        args.image_size, args.image_size
    )

    train_dataset = ImageFolder(
        root=str(train_dir),
        transform=data_transform
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    print("Category:", args.category)
    print("Training images:", len(train_dataset))
    print("Training classes:", train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    # Official RD4AD teacher and BN module
    encoder, bn = wide_resnet50_2(pretrained=True)

    encoder = encoder.to(device)
    bn = bn.to(device)

    # Freeze teacher parameters
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    # Official RD4AD decoder
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(
        list(decoder.parameters()) + list(bn.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_checkpoint = (
        checkpoint_dir / f"{args.category}_latest.pth"
    )

    for epoch in range(args.epochs):
        encoder.eval()
        bn.train()
        decoder.train()

        loss_values = []

        for images, _ in train_loader:
            images = images.to(device)

            # Teacher feature extraction without gradients
            with torch.no_grad():
                features = encoder(images)

            outputs = decoder(bn(features))

            loss = rd4ad_loss(features, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())

        average_loss = float(np.mean(loss_values))

        print(
            f"Epoch [{epoch + 1}/{args.epochs}], "
            f"Loss: {average_loss:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "category": args.category,
            "bn": bn.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": average_loss
        }

        # Save latest checkpoint after every epoch
        torch.save(checkpoint, latest_checkpoint)

    final_checkpoint = (
        checkpoint_dir / f"{args.category}_final.pth"
    )

    torch.save(checkpoint, final_checkpoint)

    print("Training completed.")
    print("Latest checkpoint:", latest_checkpoint)
    print("Final checkpoint:", final_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--category", default="bottle")
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "datasets" / "mvtec")
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(PROJECT_ROOT / "checkpoints" / "rd4ad")
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=111)

    args = parser.parse_args()
    train(args)
