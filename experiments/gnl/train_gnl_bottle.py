import os
import csv
import argparse
import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from dataset import AugMixDatasetMVTec, mean_train, std_train


def loss_function(a, b):
    cosine = torch.nn.CosineSimilarity(dim=1)
    loss = 0.0

    for i in range(len(a)):
        loss += torch.mean(
            1 - cosine(
                a[i].flatten(1),
                b[i].flatten(1)
            )
        )

    return loss


def loss_function_last(a, b):
    cosine = torch.nn.CosineSimilarity(dim=1)

    # Preserve official ADShift implementation: index 0.
    return torch.mean(
        1 - cosine(
            a[0].flatten(1),
            b[0].flatten(1)
        )
    )


def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Category: bottle")
    print("Dataset:", args.dataset_root)

    # Official ADShift image preprocessing.
    resize_transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_train,
            std=std_train
        )
    ])

    # Only normal training images are used.
    train_path = os.path.join(
        args.dataset_root,
        "train"
    )

    base_dataset = ImageFolder(
        root=train_path,
        transform=resize_transform
    )

    if len(base_dataset) == 0:
        raise RuntimeError("No training images found.")

    print("ImageFolder classes:", base_dataset.classes)
    print("Normal training images:", len(base_dataset))

    # Official DINL augmentation wrapper.
    train_dataset = AugMixDatasetMVTec(
        base_dataset,
        preprocess
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    # Pretrained teacher and attention bottleneck.
    encoder, bn = wide_resnet50_2(pretrained=True)

    encoder = encoder.to(device)
    bn = bn.to(device)

    # Explicitly freeze the teacher.
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    # Trainable decoder.
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    # Only bottleneck and decoder are optimized.
    optimizer = torch.optim.Adam(
        list(bn.parameters()) +
        list(decoder.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    log_path = os.path.join(
        args.results_dir,
        "training_log.csv"
    )

    with open(log_path, "w", newline="") as log_file:

        writer = csv.writer(log_file)

        writer.writerow([
            "epoch",
            "loss_total",
            "loss_normal",
            "loss_bn",
            "loss_last"
        ])

        for epoch in range(args.epochs):

            encoder.eval()
            bn.train()
            decoder.train()

            epoch_losses = []
            normal_losses = []
            bn_losses = []
            last_losses = []

            for batch_index, (
                normal,
                augmix_img,
                gray_img
            ) in enumerate(train_loader):

                if args.max_batches > 0 and batch_index >= args.max_batches:
                    break

                normal = normal.to(device)
                augmix_img = augmix_img.to(device)
                gray_img = gray_img.to(device)

                # Teacher feature extraction without gradients.
                with torch.no_grad():
                    inputs_normal = encoder(normal)
                    inputs_augmix = encoder(augmix_img)
                    inputs_gray = encoder(gray_img)

                # Trainable bottleneck and decoder.
                bn_normal = bn(inputs_normal)
                outputs_normal = decoder(bn_normal)

                bn_augmix = bn(inputs_augmix)
                outputs_augmix = decoder(bn_augmix)

                bn_gray = bn(inputs_gray)
                outputs_gray = decoder(bn_gray)

                # Official DINL losses.
                loss_bn = (
                    loss_function(
                        [bn_normal],
                        [bn_augmix]
                    )
                    +
                    loss_function(
                        [bn_normal],
                        [bn_gray]
                    )
                )

                loss_last = (
                    loss_function_last(
                        outputs_normal,
                        outputs_augmix
                    )
                    +
                    loss_function_last(
                        outputs_normal,
                        outputs_gray
                    )
                )

                loss_normal = loss_function(
                    inputs_normal,
                    outputs_normal
                )

                loss = (
                    0.90 * loss_normal
                    + 0.05 * loss_bn
                    + 0.05 * loss_last
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite loss encountered."
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                normal_losses.append(loss_normal.item())
                bn_losses.append(loss_bn.item())
                last_losses.append(loss_last.item())

            if not epoch_losses:
                raise RuntimeError("No training batches processed.")

            averages = [
                float(np.mean(values))
                for values in [
                    epoch_losses,
                    normal_losses,
                    bn_losses,
                    last_losses
                ]
            ]

            writer.writerow([epoch + 1] + averages)
            log_file.flush()

            print(
                "Epoch {}/{} | Total: {:.6f} | "
                "Normal: {:.6f} | BN: {:.6f} | "
                "Decoder: {:.6f}".format(
                    epoch + 1,
                    args.epochs,
                    *averages
                )
            )

            # Save one resumable checkpoint after every epoch.
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                "latest.pth"
            )
            temporary_path = checkpoint_path + ".tmp"

            torch.save({
                "epoch": epoch + 1,
                "category": "bottle",
                "bn": bn.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": averages[0],
                "config": vars(args)
            }, temporary_path)

            os.replace(temporary_path, checkpoint_path)
            print("Latest checkpoint saved:", checkpoint_path)

            if device.type == "cuda":
                print(
                    "Peak GPU memory: {:.3f} GB".format(
                        torch.cuda.max_memory_allocated()
                        / (1024 ** 3)
                    )
                )

    print("\nTraining completed.")
    print("Training log:", log_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        default="/content/mvtec_dataset/bottle"
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-batches", type=int, default=0)

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/bottle"
    )

    parser.add_argument(
        "--results-dir",
        default="results/bottle"
    )

    main(parser.parse_args())
