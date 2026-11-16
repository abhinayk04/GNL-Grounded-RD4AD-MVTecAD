
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecDataset(Dataset):
    """
    MVTec AD dataset loader.

    Expected structure:
        root/
            bottle/
                train/good/
                test/good/
                test/<defect_type>/
                ground_truth/<defect_type>/

    Returns:
        image:       [3, H, W] normalized tensor
        mask:        [1, H, W] binary tensor
        label:       0 for normal, 1 for defective
        defect_type: "good" or defect category
    """

    def __init__(self, root, category, phase="train", image_size=256):
        self.root = Path(root)
        self.category = category
        self.phase = phase
        self.image_size = image_size

        if category not in MVTEC_CATEGORIES:
            raise ValueError("Unknown MVTec category: " + category)

        if phase not in ("train", "test"):
            raise ValueError("phase must be 'train' or 'test'")

        self.category_root = self.root / category
        self.image_root = self.category_root / phase
        self.ground_truth_root = self.category_root / "ground_truth"

        if not self.image_root.is_dir():
            raise FileNotFoundError(
                "Image directory not found: " + str(self.image_root)
            )

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        defect_folders = sorted(
            p for p in self.image_root.iterdir() if p.is_dir()
        )

        if self.phase == "train":
            good_folder = self.image_root / "good"

            if not good_folder.is_dir():
                raise FileNotFoundError(
                    "Normal training folder missing: " + str(good_folder)
                )

            defect_folders = [good_folder]

        for defect_folder in defect_folders:
            defect_type = defect_folder.name

            image_paths = sorted(
    p for p in defect_folder.iterdir()
    if p.is_file() and p.suffix.lower() == ".png"
)

            for image_path in image_paths:
                if defect_type == "good":
                    mask_path = None
                    label = 0
                else:
                    if self.phase == "train":
                        continue

                    mask_path = (
                        self.ground_truth_root
                        / defect_type
                        / (image_path.stem + "_mask.png")
                    )

                    if not mask_path.is_file():
                        raise FileNotFoundError(
                            "Ground-truth mask missing for: "
                            + str(image_path)
                        )

                    label = 1

                self.samples.append(
                    (image_path, mask_path, label, defect_type)
                )

        if not self.samples:
            raise RuntimeError(
                "No images found in: " + str(self.image_root)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, label, defect_type = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        if mask_path is None:
            mask = torch.zeros(
                (1, self.image_size, self.image_size),
                dtype=torch.float32
            )
        else:
            mask_image = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask_image)
            mask = (mask > 0.5).float()

        return image, mask, label, defect_type


def validate_mvtec_root(root):
    """Check whether all 15 MVTec category folders are available."""

    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError("MVTec root not found: " + str(root))

    available = []
    missing = []

    for category in MVTEC_CATEGORIES:
        category_path = root / category

        if (
            (category_path / "train" / "good").is_dir()
            and (category_path / "test").is_dir()
            and (category_path / "ground_truth").is_dir()
        ):
            available.append(category)
        else:
            missing.append(category)

    print("MVTec dataset verification")
    print("--------------------------")
    print("Available categories:", len(available), "/ 15")
    print("Available:", available)
    print("Missing or incomplete:", missing)

    return len(available) == 15
