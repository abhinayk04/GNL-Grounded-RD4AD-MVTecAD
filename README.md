# GNL-Grounded RD4AD for MVTec AD

This repository contains the implementation, evaluation protocols, and nuisance subspace filtering methods for anomaly detection and localization on the **MVTec Anomaly Detection (MVTec AD)** benchmark.

The project investigates three interconnected components:
1. **RD4AD Baseline**: Reverse Distillation for Anomaly Detection (pretrained frozen encoder, teacher bottleneck, and multi-scale student decoder).
2. **GNL (Generalized Normality Learning)**: Normal data augmentation with multi-level consistency losses (bottleneck & low-frequency decoder consistency).
3. **PCA-Gated Nuisance Filtering**: Post-hoc nuisance subspace projection estimated from content-preserving transformations to filter out shift-induced residuals during inference.

---

## Table of Contents

- [Python Version & Prerequisites](#python-version--prerequisites)
- [Installation & Dependency Setup](#installation--dependency-setup)
- [Dataset Download & Placement](#dataset-download--placement)
- [Repository Folder Structure](#repository-folder-structure)
- [Main Files to Run](#main-files-to-run)
- [Commands to Train, Test, and Run](#commands-to-train-test-and-run)
  - [1. Sanity Test / Metric Suite](#1-sanity-test--metric-suite)
  - [2. Train RD4AD Model](#2-train-rd4ad-model)
  - [3. Evaluate RD4AD Model](#3-evaluate-rd4ad-model)
  - [4. Run PCA Nuisance Filtering](#4-run-pca-nuisance-filtering)
  - [5. Run Standalone Baseline (ADShift Source)](#5-run-standalone-baseline-adshift-source)
- [Evaluation Metrics & Benchmark Results](#evaluation-metrics--benchmark-results)

---

## Python Version & Prerequisites

- **Python Version**: **Python 3.10, 3.11, or 3.12** (Python 3.12 recommended and verified).
- **Hardware**:
  - **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 3080/3090/4090, RTX A6000, or Google Colab T4/V100/A100) recommended for model training.
  - **CPU**: CPU is supported for metric validation, evaluation, and PCA basis computations.
- **Operating System**: Linux, macOS, or Windows 10/11.

---

## Installation & Dependency Setup

### 1. Clone or Extract the Repository

Ensure you are in the repository root directory:

```bash
cd GNL-Grounded-RD4AD-MVTecAD-main
```

### 2. Set Up a Virtual Environment

#### Option A: Using `venv` (Standard Python)

```bash
# Create a virtual environment named 'env'
python -m venv env

# Activate the virtual environment
# On Linux / macOS:
source env/bin/activate
# On Windows (Command Prompt):
env\Scripts\activate.bat
# On Windows (PowerShell):
.\env\Scripts\Activate.ps1
```

#### Option B: Using `conda`

```bash
# Create and activate conda environment
conda create -n gnl_rd4ad python=3.12 -y
conda activate gnl_rd4ad
```

### 3. Install PyTorch with CUDA (Recommended for GPU users)

Install the appropriate PyTorch build matching your system's CUDA version (or CPU-only if no GPU is available):

```bash
# For CUDA 12.1 / 12.4+:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU-only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Project Requirements

Install all remaining dependencies via `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Required Libraries Summary (`requirements.txt`)

| Package | Minimum Version | Purpose |
|---|---|---|
| `torch` | `>= 2.0.0` | Deep learning framework, tensor computations, autograd |
| `torchvision` | `>= 0.15.0` | Pretrained ResNet architectures, image transforms |
| `numpy` | `>= 1.22.0, < 2.0.0` | Numerical calculations, array formatting, array storage |
| `scipy` | `>= 1.7.0` | Gaussian smoothing filter, connected component labeling |
| `scikit-learn` | `>= 1.0.0` | AUROC computation (`roc_auc_score`) |
| `scikit-image` | `>= 0.19.0` | Image processing, region labeling |
| `pandas` | `>= 1.3.0` | Results organization and dataframes |
| `opencv-python` | `>= 4.5.0` | Anomaly heatmap rendering and visualization |
| `Pillow` | `>= 8.3.0` | Image reading, RGB/grayscale loading |
| `matplotlib` | `>= 3.4.0` | Metric curve plotting and visualization |

---

## Dataset Download & Placement

This project uses the official **MVTec Anomaly Detection (MVTec AD)** dataset.

- **Official Download URL**: [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Archive File**: `mvtec_anomaly_detection.tar.xz` (~4.9 GB)
- **License**: CC BY-NC-SA 4.0 (for non-commercial research purposes).

### 15 Supported Object & Texture Categories

- **Objects (10)**: `bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`
- **Textures (5)**: `carpet`, `grid`, `leather`, `tile`, `wood`

### Placement Instructions

1. Download `mvtec_anomaly_detection.tar.xz` from the official website.
2. Unpack the archive into the `datasets/mvtec/` folder within the repository.

```bash
# Example extraction on Linux/macOS:
mkdir -p datasets/mvtec
tar -xf mvtec_anomaly_detection.tar.xz -C datasets/mvtec/

# Example extraction on Windows (PowerShell):
# tar.exe -xf mvtec_anomaly_detection.tar.xz -C datasets/mvtec/
```

### Expected Directory Layout

Each category folder must contain `train`, `test`, and `ground_truth` subdirectories:

```text
GNL-Grounded-RD4AD-MVTecAD-main/
└── datasets/
    └── mvtec/
        ├── bottle/
        │   ├── train/
        │   │   └── good/                  # Defect-free training images (*.png)
        │   ├── test/
        │   │   ├── good/                  # Defect-free test images (*.png)
        │   │   ├── broken_large/          # Anomaly sample images (*.png)
        │   │   ├── broken_small/          # Anomaly sample images (*.png)
        │   │   └── contamination/         # Anomaly sample images (*.png)
        │   └── ground_truth/
        │       ├── broken_large/          # Binary anomaly masks (*_mask.png)
        │       ├── broken_small/          # Binary anomaly masks (*_mask.png)
        │       └── contamination/         # Binary anomaly masks (*_mask.png)
        ├── cable/
        ├── capsule/
        └── ... (other 12 categories)
```

> **Note**: Custom dataset directories can also be specified at runtime using the `--data-root` CLI argument (e.g. `--data-root /path/to/mvtec`).

---

## Repository Folder Structure

```text
GNL-Grounded-RD4AD-MVTecAD-main/
│
├── README.md                               # Project documentation & instructions
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git exclusion rules
│
├── ADShift/                                # Official baseline implementation source
│   ├── rd4ad_baseline/
│   │   └── source/
│   │       ├── dataset.py                  # Official PyTorch dataset & transforms
│   │       ├── de_resnet.py                # Reverse distillation student decoder
│   │       ├── main.py                     # Official baseline train/eval entry point
│   │       ├── readme.md                   # Upstream baseline notes
│   │       ├── resnet.py                   # Pretrained teacher encoder architectures
│   │       └── test.py                     # Baseline evaluation & visualization routines
│   └── .gitkeep
│
├── checkpoints/                            # Saved model weights (.pth)
│   ├── gnl/                                # GNL model checkpoints
│   ├── rd4ad/                              # RD4AD checkpoints (e.g. bottle_final.pth)
│   └── .gitkeep
│
├── datasets/                               # Dataset storage root (git-ignored)
│   ├── mvtec/                              # Extracted MVTec AD categories
│   └── .gitkeep
│
├── documentation/                          # Detailed technical reports & notes
│   ├── dataset_report/
│   │   └── mvtec_verification.md           # Dataset verification & image counts
│   ├── final_report/                       # Milestone and final report outputs
│   └── implementation_notes/
│       ├── member3_integration_handoff.md  # Member 3 PCA integration guide
│       └── member3_pca_implementation.md   # Mathematical derivation & PCA tests
│
├── evaluation/                             # Evaluation metrics & data loaders
│   ├── __init__.py
│   ├── compare_official_aupro.py           # Verification against official MVTec AUPRO
│   ├── metrics.py                          # Image AUROC, Pixel AUROC, and AUPRO
│   ├── mvtec_loader.py                     # MVTec dataset loader (images + masks)
│   └── test_metrics.py                     # Unit test suite for evaluation metrics
│
├── experiments/                            # Training & evaluation experimental scripts
│   ├── gnl/                                # GNL training scripts
│   ├── pca/                                # PCA experimental scripts
│   └── rd4ad/
│       ├── train_rd4ad.py                  # Main training runner for RD4AD
│       └── evaluate_rd4ad.py               # Main evaluation runner for RD4AD
│
├── pca_method/                             # Nuisance PCA subspace filtering
│   ├── __init__.py
│   ├── compute_nuisance_pca.py             # Computes and saves nuisance PCA basis
│   ├── inference_pca_gating.py             # Orthogonal projection for anomaly residuals
│   ├── integration_interface.py            # Clean interface for RD4AD/GNL pipelines
│   ├── pca_config.py                       # Configuration hyperparameters for PCA
│   └── saved_bases/                        # Serialized PCA basis checkpoints (.pth)
│
└── results/                                # Output evaluation metrics and heatmaps
    ├── comparison/                         # Model comparison summaries
    └── full_baseline/
        └── rd4ad/                          # Baseline metric logs across 15 categories
            ├── bottle_metrics.txt
            ├── cable_metrics.txt
            └── ... (all 15 categories)
```

---

## Main Files to Run

The key execution files and their roles are:

| File | Purpose | Main Command |
|---|---|---|
| **`experiments/rd4ad/train_rd4ad.py`** | **Primary training script** for RD4AD on MVTec AD classes. Freezes pretrained teacher encoder, trains reverse decoder with cosine similarity loss, and saves checkpoints. | `python experiments/rd4ad/train_rd4ad.py --category bottle` |
| **`experiments/rd4ad/evaluate_rd4ad.py`** | **Primary evaluation script**. Loads trained model checkpoint, processes test images, performs Gaussian smoothing, and computes Image AUROC, Pixel AUROC, and AUPRO. | `python experiments/rd4ad/evaluate_rd4ad.py --category bottle` |
| **`evaluation/test_metrics.py`** | **Metric unit test runner**. Validates correct mathematical calculation of Image AUROC, Pixel AUROC, and AUPRO with boundary assertions. | `python -m evaluation.test_metrics` |
| **`pca_method/compute_nuisance_pca.py`** | Computes nuisance PCA subspace basis $V \in \mathbb{R}^{C \times K}$ from teacher feature difference vectors under shift transforms and saves basis checkpoints. | Imported or executed via integration interface |
| **`pca_method/integration_interface.py`** | Module interface that takes teacher features, student features, and PCA basis, returning raw and nuisance-filtered anomaly residuals. | Interfaced into inference |
| **`ADShift/rd4ad_baseline/source/main.py`** | Original standalone reference implementation of RD4AD. | `python ADShift/rd4ad_baseline/source/main.py` |

---

## Commands to Train, Test, and Run

All commands should be executed from the repository root directory (`GNL-Grounded-RD4AD-MVTecAD-main`).

### 1. Sanity Test / Metric Suite

To verify that your Python environment, PyTorch, NumPy, SciPy, and Scikit-Learn installations are functioning properly, run the test metric suite:

```bash
python -m evaluation.test_metrics
```

Expected output:
```text
PASSED: Image AUROC
PASSED: Pixel AUROC
PASSED: Multiple images and defect regions
...
ALL EXPANDED METRIC TESTS PASSED
```

---

### 2. Train RD4AD Model

The training script trains the Reverse Distillation student decoder using normal training samples only. Pretrained `wide_resnet50_2` weights from torchvision are automatically downloaded on first run.

#### Basic Training (Default category: `bottle`):

```bash
python experiments/rd4ad/train_rd4ad.py --category bottle
```

#### Training with Custom Parameters:

```bash
python experiments/rd4ad/train_rd4ad.py \
  --category bottle \
  --epochs 200 \
  --batch-size 16 \
  --learning-rate 0.005 \
  --image-size 256 \
  --data-root datasets/mvtec \
  --checkpoint-dir checkpoints/rd4ad \
  --workers 4 \
  --seed 111
```

#### Training Command Arguments:

| Argument | Default | Description |
|---|---|---|
| `--category` | `bottle` | MVTec AD category name (e.g. `bottle`, `cable`, `capsule`, etc.) |
| `--data-root` | `datasets/mvtec` | Path to MVTec AD dataset root |
| `--checkpoint-dir` | `checkpoints/rd4ad` | Directory to save trained model checkpoints |
| `--epochs` | `200` | Number of training epochs |
| `--batch-size` | `16` | Batch size for training data loader |
| `--image-size` | `256` | Input image resolution (height and width) |
| `--learning-rate` | `0.005` | Adam optimizer learning rate ($\beta_1=0.5, \beta_2=0.999$) |
| `--workers` | `2` | Number of DataLoader worker threads |
| `--seed` | `111` | Random seed for reproducibility |

Outputs generated:
- `checkpoints/rd4ad/<category>_latest.pth` (updated every epoch)
- `checkpoints/rd4ad/<category>_final.pth` (final weights after training)

---

### 3. Evaluate RD4AD Model

The evaluation script loads the trained model checkpoint (`checkpoints/rd4ad/<category>_final.pth`), runs inference across all test samples (normal and anomalous), computes the anomaly score maps, and evaluates Image AUROC, Pixel AUROC, and AUPRO.

#### Basic Evaluation:

```bash
python experiments/rd4ad/evaluate_rd4ad.py --category bottle
```

#### Evaluation with Custom Parameters:

```bash
python experiments/rd4ad/evaluate_rd4ad.py \
  --category bottle \
  --data-root datasets/mvtec \
  --checkpoint-dir checkpoints/rd4ad \
  --results-dir results \
  --image-size 256 \
  --workers 0
```

#### Evaluation Command Arguments:

| Argument | Default | Description |
|---|---|---|
| `--category` | `bottle` | Category to evaluate |
| `--data-root` | `datasets/mvtec` | Path to MVTec AD dataset directory |
| `--checkpoint-dir` | `checkpoints/rd4ad` | Path to directory containing trained weights |
| `--results-dir` | `results` | Directory where metric results and predictions are saved |
| `--image-size` | `256` | Image resolution |
| `--workers` | `0` | Number of DataLoader workers |

Outputs generated:
- `results/rd4ad/<category>_metrics.txt`: Text file containing Image AUROC, Pixel AUROC, and AUPRO.
- `results/rd4ad/<category>_anomaly_outputs.npz`: Compressed NumPy archive containing predicted anomaly maps, image scores, binary ground-truth masks, and labels.

---

### 4. Run PCA Nuisance Filtering

The PCA nuisance gating method filters out shift-induced variance from the teacher-student feature residuals.

#### Integration in Python:

```python
import torch
from pca_method.integration_interface import pca_filtered_residual

# Load computed nuisance basis for the category (e.g., bottle)
checkpoint = torch.load("pca_method/saved_bases/bottle.pth", map_location="cpu")
basis = checkpoint["basis"]  # Shape: [C, K], e.g., [512, 16]

# During inference:
# teacher_features: [B, C, H, W]
# student_features: [B, C, H, W]
result = pca_filtered_residual(
    teacher_features=teacher_features,
    student_features=student_features,
    nuisance_basis=basis
)

raw_residual = result["raw_residual"]          # Original anomaly residual
filtered_residual = result["filtered_residual"] # Nuisance-subspace projected residual
```

---

### 5. Run Standalone Baseline (ADShift Source)

To run the upstream RD4AD baseline directly from `ADShift/rd4ad_baseline/source/`:

```bash
cd ADShift/rd4ad_baseline/source
python main.py
```

---

## Evaluation Metrics & Benchmark Results

### Evaluation Metrics Defined

1. **Image-level AUROC**: Evaluates classification performance (discriminating normal vs. defective images).
2. **Pixel-level AUROC**: Evaluates segmentation performance across all test image pixels.
3. **AUPRO (Area Under the Per-Region Overlap)**: Evaluates localization accuracy up to false positive rate $\text{FPR} \le 0.30$, treating all connected defect regions equally regardless of size.

### RD4AD Baseline Reference Results (MVTec AD)

| Category | Image AUROC | Pixel AUROC | AUPRO |
|---|:---:|:---:|:---:|
| **bottle** | **1.000** | **0.988** | **0.965** |
| **cable** | 0.942 | 0.970 | 0.902 |
| **capsule** | 0.979 | 0.988 | 0.941 |
| **carpet** | 0.989 | 0.989 | 0.967 |
| **grid** | 0.999 | 0.993 | 0.967 |
| **hazelnut** | 0.998 | 0.987 | 0.947 |
| **leather** | 1.000 | 0.993 | 0.982 |
| **metal_nut** | 0.996 | 0.971 | 0.923 |
| **pill** | 0.957 | 0.973 | 0.928 |
| **screw** | 0.959 | 0.993 | 0.964 |
| **tile** | 0.999 | 0.956 | 0.898 |
| **toothbrush** | 0.989 | 0.990 | 0.932 |
| **transistor** | 0.963 | 0.925 | 0.771 |
| **wood** | 0.990 | 0.952 | 0.903 |
| **zipper** | 0.987 | 0.983 | 0.951 |
| **Mean** | **0.983** | **0.977** | **0.930** |

All official baseline logs can be found in `results/full_baseline/rd4ad/`.
