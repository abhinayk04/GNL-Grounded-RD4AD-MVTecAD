\# MVTec AD Dataset Verification Report



\*\*Project:\*\* Reliable Anomaly Detection under Distribution Shift and Novel Defect Conditions

\*\*Responsibility:\*\* Member 2 – Data \& Evaluation Lead

\*\*Verification date:\*\* 16 November 2026



\## 1. Dataset source



Dataset: MVTec Anomaly Detection (MVTec AD), original 15-category version.



Official source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad



The dataset was downloaded as `mvtec\_anomaly\_detection.tar.xz` and extracted locally.



Dataset licence: CC BY-NC-SA 4.0. The dataset must not be uploaded to the project Git repository.



\## 2. Local dataset location



```text

C:\\Users\\chara\\DL\_PROJECT\\GNL-Grounded-RD4AD-MVTecAD\\datasets

```



The dataset is stored locally on Windows. Dataset files are excluded from Git through `.gitignore`.



\## 3. Dataset structure



Each category follows this general structure:



```text

datasets/

└── bottle/

&#x20;   ├── train/

&#x20;   │   └── good/

&#x20;   ├── test/

&#x20;   │   ├── good/

&#x20;   │   └── defect\_type/

&#x20;   └── ground\_truth/

&#x20;       └── defect\_type/

```



The same general structure is present for all 15 categories.



\## 4. Category-wise image counts



| Category | Train/good | Test/good | Test defects |

|---|---:|---:|---:|

| bottle | 209 | 20 | 63 |

| cable | 224 | 58 | 92 |

| capsule | 219 | 23 | 109 |

| carpet | 280 | 28 | 89 |

| grid | 264 | 21 | 57 |

| hazelnut | 391 | 40 | 70 |

| leather | 245 | 32 | 92 |

| metal\_nut | 220 | 22 | 93 |

| pill | 267 | 26 | 141 |

| screw | 320 | 41 | 119 |

| tile | 230 | 33 | 84 |

| toothbrush | 60 | 12 | 30 |

| transistor | 213 | 60 | 40 |

| wood | 247 | 19 | 60 |

| zipper | 240 | 32 | 119 |

| \*\*Total\*\* | \*\*3,642\*\* | \*\*435\*\* | \*\*1,253\*\* |



Total test images: 1,688.



\## 5. Data loader implementation



Implementation file:



`evaluation/mvtec\_loader.py`



The loader:

\- Supports all 15 MVTec categories.

\- Uses only `train/good` for training.

\- Loads normal and defective test images.

\- Loads corresponding ground-truth masks for defective images.

\- Returns image tensors, binary masks, image labels and defect types.

\- Resizes images and masks to 256 × 256 during loading.



Original dataset files are not permanently modified.



\## 6. Verification performed



The following checks passed:



1\. All 15 category directories verified.

2\. Training and test directories verified.

3\. All training images successfully opened and transformed.

4\. All test images successfully opened and transformed.

5\. All test masks successfully opened and transformed.

6\. Tensor dimensions verified:

&#x20;  - Image: \[3, 256, 256]

&#x20;  - Mask: \[1, 256, 256]



Full image and mask loading test: PASS for all 15 categories.



\## 7. Evaluation implementation



Implementation file:



`evaluation/metrics.py`



Implemented functions:

\- Image-level AUROC.

\- Pixel-level AUROC.

\- AUPRO.



Test file:



`evaluation/test\_metrics.py`



All three functions passed the current controlled synthetic tests.



\*\*Important:\*\* Synthetic test success does not establish benchmark compatibility. AUPRO threshold sampling and integration conventions still require protocol validation. Real model evaluation has not yet been performed.



\## 8. Corrupted datasets



The following corruption types are planned:

\- Brightness.

\- Contrast.

\- Defocus blur.

\- Gaussian noise.



These corrupted datasets have not yet been generated.



\## 9. Current readiness



| Component | Status |

|---|---|

| Original MVTec AD download | Complete |

| Extraction | Complete |

| All 15 categories | Verified |

| Dataset loader | Implemented and tested |

| Full image/mask reading | Passed |

| Initial evaluation metrics | Implemented |

| Synthetic metric tests | Passed |

| Benchmark protocol validation | Pending |

| Corrupted dataset generation | Pending |

| RD4AD/GNL baseline results | Pending |

| PCA experiments | Pending |



\## 10. Team handoff



Member 1 can use the verified dataset root and loader for RD4AD/GNL training setup.



Member 3 can prepare PCA integration, subject to agreement on the model feature interface and availability of normal training features.



The dataset itself must remain outside Git. Only code and documentation should be shared through the repository.



\*\*No real anomaly detection performance results are claimed in this report.\*\*
