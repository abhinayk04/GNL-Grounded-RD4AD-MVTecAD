# Member 3 — Final Experimental Results

## Scope

This report consolidates the completed PCA, GNL and EFDM-GNL experiment
artifacts available in the project workspace.

The corrected EFDM-GNL evaluation uses the normal reference image
`train/good/000.png` for each category. The evaluation did not retrain
or modify trained checkpoints.

## Corrected macro results

| Metric | Baseline GNL | EFDM-GNL | Difference |
|---|---:|---:|---:|
| Image AUROC | 0.955807 | 0.970917 | +0.015109 |
| Pixel AUROC | 0.965814 | 0.970607 | +0.004793 |
| AUPRO | 0.900749 | 0.927038 | +0.026290 |

## Category-level results

| category   |   baseline_image_auroc |   efdm_gnl_image_auroc |   baseline_pixel_auroc |   efdm_gnl_pixel_auroc |   baseline_aupro |   efdm_gnl_aupro |
|:-----------|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------:|-----------------:|
| bottle     |               0.993651 |               0.993651 |               0.982855 |               0.98187  |         0.959901 |         0.958925 |
| cable      |               0.946027 |               0.949025 |               0.976072 |               0.974801 |         0.927427 |         0.926774 |
| capsule    |               0.937375 |               0.940566 |               0.985233 |               0.984588 |         0.914112 |         0.912971 |
| carpet     |               0.991172 |               0.995185 |               0.991443 |               0.992058 |         0.96503  |         0.968608 |
| grid       |               0.771094 |               0.954052 |               0.852648 |               0.97177  |         0.497174 |         0.927326 |
| hazelnut   |               1        |               1        |               0.990586 |               0.987897 |         0.958765 |         0.954684 |
| leather    |               1        |               1        |               0.99479  |               0.993939 |         0.986408 |         0.985527 |
| metal_nut  |               0.999511 |               0.999511 |               0.976684 |               0.973986 |         0.951741 |         0.94751  |
| pill       |               0.969995 |               0.96263  |               0.966667 |               0.948991 |         0.957417 |         0.950431 |
| screw      |               0.926009 |               0.932773 |               0.989762 |               0.988698 |         0.954881 |         0.952078 |
| tile       |               0.992424 |               0.998196 |               0.959551 |               0.950433 |         0.878922 |         0.872159 |
| toothbrush |               0.963889 |               0.988889 |               0.988443 |               0.988222 |         0.900183 |         0.894561 |
| transistor |               0.969583 |               0.975    |               0.897383 |               0.896692 |         0.788316 |         0.786316 |
| wood       |               0.990351 |               0.987719 |               0.956961 |               0.947559 |         0.931341 |         0.928348 |
| zipper     |               0.886029 |               0.886555 |               0.978123 |               0.9776   |         0.939614 |         0.939357 |

## Category outcome counts

- Image AUROC: 9 improved,
  2 declined,
  4 unchanged.
- Pixel AUROC: 2 improved,
  13 declined,
  0 unchanged.
- AUPRO: 2 improved,
  13 declined,
  0 unchanged.

## Interpretation

The corrected EFDM-GNL macro averages are higher than the baseline GNL
averages for all three reported metrics. These macro improvements do not
imply that every category improves.

Grid has a particularly large AUPRO increase and materially influences
the macro AUPRO change. The category-wise results therefore need to be
considered alongside the macro averages.

## Methodological notes

- Dataset: MVTec AD, 15 categories.
- Corrected normal reference: `train/good/000.png`.
- AUPRO maximum false-positive rate: 0.30.
- AUPRO threshold count: 200.
- Connected-component connectivity: 8-connected.
- The corrected AUROC and AUPRO results are loaded from separate saved
  category-level evaluation CSVs.
- PCA and GNL source CSVs are identified separately; their metrics are
  not substituted for the corrected EFDM-GNL measurements.

## Reproducibility artifacts

- Corrected EFDM-GNL AUROC:
  `docs/member3_batch11y_efdm_gnl_corrected_all15.json`
- Corrected EFDM-GNL AUPRO:
  `docs/member3_batch11z_aupro_corrected_all15.json`
- Consolidated CSV:
  `results/member3/final_consolidated/member3_final_consolidated_all15.csv`
- Consolidated JSON:
  `results/member3/final_consolidated/member3_final_consolidated_all15.json`

## Limitations

The macro values are unweighted means of the 15 category-level scores.
They are not pooled image-level or pixel-level metrics. Category-level
variability should be retained when interpreting the results.
