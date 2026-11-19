# Member 3 — EFDM Results Addendum

Generated: 2026-11-19T05:42:13.075547+00:00

## Comparison CSV inventory

### `standard_vs_efdm_final_comparison.csv`
- Rows: 15
- Columns: category, standard_pixel_auroc, efdm_pixel_auroc, pixel_change, standard_image_auroc, efdm_image_auroc, image_change, test_images
- Macro mean `standard_pixel_auroc`: 0.966469
- Macro mean `efdm_pixel_auroc`: 0.971619
- Macro mean `pixel_change`: 0.005150
- Macro mean `standard_image_auroc`: 0.959216
- Macro mean `efdm_image_auroc`: 0.973364
- Macro mean `image_change`: 0.014148

### `standard_vs_efdm_aligned.csv`
- Rows: 15
- Columns: category, test_images, standard_pixel_auroc, aligned_efdm_pixel_auroc, pixel_change, standard_image_auroc, aligned_efdm_image_auroc, image_change
- Macro mean `standard_pixel_auroc`: 0.966469
- Macro mean `aligned_efdm_pixel_auroc`: 0.971619
- Macro mean `pixel_change`: 0.005150
- Macro mean `standard_image_auroc`: 0.959216
- Macro mean `aligned_efdm_image_auroc`: 0.973364
- Macro mean `image_change`: 0.014148

### `efdm_gnl_all_categories.csv`
- Rows: 15
- Columns: category, test_images, pixel_auroc, image_auroc, map_shape, maps_valid, test_count_match
- Macro mean `pixel_auroc`: 0.971576
- Macro mean `image_auroc`: 0.973364

## Per-category EFDM JSON metrics

- Category JSON records found: 15

- Macro `pixel_auroc` across valid JSON records: 0.971576 (15 valid records)
- Macro `image_auroc` across valid JSON records: 0.973364 (15 valid records)

## RD4AD source discovery

- Likely source files found: 0
- Checkpoint directories found: 15

Checkpoint files are not assumed to be RD4AD until their model architecture and state-dictionary keys are verified.

## Interpretation caution

The standard-vs-EFDM comparison CSVs are retained as the primary comparison evidence. Do not combine separate CSV summaries unless their test-image sets, category coverage, and metric definitions align.
No models were retrained in this task.