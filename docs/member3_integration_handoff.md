# Member 3 - PCA Integration Handoff

## Implementation
PCA nuisance-subspace filtering is implemented in:
- pca_method/pca_config.py
- pca_method/compute_nuisance_pca.py
- pca_method/inference_pca_gating.py
- pca_method/integration_interface.py

## Reproduction artifacts
PCA bases:
`GNL_Project_Backup/pca_reproduction_v1/bases/`

Bottle/Cable evaluation:
`GNL_Project_Backup/pca_reproduction_v1/evaluation_run_01/`

All-category evaluation:
See the `all_categories_run_*` directory created by the final pipeline.

## Inference method
The original GNL inference script remains unchanged.
PCA-GNL uses orthogonal-complement projection at all three feature levels,
followed by the original GNL cosine-distance anomaly-map protocol.

## Check before submission
1. Confirm all 15 category result entries exist.
2. Inspect baseline reproduction checks.
3. Verify all 45 PCA bases exist and are finite/orthonormal.
4. Include the generated CSV and JSON reports.
5. Keep the original GNL checkpoints and baseline results unchanged.

## Interpretation
Report per-category pixel and image AUROC changes. Do not describe PCA as
universally improving GNL if results are mixed.
