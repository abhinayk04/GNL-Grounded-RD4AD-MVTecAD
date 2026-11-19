# Member 3 Task 1R — EFDM and checkpoint recovery audit

## Summary
- Notebooks scanned: 2
- Relevant notebook cells: 54
- Grid map: [78, 256, 256] (float32)
- Grid map ordering remains unverified unless original per-image provenance is recovered.
- Checkpoint key patterns are recorded for all available categories.
- No checkpoint tensors were loaded and no model was modified.

## Checkpoint signature groups
- Categories: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
  - Prefix signature: `{'bn1': 5, 'bn2': 5, 'bn3': 5, 'bn4': 5, 'bn_layer': 87, 'conv1': 1, 'conv2': 1, 'conv3': 1, 'conv4': 1, 'layer1': 87, 'layer2': 113, 'layer3': 165}`

## Interpretation
The grid anomaly-map array has no intrinsic filename metadata. A current dataset ordering
cannot prove the original map-to-image mapping. Do not publish the stored grid metric as
independently reproduced until the original inference order or a filename manifest is found.

Checkpoint key-name patterns are evidence about likely model structure, not proof of
RD4AD compatibility. Do not run RD4AD evaluation against these checkpoints until the
expected architecture and checkpoint loading contract are established.

## Detailed evidence
See `member3_task1r_recovery_audit.json` for notebook cells, source contexts, checkpoint fingerprints, and sidecar candidates.
