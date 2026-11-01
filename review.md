# Review notes

## Scope

Reviewed the current changes to data loading and splitting, augmentations, model architecture, evaluation helpers, and webcam inference. The repository has no training script or automated test suite, and no FER data or checkpoints are present in this checkout, so reported model performance and end-to-end inference could not be verified here.

## Changes in this revision

- Added two-stage image deduplication, crowd-label filtering, optional sample weights, soft labels, and configurable stratified splits.
- Added rare-class-aware augmentation and mixup helpers.
- Added ECA and spatial attention modules, configurable backbone freezing, and evaluation/calibration utilities.
- Updated webcam inference to load old/new model state dicts and optional calibration settings.
- Removed tracked model checkpoints and training notebooks from the repository working tree; these assets are not required to inspect or use the source modules but training examples are no longer bundled.

## Review findings

### Fixed

- Rare-tie samples could previously be collected from FER+ `PublicTest` and `PrivateTest` rows and appended to the training set whenever `retain_rare_ties_in_train=True`. The loader now only rescues ties from rows marked `Training`, preventing that split leakage.

### Remaining considerations

- Perceptual deduplication currently drops any later image with the same 64-bit dHash. dHash is a compact similarity fingerprint, so equal values do not prove that two source images are duplicates. This can discard distinct faces that happen to share a hash. Review the `dropped_phash_duplicates` count on the actual dataset; consider Hamming-distance matching and dataset-level validation before treating it as near-duplicate removal.
- The sample weights are clamped to a nonzero floor, so very ambiguous or noise-heavy samples still contribute to the loss. Confirm that this matches the intended training policy.
- `webcam_infer.py` catches checkpoint-loading errors, but camera opening and runtime failures are not handled with a `finally` cleanup path. Ensure the camera is released if inference exits unexpectedly.
- The README documents the supported module interfaces, but exact dependency versions and a reproducible training command are unavailable because no dependency lockfile or training entry point is included.

## Verification

Reviewed the source diff and whitespace errors. No tests or model runs were performed; the required dataset, checkpoints, and training entry point are absent.
