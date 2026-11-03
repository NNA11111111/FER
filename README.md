# FER+ Facial Expression Recognition

FER2013 image and FER+ crowd-label pipeline for eight facial-expression classes. The code supports soft targets, optional official FER partitions, global deduplication, a separate tied-label evaluation set, EAC-style attention consistency, calibration metrics, webcam inference, and 112×112 model inputs.

## Modules

- `data_utils.py`: CSV validation, filtering, global exact/perceptual deduplication, stratified or original-Usage splits, and manifests.
- `preprocessing.py`: shared resize, CLAHE, and normalization. The current input is 112×112; FER2013 source pixels remain 48×48, so interpolation cannot restore image detail.
- `models.py`: custom attention CNN and a ResNet18 adapter requiring a local face-pretrained checkpoint.
- `eval_utils.py`: soft-label losses, EAC-style attention consistency, scalar logit adjustment, ECE, temperature fitting, and metrics.
- `dataset.py`: training augmentation and soft-label mixup.
- `webcam_infer.py`: webcam inference using shared preprocessing and saved calibration.
- `train-fer.ipynb`: training workflow. The notebook contains the executed training and evaluation workflow, including saved outputs and metrics.

## Install

Use Python 3.11 or 3.12 and install pinned dependencies from `requirements.txt`. Install a matching CPU/CUDA PyTorch wheel where your runtime requires a different wheel index.

## Data and split modes

The loader expects FER2013 `pixels` plus FER+ `Usage`, eight emotion vote columns, and optional `unknown`/`NF`. It joins by common `id`/`image_id` when possible. Positional joins require `allow_positional_join=True` and matching Usage values.

The default is stratified 60/30/10 (train/validation/test) by the majority hard label. To keep official partitions, use `split_mode="original_usage"`: Training → train, PublicTest → validation, PrivateTest → test. Deduplication runs globally before splitting, preventing duplicate images from crossing partitions.

For a separate tied-label evaluation set, use `split_mode="original_usage", return_tie_eval=True`. The loader then returns a fourth value `(X_tie, y_tie_soft, y_tie_hard)` for eligible PrivateTest ties. These examples are excluded from the ordinary splits and deduplicated against retained images. Use the soft labels for probability metrics.

```python
from data_utils import load_fer2013_csv

(train, val, test, tie_eval) = load_fer2013_csv(
    "fer2013.csv", "fer2013new.csv",
    return_soft_labels=True,
    return_sample_weights=True,
    split_mode="original_usage",
    return_tie_eval=True,
    allow_positional_join=True,
)
```

Set `split_manifest_dir` to save split IDs and configuration. `load_split_from_manifest` reconstructs the ordinary three splits.

## Noise robustness and calibration

`CustomCNN(images, return_attention=True, attention_classes=...)` returns logits and target-class CAMs from the final feature map. `eac_attention_consistency_loss` aligns a CAM with the horizontally flipped view. Set `FER2013Dataset(..., eac_mode=True)` to use horizontal flip plus dynamic random erasing on every class (including rare classes), without the extra affine transform. The notebook must compute classification loss on the original view only, add the CAM consistency loss, and compare EAC on/off by ablation. This uses FER+ soft-label loss in place of EAC's hard-label classification loss.

`fit_temperature` fits a single positive temperature on held-out logits. `expected_calibration_error` supports hard-label ECE and crowd-agreement ECE when soft targets are provided. Fit calibration only on validation or out-of-fold predictions, then report test metrics without refitting.

`FacePretrainedResNet18(checkpoint_path=...)` adapts a local ResNet18-compatible face-pretrained checkpoint to grayscale input. No pretrained weights are included. The 112×112 image is an upsampled FER2013 face, not a higher-resolution source image.

## Webcam

Save a checkpoint bundle as `best_bundle.pth` and run `python webcam_infer.py`. A bundle can store one scalar `posthoc_tau`, `temperature`, and class priors. Generated checkpoints and split manifests are ignored by Git.

## Git artifacts

The calibrated inference bundle (`best_bundle.pth`) and calibration metadata are included for webcam inference. Per-seed and redundant standalone checkpoints are excluded by `.gitignore` to keep the repository compact.
