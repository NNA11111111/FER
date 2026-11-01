# FER+ Facial Expression Recognition

A Python project for facial expression recognition on FER2013 images with FER+ crowd labels. It contains data preparation, a PyTorch dataset with training augmentations, a custom CNN, evaluation utilities, and webcam inference.

## Repository contents

- `data_utils.py`: loads the FER2013 and FER+ CSV files, filters labels, deduplicates images, creates train/validation/test splits, and optionally returns soft labels and sample weights.
- `dataset.py`: PyTorch dataset, training augmentations, and selective mixup helper.
- `models.py`: residual CNN with channel and spatial attention.
- `eval_utils.py`: losses, class balancing, metrics, calibration, Grad-CAM, and evaluation helpers.
- `webcam_infer.py`: real-time inference with a webcam and a compatible model checkpoint.
- `preprocessing.py`: shared CLAHE and normalization for training and webcam inference.
- `test_data_utils.py`: synthetic-data checks for loading, splitting, and preprocessing.
- `review.md`: code review notes for this revision.
- `train-fer.ipynb`: training and evaluation workflow for Kaggle, including multiple seeds and checkpoint export.

## Requirements

Python 3.9+ and the packages `numpy`, `pandas`, `opencv-python`, `scikit-learn`, `torch`, `torchvision`, `matplotlib`, and `seaborn` are needed. Install compatible PyTorch and torchvision builds for your CPU or CUDA environment, then install the remaining packages with pip.

## Data

Place FER2013 images and FER+ labels in CSV format locally. The loader expects an image CSV with `pixels`, and a label CSV with `Usage`, the eight emotion vote columns (`neutral`, `happiness`, `surprise`, `sadness`, `anger`, `disgust`, `fear`, `contempt`), and optionally `unknown` and `NF`. If both files share `image_id` or `id`, they are joined by that field; otherwise row order and row counts must match.

The raw image CSV is excluded from Git by `.gitignore`. The loader defaults to a deterministic 70/15/15 stratified split, tie filtering, and perceptual-hash deduplication. Set `use_stratify=False` to use the original `Usage` partitions. `return_soft_labels=True` and `return_sample_weights=True` expose the corresponding training targets and weights.

Set `split_manifest_dir` to write split manifests and a deduplication report. Use `load_split_from_manifest` to reload a recorded split. The shared preprocessing applies CLAHE to 48×48 grayscale images and scales values to [-1, 1].

Example:

```python
from data_utils import load_fer2013_csv

(train, val, test) = load_fer2013_csv(
    "dataset/fer2013.csv",
    "dataset/fer2013new.csv",
    return_soft_labels=True,
    return_sample_weights=True,
)
```

## Webcam inference

Save a trained `CustomCNN` state dict or checkpoint as `best.pth` in the working directory, then run:

```bash
python webcam_infer.py
```

The webcam script expects a camera accessible as device 0. A `best_calibration.pt` file is optional; when available, its class priors and post-hoc adjustment settings are used. Model weights, FER data, and calibration files are not included in this repository.

## Training

Open `train-fer.ipynb` in Kaggle with the FER2013 image CSV (`fer2013.csv`), FER+ label CSV (`fer2013new.csv`), and repository Python modules available as inputs. Run its cells in order. The notebook locates those files, prepares deterministic splits, trains and evaluates the model, and saves checkpoints. Adjust paths and runtime settings in the notebook for a local environment.

Generated checkpoints (`*.pth`, `*.pt`) and split reports are kept out of Git. Keep your own copies of those outputs if you need to reproduce inference or inspect a particular training run.

## Tests

Install `pytest` and run `python -m pytest test_data_utils.py` to check the data pipeline with synthetic inputs.
