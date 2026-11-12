# REDCPNet: Salient Road Crack Detection with a Square Convolutional Pyramid

This repository contains a PyTorch implementation of **REDCPNet**, an encoder–decoder network for pixel-level road-crack segmentation on **CRACK500**. The project is inspired by the prediction architecture of BASNet and the square convolution-pyramid design of RENet, and compares REDCPNet with U-Net, FCN, and BASNet.

> **Repository scope:** REDCPNet and its training, inference, data-loading, and metric utilities are included. The U-Net, FCN, and BASNet comparison implementations were used in the original experiments but are currently stored locally and are not included here. The results below are transcribed from the thesis and have not been independently reproduced from this repository update.

## Project at a Glance

| Item | Description |
|---|---|
| Task | Salient road-crack / pavement-defect segmentation |
| Dataset | CRACK500, pixel-level masks |
| Proposed model | REDCPNet |
| Framework | PyTorch |
| Architecture | Residual encoder–bridge–decoder with multi-scale convolution pyramid |
| Intended training loss in thesis | BCE + SSIM + IoU |
| Evaluation | F-measure (β² = 0.3), MAE |
| Baselines in thesis | U-Net, FCN, BASNet |

## Method

### REDCPNet architecture

REDCPNet uses an end-to-end encoder–bridge–decoder structure:

- **Encoder:** six stages of residual feature extraction. Dilated convolutions (dilation rates 2 and 5 in later residual blocks) enlarge the receptive field while retaining feature-map resolution.
- **Square convolution-pyramid modules:** the supplied implementation attaches these modules after encoder stages 1–5. Each module has three branches:
  1. a 1×1 convolution branch;
  2. four parallel factorized asymmetric-kernel paths using 1×3/3×1, 1×5/5×1, 1×7/7×1, and 1×9/9×1 convolutions;
  3. global average pooling, 1×1 convolution, and upsampling.
- **Bridge:** three dilated 3×3 convolution blocks connect encoder and decoder representations.
- **Decoder:** progressively fuses decoder features with encoder features through skip connections and upsampling.
- **Side outputs:** seven prediction maps are returned; the inference script selects `d1` as the final prediction.

The pyramid aggregates local details, multi-scale context, and global information to support crack-boundary delineation.

### Training objective

The thesis defines the combined objective as:

`L_total = L_BCE + L_SSIM + L_IoU`

BCE provides pixel-wise supervision, SSIM encourages structural and boundary similarity, and IoU emphasizes overlap between predicted and ground-truth defect regions.

**Important code note:** in the uploaded `train(1).py`, `bce_loss`, `ssim_loss`, and `iou_loss` are instantiated, but `bce_ssim_loss()` returns only `ssim_out`; the BCE and IoU terms are not included in the returned loss. The current uploaded training script therefore does not implement the full three-term objective described in the thesis. See `docs/implementation_review.md`.

## Dataset and Preprocessing

The thesis reports the following CRACK500 split:

| Split | Original images | Cropped patches |
|---|---:|---:|
| Training | 250 | 1,896 |
| Validation | 50 | 348 |
| Test | 200 | 1,124 |

The reported preprocessing divides each original image into 16 non-overlapping crops and retains patches containing more than 1,000 defect pixels. Training resizes images to 256×256 and randomly crops them to 224×224.

The dataset is not included. Obtain CRACK500 from its original provider and follow its terms of use. Do not upload the full dataset unless redistribution is permitted.

## Experimental Results

### Baseline comparison

Higher F-measure and lower MAE are better. Values are from the thesis' quantitative comparison table.

| Model | F-measure ↑ | MAE ↓ |
|---|---:|---:|
| U-Net | 0.1120 | 0.1928 |
| FCN | 0.1337 | 0.1274 |
| BASNet | 0.6964 | 0.0340 |
| **REDCPNet (proposed)** | **0.7171** | **0.0318** |

Compared with BASNet, the thesis reports an absolute F-measure increase of 0.0207 and an MAE reduction from 0.0340 to 0.0318. The thesis reports evaluation on 1,124 cropped test images.

**Baseline status:** U-Net, FCN, and BASNet source code is not currently included in this GitHub repository. These are report-derived results and cannot be independently rerun from the current repository alone.

### Loss-function ablation

| Loss configuration | F-measure ↑ | MAE ↓ |
|---|---:|---:|
| BCE | 0.4351 | 0.0467 |
| SSIM | 0.4444 | 0.0396 |
| BCE + IoU | 0.6197 | 0.0374 |
| BCE + SSIM | 0.6550 | 0.0369 |
| IoU | 0.7150 | 0.0338 |
| BCE + SSIM + IoU | **0.7171** | **0.0318** |

The thesis reports the best result for the combined loss. These are historical thesis results, not fresh runs from this repository update.

## Repository Structure

```text
Salient-Object-Detection/
├── README.md
├── REDCPNet.py
├── residual_model.py
├── convolution_pyramid.py
├── train.py
├── test.py
├── data_loader.py
├── evaluation_metrics.py
├── IOU.py
├── SSIM.py
├── test_metrics.py
├── requirements.txt
├── .gitignore
├── docs/implementation_review.md
├── data/README.md
└── results/README.md
```

The uploaded filenames include `train(1).py` and `data_loader(1).py`. Rename them to `train.py` and `data_loader.py` only after confirming imports and updating references consistently.

## Installation

The uploaded scripts import PyTorch, torchvision, NumPy, scikit-image, Pillow, Matplotlib, and OpenCV. `test_metrics.py` additionally imports the external `sod_metrics` package.

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS / Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch/torchvision build compatible with the local CPU/CUDA environment using the official PyTorch installation selector. Exact dependency versions from the original experiment were not fully pinned in the supplied files.

## Training and Inference

The supplied scripts contain machine-specific Windows paths. Before running:

1. Download and prepare CRACK500.
2. Arrange image and mask folders to match the dataset loader.
3. Update dataset and checkpoint paths.
4. Resolve the training-loss discrepancy and verify the experiment configuration.
5. Run a small smoke test before full training.

```bash
python train.py
python test.py
```

The inference script loads REDCPNet, obtains seven output maps, selects `d1`, normalizes it, and saves a prediction map. Convert hard-coded paths to command-line arguments or a configuration file before presenting this as a plug-and-play demo.

## Evaluation

The thesis uses:

- **F-measure:** calculated from precision and recall with β² = 0.3, weighting precision more heavily.
- **MAE:** average absolute pixel-wise difference between prediction and ground-truth masks.

`test_metrics.py` additionally computes weighted F-measure, S-measure, E-measure, and MAE through `sod_metrics`. Pair prediction and mask files by filename before formal evaluation; do not rely on independently sorted glob lists.

## Reproducibility Notes and Limitations

- U-Net, FCN, and BASNet implementations are not included in this repository yet.
- The uploaded training loss does not match the full BCE + SSIM + IoU objective in the thesis.
- The thesis specifies learning rate 1e-3, while uploaded `train(1).py` sets 1e-4.
- The thesis describes test resizing to 256×256, while uploaded `test.py` uses `RescaleT(224)`.
- The uploaded `test.py` points to a training-image directory rather than an explicit held-out test split.
- Several scripts contain absolute Windows paths.
- `test_metrics.py` pairs masks and predictions by list index; match by filename and assert one-to-one correspondence.
- `test_metrics.py` requires the external `sod_metrics` package.
- The model uses deprecated `F.sigmoid`; prefer `torch.sigmoid` for maintenance.
- Reported metrics are historical thesis values, not reproduced results from this README update.

## Future Work

1. Align and unit-test the implemented loss against BCE + SSIM + IoU.
2. Standardize train/validation/test preprocessing and configurable paths.
3. Add baseline implementations and exact experiment configurations.
4. Add qualitative input/ground-truth/prediction visualizations.
5. Add fixed seeds, checkpoint metadata, and reproducible logs.
6. Profile parameter count, FLOPs, inference latency, and GPU memory.

## Reference

**基于深度学习的路面裂纹缺陷检测研究** (Undergraduate thesis, 2023). The thesis proposes REDCPNet and reports experiments on CRACK500.
