# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TomoML is an engineering thesis project on CT image reconstruction using deep learning. The core task is sinogram-to-CT-image reconstruction: given a sinogram (radon transform of a CT slice), reconstruct the original CT image. The main approach is a Pix2Pix GAN (conditional adversarial network).

## Commands

### Running scripts

```bash
# Plot training metrics from a log file
python src/train_plotter.py <log_file>         # default: training_log.txt
python src/metrics_plotter.py <log_file>       # alternative metrics visualizer

# Interactive DICOM dataset selection/categorization
python src/data_selection.py <ct_root_dir>
# or via environment variable:
CT_ROOT_DIR=<path> python src/data_selection.py

# Interactive CT windowing visualizer (compares original vs reconstructed)
# Edit RECONSTRUCTOR = "FBP" | "FBP_CUDA" | "NN" inside the script before running
python src/windowing_visualizer.py
```

> **Note**: `data_selection.py` has hardcoded Windows paths (`ROOT_DIR`, `OUTPUT_ACCEPT`, `OUTPUT_REJECT`) near the top of the file — update these before running on Linux.

### Compiling LaTeX (thesis)

```bash
docker run --rm -it \
  -v $HOME/TomoML:/TomoML \
  -w /TomoML \
  texlive/texlive:latest bash

latexmk -pdf main.tex
```

### Training and experiments

Training is done via Jupyter notebooks in `src/`. Launch Jupyter and open the relevant notebook.

## Architecture

### Model versions (src/models/)

**Pix2Pix models** — a **Generator** (UNet-style encoder-decoder) paired with a **Conditional Discriminator**:

- `Pix2Pix_128.py` — 128×128 generator with residual blocks (`ResidualConvBlock`) and `ConvTranspose2d` upsampling; discriminator uses `InstanceNorm2d` + `LeakyReLU`.
- `Pix2Pix_128_V2.py` — Same architecture but decoder uses `Upsample + Conv2d` instead of `ConvTranspose2d` (avoids checkerboard artifacts).
- `Pix2Pix_256_V1.py` — 256×256 variant with plain `conv_block` (no residuals), lighter encoder (32→64→128→256 channels).

All Pix2Pix generators end with `AdaptiveAvgPool2d` to force a fixed output resolution.

`ConditionalDiscriminator.forward(x, cond)` takes the generated/real image `x` and the sinogram `cond` as separate arguments — it concatenates them internally after resizing `cond` to match `x`.

**ConvNetX models** — based on ["Limited-Angle Tomography Reconstruction via Deep End-To-End Learning on Synthetic Data"](https://arxiv.org/abs/2309.06948) (Germer et al.):

- `conv_netx.py` — Contains `ConvNetX` (512×256) and `ConvNetX_128` (256×183) variants. Uses residual `Block` modules with Conv2d, BatchNorm, and GELU activation. No GAN discriminator — standalone encoder-decoder.

### Data pipeline

- **Dataset split lists**: `src/resources/ct_scan_train.txt`, `ct_scan_val.txt`, `ct_scan_test.txt` — each line is a relative path to a CT scan directory.
- **Accepted/rejected scans**: `src/resources/ct_scan_accepted.txt`, `ct_scan_rejected.txt` — manually curated via `data_selection.py`.
- **Dataset class** (`experiments/v1/CTSinogramDataset.py`): pairs PNG sinograms (`sinogram_<name>.png`) with CT images (`<name>.png`).
- Sinograms are generated from DICOM CT files using ASTRA toolbox (see `experiments/astra/generate_sinograms.ipynb`).

### Key notebooks (src/)

| Notebook | Purpose |
|---|---|
| `data_generation.ipynb` | Generate sinogram/CT image pairs from DICOM files |
| `generate_ellipses.ipynb` | Generate synthetic ellipse dataset for pretraining |
| `load_dataset.ipynb` | Visualize and inspect the dataset |
| `train_pix2pix_128.ipynb` | Train 128×128 Pix2Pix model |
| `train_pix2pix_256.ipynb` | Train 256×256 Pix2Pix model |
| `train_conv_netx.ipynb` | Train ConvNetX model |
| `test_model.ipynb` | Evaluate a trained model |
| `load_natural_images.ipynb` | Load natural images for domain experiments |

### Utilities (src/utils.py)

- `SinogramNoise` — augmentation transform adding Gaussian noise to sinograms
- `calculate_mse`, `calculate_psnr`, `calculate_ssim`, `calculate_correlation` — batch-aware image quality metrics
- `count_parameters` — count trainable model parameters

### Training log format

`train_plotter.py` parses logs with this format per epoch:
```
Epoch N/M | Train Loss: X | G Val Loss: X | Train MSE: X | Val MSE: X | Train SSIM: X | Val SSIM: X | Train PSNR: X | Val PSNR: X | Train Corr: X | Val Corr: X
```

### Models directory

Trained model checkpoints stored as `.pth` files in `models/`, named `<resolution>_<date>_<variant>.pth`.

## Data sources

- **Hospital CT scans**: Private dataset of DICOM files; paths referenced in `src/resources/`. DICOM files are named `I10` or `I10.dcm` inside per-series subdirectories.
- **LoDoPaB-CT**: Public dataset used in earlier experiments.
- **Synthetic ellipses**: Generated internally to pretrain models before fine-tuning on real CT data.

## Earlier experiments

The `experiments/` directory contains older, standalone work:
- `v1/` — First UNet (`UNetV1`) and dataset loader
- `v2/` — UNet V2 (notebook-only)
- `v3/` — Early Pix2Pix (notebook-only)
- `astra/` — ASTRA toolbox sinogram generation and ODL experiments
- `helpers/` — Utility notebooks for DICOM reading, augmentation, downsampling
