import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim


class SinogramNoise:
    def __init__(self, mean=0.0, std=0.001, p=1.0):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, sample):
        sinogram, image = sample

        if random.random() < self.p:
            noise = torch.randn_like(sinogram) * self.std + self.mean
            sinogram = sinogram + noise
            sinogram.clamp_(0.0, 1.0)

        return sinogram, image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_mse(img1, img2):
    """Returns mean MSE over batch (scalar float)."""
    return F.mse_loss(img1, img2, reduction="mean").item()


def calculate_psnr(img1, img2, max_val=1.0, eps=1e-12):
    """Batch-aware PSNR. Computes PSNR per image and averages over batch."""
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    if img1_np.ndim == 4:
        psnr_scores = []
        for i1, i2 in zip(img1_np, img2_np):
            mse = np.mean((i1 - i2) ** 2)
            mse = max(mse, eps)
            psnr = 20 * math.log10(max_val) - 10 * math.log10(mse)
            psnr_scores.append(psnr)
        return float(np.mean(psnr_scores))
    else:
        mse = np.mean((img1_np - img2_np) ** 2)
        mse = max(mse, eps)
        return float(20 * math.log10(max_val) - 10 * math.log10(mse))


def calculate_ssim(img1, img2):
    """Batch-aware SSIM."""
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    if img1_np.ndim == 4:
        scores = [
            ssim(i1.squeeze(), i2.squeeze(), data_range=1.0)
            for i1, i2 in zip(img1_np, img2_np)
        ]
        return float(np.mean(scores))
    else:
        return float(
            ssim(img1_np.squeeze(), img2_np.squeeze(), data_range=1.0)
        )


def calculate_correlation(img1, img2):
    """Batch-aware Pearson correlation. Computes per-image and averages."""
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    if img1_np.ndim == 4:
        scores = []
        for i1, i2 in zip(img1_np, img2_np):
            corr = np.corrcoef(i1.flatten(), i2.flatten())[0, 1]
            scores.append(corr)
        return float(np.nanmean(scores))
    else:
        return float(
            np.corrcoef(img1_np.flatten(), img2_np.flatten())[0, 1]
        )
