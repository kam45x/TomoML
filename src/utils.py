import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import convolve
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
    """Batch-aware MSE. Computes MSE per image and averages over batch."""
    if img1.ndim == 4:
        per_image_mse = F.mse_loss(img1, img2, reduction="none")
        per_image_mse = per_image_mse.mean(dim=[1, 2, 3])
        return per_image_mse.mean().item()
    else:
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


def _gmsd_single(img1, img2, c=0.0026):
    """Compute GMSD for a single 2D image pair."""
    prewitt_h = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
    prewitt_v = prewitt_h.T

    g1_h = convolve(img1, prewitt_h)
    g1_v = convolve(img1, prewitt_v)
    g2_h = convolve(img2, prewitt_h)
    g2_v = convolve(img2, prewitt_v)

    m1 = np.sqrt(g1_h ** 2 + g1_v ** 2)
    m2 = np.sqrt(g2_h ** 2 + g2_v ** 2)

    gms = (2 * m1 * m2 + c) / (m1 ** 2 + m2 ** 2 + c)
    return float(np.std(gms))


def calculate_gmsd(img1, img2):
    """Batch-aware GMSD (Gradient Magnitude Similarity Deviation).
    Lower is better (0 = identical gradient structure).
    Based on Xue et al., IEEE TIP 2014."""
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    if img1_np.ndim == 4:
        scores = [
            _gmsd_single(i1.squeeze(), i2.squeeze())
            for i1, i2 in zip(img1_np, img2_np)
        ]
        return float(np.mean(scores))
    else:
        return _gmsd_single(img1_np.squeeze(), img2_np.squeeze())


def calculate_lpips(img1, img2, lpips_net):
    """Batch-aware LPIPS (Learned Perceptual Image Patch Similarity).
    Lower is better (0 = perceptually identical).
    Expects a pre-initialized lpips.LPIPS model passed as lpips_net."""
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # Repeat single-channel to 3 channels for VGG
    if img1.shape[1] == 1:
        img1 = img1.repeat(1, 3, 1, 1)
        img2 = img2.repeat(1, 3, 1, 1)

    # Scale from [0, 1] to [-1, 1] as expected by LPIPS
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    with torch.no_grad():
        scores = lpips_net(img1, img2)
    return float(scores.mean().item())
