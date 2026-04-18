"""
Tests for evaluation/metrics.py.
Skips FID, LPIPS, and CLIP-based tests in CI (too heavy / need nets).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import torch
import pytest
from evaluation.metrics import compute_psnr, compute_ssim, compute_bpp


# -----------------------------------------------------------------------
# compute_psnr
# -----------------------------------------------------------------------

def test_psnr_identical():
    x = torch.rand(4, 3, 32, 32) * 2 - 1   # in [-1, 1]
    psnr = compute_psnr(x, x)
    assert psnr > 80.0, f"Identical images should give very high PSNR, got {psnr}"


def test_psnr_finite():
    orig  = torch.rand(4, 3, 32, 32) * 2 - 1
    recon = torch.rand(4, 3, 32, 32) * 2 - 1
    psnr = compute_psnr(orig, recon)
    assert math.isfinite(psnr)


def test_psnr_monotone():
    """Higher MSE → lower PSNR."""
    orig  = torch.zeros(4, 3, 32, 32)
    small = orig + 0.01
    large = orig + 0.5
    assert compute_psnr(orig, small) > compute_psnr(orig, large)


# -----------------------------------------------------------------------
# compute_bpp
# -----------------------------------------------------------------------

def test_bpp_k128():
    assert compute_bpp(128, 8, 32, 32) == pytest.approx(1.0)


def test_bpp_k64():
    assert compute_bpp(64, 8, 32, 32) == pytest.approx(0.5)


def test_bpp_k32():
    assert compute_bpp(32, 8, 32, 32) == pytest.approx(0.25)


def test_bpp_k256():
    assert compute_bpp(256, 8, 32, 32) == pytest.approx(2.0)


def test_bpp_4bit():
    assert compute_bpp(128, 4, 32, 32) == pytest.approx(0.5)


# -----------------------------------------------------------------------
# compute_ssim
# -----------------------------------------------------------------------

def test_ssim_identical():
    x = torch.rand(4, 3, 32, 32) * 2 - 1
    ssim = compute_ssim(x, x)
    assert ssim > 0.99, f"Identical images should give SSIM ≈ 1, got {ssim}"


def test_ssim_range():
    orig  = torch.rand(4, 3, 32, 32) * 2 - 1
    recon = torch.rand(4, 3, 32, 32) * 2 - 1
    ssim  = compute_ssim(orig, recon)
    assert -1.1 <= ssim <= 1.1


# -----------------------------------------------------------------------
# compute_lpips (skipped if lpips not installed)
# -----------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get('CI', 'false') == 'true',
    reason="lpips model download not available in CI"
)
def test_lpips_identical():
    from evaluation.metrics import compute_lpips
    device = torch.device('cpu')
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    score = compute_lpips(x, x, device)
    assert score < 0.05, f"LPIPS of identical images should be ~0, got {score}"


@pytest.mark.skipif(
    os.environ.get('CI', 'false') == 'true',
    reason="lpips model download not available in CI"
)
def test_lpips_range():
    from evaluation.metrics import compute_lpips
    device = torch.device('cpu')
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    y = torch.rand(2, 3, 32, 32) * 2 - 1
    score = compute_lpips(x, y, device)
    assert 0.0 <= score <= 2.0
