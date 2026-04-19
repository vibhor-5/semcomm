"""Tests for data/synthetic_channel.py."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
import pytest  # noqa: E402
from data.synthetic_channel import (  # noqa: E402
    AWGNChannel,
    RayleighChannel,
    BurstErasureChannel,
)


# ---------------------------------------------------------------------------
# AWGNChannel
# ---------------------------------------------------------------------------


def test_awgn_shape_preserved():
    ch = AWGNChannel(snr_db=10)
    x = torch.randn(8, 128)
    assert ch(x).shape == (8, 128)


def test_awgn_noise_power():
    """SNR=10 dB on unit-power signal → noise power ≈ 0.1."""
    ch = AWGNChannel(snr_db=10)
    x = torch.ones(200, 128)  # unit power
    noisy = ch(x)
    noise_power = (noisy - x).pow(2).mean().item()
    assert abs(noise_power - 0.1) < 0.025, f"noise_power={noise_power:.4f}"


def test_awgn_random_snr_range():
    ch = AWGNChannel(snr_range=(5, 15), return_snr=True)
    _, snr = ch(torch.randn(4, 64))
    assert 5 <= snr <= 15


def test_awgn_return_snr_false():
    ch = AWGNChannel(snr_db=10, return_snr=False)
    result = ch(torch.randn(4, 64))
    assert isinstance(result, torch.Tensor)


def test_awgn_return_snr_true():
    ch = AWGNChannel(snr_db=10, return_snr=True)
    noisy, snr = ch(torch.randn(4, 64))
    assert isinstance(noisy, torch.Tensor)
    assert snr == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# RayleighChannel
# ---------------------------------------------------------------------------


def test_rayleigh_shape():
    ch = RayleighChannel(snr_db=10)
    x = torch.randn(4, 64)
    assert ch(x).shape == (4, 64)


def test_rayleigh_return_snr():
    ch = RayleighChannel(snr_db=15, return_snr=True)
    noisy, snr = ch(torch.randn(4, 64))
    assert snr == pytest.approx(15.0)
    assert noisy.shape == (4, 64)


# ---------------------------------------------------------------------------
# BurstErasureChannel
# ---------------------------------------------------------------------------


def test_burst_shape():
    ch = BurstErasureChannel(erasure_prob=0.2, snr_db=10)
    x = torch.randn(4, 64)
    assert ch(x).shape == (4, 64)


def test_burst_erasure_rate():
    """With erasure_prob=1.0, all elements after erasure should be ~noise only."""
    ch = BurstErasureChannel(erasure_prob=1.0, snr_db=100)  # high SNR → noise≈0
    x = torch.ones(1000, 1)
    out = ch(x)
    # All elements erased → values should be very close to 0 (+ tiny noise)
    assert out.abs().mean().item() < 0.1


def test_burst_return_snr():
    ch = BurstErasureChannel(erasure_prob=0.2, snr_db=10, return_snr=True)
    out, snr = ch(torch.randn(4, 64))
    assert snr == pytest.approx(10.0)
    assert out.shape == (4, 64)
