"""
Tests for models/diffusion_decoder.py.
Uses tiny model (base_channels=16, image_size=8) for speed.
diffusers must be installed.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest

try:
    from models.diffusion_decoder import DiffusionDecoder
    HAS_DIFFUSERS = True
except Exception:
    HAS_DIFFUSERS = False

pytestmark = pytest.mark.skipif(
    not HAS_DIFFUSERS,
    reason="diffusers not installed or DiffusionDecoder failed to import"
)

LDIM  = 32
TDIM  = 16
ISIZE = 8
BCH   = 16
BATCH = 2

# We test with 'concat' conditioning because it doesn't require cross-attention
# and is the simplest/fastest variant.

@pytest.fixture(scope="module")
def diff_model_concat():
    return DiffusionDecoder(
        latent_dim=LDIM, token_dim=TDIM, image_size=ISIZE,
        conditioning='concat', base_channels=BCH, n_levels=2, timesteps=100
    )


# -----------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------

def test_diff_loss_finite(diff_model_concat):
    x0  = torch.randn(BATCH, 3, ISIZE, ISIZE)
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    loss = diff_model_concat.compute_loss(x0, lat, tok)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_diff_loss_backward(diff_model_concat):
    x0  = torch.randn(BATCH, 3, ISIZE, ISIZE)
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    loss = diff_model_concat.compute_loss(x0, lat, tok)
    loss.backward()


# -----------------------------------------------------------------------
# Sample
# -----------------------------------------------------------------------

def test_diff_sample_shape(diff_model_concat):
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    out = diff_model_concat.sample(lat, tok, steps=3)
    assert out.shape == (BATCH, 3, ISIZE, ISIZE)


def test_diff_sample_range(diff_model_concat):
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    out = diff_model_concat.sample(lat, tok, steps=3)
    assert out.min() >= -1.1 and out.max() <= 1.1


def test_diff_guidance(diff_model_concat):
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    out = diff_model_concat.sample(lat, tok, steps=3, guidance_scale=2.0)
    assert out.shape == (BATCH, 3, ISIZE, ISIZE)


# -----------------------------------------------------------------------
# SNR conditioning
# -----------------------------------------------------------------------

def test_diff_snr_conditioning():
    model = DiffusionDecoder(
        latent_dim=LDIM, token_dim=TDIM, image_size=ISIZE,
        conditioning='concat', base_channels=BCH, n_levels=2,
        timesteps=100, snr_conditioning=True
    )
    lat     = torch.randn(BATCH, LDIM)
    tok     = torch.randn(BATCH, TDIM)
    snr_emb = torch.tensor([[10.0], [5.0]])
    out = model.sample(lat, tok, steps=3, snr_emb=snr_emb)
    assert out.shape == (BATCH, 3, ISIZE, ISIZE)
