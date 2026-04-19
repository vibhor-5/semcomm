"""
Tests for models/flow_decoder.py.
Uses tiny model (n_channels=16, n_blocks=2, image_size=8) for speed.
torchdiffeq must be installed.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
import pytest  # noqa: E402

try:
    from models.flow_decoder import FlowDecoder

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

pytestmark = pytest.mark.skipif(not HAS_TORCHDIFFEQ, reason="torchdiffeq not installed")

# Small test model
LDIM = 32
TDIM = 16
ISIZE = 8
N_CH = 16
N_BLK = 2
BATCH = 2


@pytest.fixture(scope="module")
def flow_model_linear():
    return FlowDecoder(
        latent_dim=LDIM,
        token_dim=TDIM,
        image_size=ISIZE,
        n_channels=N_CH,
        n_blocks=N_BLK,
        path_type="linear",
    )


@pytest.fixture(scope="module")
def flow_model_no_token():
    return FlowDecoder(
        latent_dim=LDIM, token_dim=0, image_size=ISIZE, n_channels=N_CH, n_blocks=N_BLK
    )


# -----------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------


def test_loss_finite(flow_model_linear):
    x0 = torch.randn(BATCH, 3, ISIZE, ISIZE)
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    loss = flow_model_linear.compute_loss(x0, lat, tok)
    assert loss.item() > 0, "Loss must be positive"
    assert not torch.isnan(loss), "Loss must not be NaN"


def test_loss_no_token(flow_model_no_token):
    x0 = torch.randn(BATCH, 3, ISIZE, ISIZE)
    lat = torch.randn(BATCH, LDIM)
    tok = torch.zeros(BATCH, 0)  # empty token
    loss = flow_model_no_token.compute_loss(x0, lat, tok)
    assert not torch.isnan(loss)


def test_loss_backward(flow_model_linear):
    """Backward pass must not raise."""
    x0 = torch.randn(BATCH, 3, ISIZE, ISIZE)
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    loss = flow_model_linear.compute_loss(x0, lat, tok)
    loss.backward()


# -----------------------------------------------------------------------
# Sample
# -----------------------------------------------------------------------


def test_sample_shape(flow_model_linear):
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    out = flow_model_linear.sample(lat, tok, steps=3)
    assert out.shape == (BATCH, 3, ISIZE, ISIZE)


def test_sample_range(flow_model_linear):
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    out = flow_model_linear.sample(lat, tok, steps=3)
    assert out.min() >= -1.1 and out.max() <= 1.1


def test_sample_guidance(flow_model_linear):
    model = FlowDecoder(
        latent_dim=LDIM,
        token_dim=TDIM,
        image_size=ISIZE,
        n_channels=N_CH,
        n_blocks=N_BLK,
        use_guidance=True,
    )
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    out = model.sample(lat, tok, steps=3, guidance_scale=2.0)
    assert out.shape == (BATCH, 3, ISIZE, ISIZE)


# -----------------------------------------------------------------------
# SNR conditioning
# -----------------------------------------------------------------------


def test_snr_conditioning_shapes():
    model = FlowDecoder(
        latent_dim=LDIM,
        token_dim=TDIM,
        image_size=ISIZE,
        n_channels=N_CH,
        n_blocks=N_BLK,
        snr_conditioning=True,
    )
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    snr_emb = torch.tensor([[10.0], [5.0]])
    out = model.sample(lat, tok, steps=3, snr_emb=snr_emb)
    assert out.shape == (BATCH, 3, ISIZE, ISIZE)


# -----------------------------------------------------------------------
# OT path (skipped if POT not installed)
# -----------------------------------------------------------------------


def test_ot_path_loss():
    try:
        import ot  # noqa: F401
    except ImportError:
        pytest.skip("POT (ot) not installed")
    model = FlowDecoder(
        latent_dim=LDIM,
        token_dim=TDIM,
        image_size=ISIZE,
        n_channels=N_CH,
        n_blocks=N_BLK,
        path_type="ot",
    )
    x0 = torch.randn(BATCH, 3, ISIZE, ISIZE)
    lat = torch.randn(BATCH, LDIM)
    tok = torch.randn(BATCH, TDIM)
    loss = model.compute_loss(x0, lat, tok)
    assert not torch.isnan(loss)
