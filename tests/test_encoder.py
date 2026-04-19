"""
Tests for models/encoder.py.
Skips CLIP-dependent tests in CI (use --ignore=tests/test_encoder.py ci flag or
mark them with skip_clip).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
import pytest  # noqa: E402
from models.encoder import Encoder, quantise  # noqa: E402

# -----------------------------------------------------------------------
# quantise()
# -----------------------------------------------------------------------


def test_quantise_range():
    x = torch.randn(32, 128)
    q = quantise(x, bits=8)
    assert q.min() >= -1.05 and q.max() <= 1.05


def test_quantise_straight_through():
    """Gradient must flow through the quantisation gate."""
    x = torch.randn(4, 64, requires_grad=True)
    q = quantise(x, bits=8)
    q.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_quantise_4bit_coarser():
    """4-bit quantisation has coarser steps than 8-bit."""
    x = torch.tensor([[0.1, 0.5, -0.3]])
    q8 = quantise(x, bits=8)
    q4 = quantise(x, bits=4)
    # 4-bit values are rounded to coarser grid → larger max diff from original
    assert (q4 - x).abs().max() >= (q8 - x).abs().max() - 1e-6


# -----------------------------------------------------------------------
# Encoder (CNN backbone, no CLIP)
# -----------------------------------------------------------------------


def test_encoder_cnn_shapes():
    enc = Encoder(latent_dim=64, encoder_type="cnn", semantic_token_type="none")
    img = torch.randn(4, 3, 32, 32)
    lat, tok = enc(img)
    assert lat.shape == (4, 64)
    assert tok.shape == (4, 0)  # 'none' → empty token


def test_encoder_latent_range():
    enc = Encoder(latent_dim=128, encoder_type="cnn", semantic_token_type="none")
    img = torch.randn(4, 3, 32, 32)
    lat, _ = enc(img)
    assert (
        lat.min() >= -1.05 and lat.max() <= 1.05
    ), "Latent should be in [-1,1] after tanh + quantise"


def test_encoder_no_nan():
    enc = Encoder(latent_dim=128, encoder_type="cnn", semantic_token_type="none")
    img = torch.randn(4, 3, 32, 32)
    lat, tok = enc(img)
    assert not torch.isnan(lat).any()


def test_encoder_class_onehot():
    enc = Encoder(
        latent_dim=32,
        encoder_type="cnn",
        semantic_token_type="class_onehot",
        num_classes=10,
    )
    img = torch.randn(2, 3, 32, 32)
    lat, tok = enc(img)
    # class_onehot token falls back to zeros here (labels not passed to encoder)
    assert lat.shape == (2, 32)


@pytest.mark.skipif(
    os.environ.get("CI", "false") == "true",
    reason="CLIP not installed in CI environment",
)
def test_encoder_clip_shapes():
    enc = Encoder(latent_dim=128, encoder_type="cnn", semantic_token_type="clip")
    img = torch.randn(4, 3, 32, 32)
    lat, tok = enc(img)
    assert lat.shape == (4, 128)
    assert tok.shape == (4, 512)


@pytest.mark.skipif(
    os.environ.get("CI", "false") == "true",
    reason="CLIP not installed in CI environment",
)
def test_encoder_clip_frozen():
    enc = Encoder(latent_dim=128, encoder_type="cnn", semantic_token_type="clip")
    for p in enc.clip_model.parameters():
        assert not p.requires_grad, "CLIP parameters must be frozen"


# -----------------------------------------------------------------------
# Encoder (ResNet-18 / MobileNetV2) — just check shapes, no CLIP
# -----------------------------------------------------------------------


def test_encoder_resnet18_shape():
    enc = Encoder(latent_dim=64, encoder_type="resnet18", semantic_token_type="none")
    img = torch.randn(2, 3, 32, 32)
    lat, tok = enc(img)
    assert lat.shape == (2, 64)


def test_encoder_mobilenetv2_shape():
    enc = Encoder(latent_dim=64, encoder_type="mobilenetv2", semantic_token_type="none")
    img = torch.randn(2, 3, 32, 32)
    lat, tok = enc(img)
    assert lat.shape == (2, 64)
