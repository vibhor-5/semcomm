"""
Shared pytest fixtures for the semcomm test suite.
All tests run on CPU without CLIP or diffusers for CI compatibility.
"""
import sys
import os
import pytest
import torch

# Make the project root importable from tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def device():
    return torch.device('cpu')


@pytest.fixture(scope="session")
def tiny_image_batch():
    """2 images, 3-channel, 8×8 — smallest useful spatial size for flow/diffusion."""
    return torch.randn(2, 3, 8, 8)


@pytest.fixture(scope="session")
def tiny_latent(latent_dim=32):
    return torch.randn(2, latent_dim)


@pytest.fixture(scope="session")
def tiny_token(token_dim=16):
    return torch.randn(2, token_dim)
