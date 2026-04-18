"""
Tests for data/datasets.py.
Creates a tiny in-memory CIFAR-10 substitute using TensorDataset so we don't
need to download the actual dataset in CI.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from unittest.mock import patch, MagicMock
from data.augmentations import get_transforms


# -----------------------------------------------------------------------
# Patch torchvision.datasets.CIFAR10 so tests don't hit the network
# -----------------------------------------------------------------------

class _FakeCIFAR10:
    """Tiny in-memory CIFAR10 stand-in (8 samples)."""
    def __init__(self, *args, **kwargs):
        from PIL import Image
        import numpy as np
        self._data = [(Image.fromarray(np.random.randint(0, 255, (32, 32, 3),
                                                         dtype=np.uint8)), i % 10)
                      for i in range(8)]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


@pytest.fixture()
def fake_ds(tmp_path):
    """Return a SemCommDataset backed by the in-memory fake CIFAR10."""
    with patch('torchvision.datasets.CIFAR10', _FakeCIFAR10):
        from data.datasets import SemCommDataset
        train_trans, _ = get_transforms(32)
        ds = SemCommDataset(
            root=str(tmp_path),
            split='train',
            dataset_name='cifar10',
            image_size=32,
            transform=train_trans['basic'],
        )
    return ds


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

def test_dataset_len(fake_ds):
    assert len(fake_ds) == 8


def test_dataset_item_shapes(fake_ds):
    img, lbl, tok = fake_ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(lbl, int)
    assert tok.shape == (512,)   # zeros (no cache)


def test_image_range(fake_ds):
    img, _, _ = fake_ds[0]
    assert img.min() >= -1.1 and img.max() <= 1.1


def test_clip_token_zeros_without_cache(fake_ds):
    """Without a cache file, tokens should be all-zero tensors."""
    _, _, tok = fake_ds[0]
    assert tok.sum().item() == 0.0


def test_clip_token_from_cache(tmp_path):
    """With a valid cache, tokens should be loaded correctly."""
    # Write a fake cache file
    fake_tokens = torch.randn(8, 512)
    cache_path = tmp_path / 'fake_cache.pt'
    torch.save(fake_tokens, str(cache_path))

    with patch('torchvision.datasets.CIFAR10', _FakeCIFAR10):
        from data.datasets import SemCommDataset
        train_trans, _ = get_transforms(32)
        ds = SemCommDataset(
            root=str(tmp_path),
            split='train',
            dataset_name='cifar10',
            image_size=32,
            clip_cache_path=str(cache_path),
            transform=train_trans['basic'],
        )

    _, _, tok = ds[0]
    assert tok.shape == (512,)
    assert torch.allclose(tok, fake_tokens[0])


def test_unknown_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        from data.datasets import SemCommDataset
        SemCommDataset(str(tmp_path), 'train', 'imagenet_huge', 32)
