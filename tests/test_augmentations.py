"""Tests for data/augmentations.py."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402
from data.augmentations import get_transforms, mixup_collate_fn  # noqa: E402


def _make_pil(size=32):
    return Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))


def test_basic_transform_output_range():
    train_trans, _ = get_transforms(32)
    img = _make_pil(32)
    t = train_trans["basic"](img)
    assert t.shape == (3, 32, 32)
    assert t.min() >= -1.1 and t.max() <= 1.1


def test_val_transform_output_range():
    _, val_trans = get_transforms(32)
    img = _make_pil(32)
    t = val_trans(img)
    assert t.shape == (3, 32, 32)
    assert t.min() >= -1.1 and t.max() <= 1.1


def test_none_transform():
    train_trans, _ = get_transforms(32)
    img = _make_pil(32)
    t = train_trans["none"](img)
    assert t.shape == (3, 32, 32)


def test_full_transform():
    train_trans, _ = get_transforms(32)
    img = _make_pil(32)
    t = train_trans["full"](img)
    assert t.shape == (3, 32, 32)


def test_mixup_collate_shapes():
    batch = [
        (torch.randn(3, 8, 8), 0, torch.randn(16)),
        (torch.randn(3, 8, 8), 1, torch.randn(16)),
        (torch.randn(3, 8, 8), 2, torch.randn(16)),
        (torch.randn(3, 8, 8), 3, torch.randn(16)),
    ]
    collate = mixup_collate_fn(alpha=0.4)
    images, labels, tokens = collate(batch)
    assert images.shape == (4, 3, 8, 8)
    assert labels.shape == (4,)
    assert tokens.shape == (4, 16)


def test_mixup_range():
    """MixUp of images in [-1,1] must stay in [-1,1]."""
    batch = [
        (torch.clamp(torch.randn(3, 8, 8), -1, 1), 0, torch.zeros(16)) for _ in range(4)
    ]
    collate = mixup_collate_fn(alpha=1.0)
    images, _, _ = collate(batch)
    assert images.min() >= -1.1 and images.max() <= 1.1
