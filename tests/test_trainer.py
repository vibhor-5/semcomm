"""
Smoke test for Trainer._train_step.
Uses tiny dummy models so no GPU or downloaded data is needed.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch

try:
    from models.encoder import Encoder
    from data.synthetic_channel import AWGNChannel
    from models.flow_decoder import FlowDecoder
    from training.trainer import Trainer
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


MINIMAL_CFG = {
    'experiment_id': 'test_smoke',
    'seed': 42,
    'encoder': {'latent_dim': 32, 'encoder_type': 'cnn', 'semantic_token_type': 'none'},
    'channel': {'type': 'awgn', 'snr_db': 10, 'snr_conditioning': False},
    'decoder': {'type': 'flow', 'n_blocks': 2, 'n_channels': 16},
    'training': {
        'n_epochs': 1,
        'batch_size': 2,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'gradient_clip': 1.0,
        'num_workers': 0,
    },
    'evaluation': {'eval_every_n_epochs': 1, 'sampling_steps': 3},
    'logging': {
        'use_wandb': False,
        'project': 'test',
        'save_every_n_epochs': 1,
        'checkpoint_dir': '/tmp/test_ckpts',
    },
    'loss': {'flow_weight': 1.0, 'clip_weight': 0.0, 'vgg_weight': 0.0},
}


def _make_dummy_loader(image_size=8, batch_size=2, n_batches=2):
    """Returns a DataLoader that yields (images, labels, tokens)."""
    images  = torch.randn(n_batches * batch_size, 3, image_size, image_size)
    labels  = torch.zeros(n_batches * batch_size, dtype=torch.long)
    tokens  = torch.zeros(n_batches * batch_size, 512)
    ds = TensorDataset(images, labels, tokens)
    return DataLoader(ds, batch_size=batch_size)


@pytest.mark.skipif(not HAS_TORCHDIFFEQ, reason="torchdiffeq not installed")
def test_trainer_one_step():
    """One training step must not raise and must return a valid loss dict."""
    enc = Encoder(latent_dim=32, encoder_type='cnn', semantic_token_type='none')
    ch  = AWGNChannel(snr_db=10)
    dec = FlowDecoder(latent_dim=32, token_dim=0, image_size=8, n_channels=16, n_blocks=2)

    trainer = Trainer(enc, ch, dec, MINIMAL_CFG['loss'], MINIMAL_CFG, drive_root='/tmp')
    loader  = _make_dummy_loader(image_size=8, batch_size=2)
    batch   = next(iter(loader))

    result = trainer._train_step(batch)
    assert 'total' in result
    assert 'main'  in result
    assert 'grad_norm' in result
    assert result['total'] >= 0


@pytest.mark.skipif(not HAS_TORCHDIFFEQ, reason="torchdiffeq not installed")
def test_trainer_validate_returns_dict():
    enc = Encoder(latent_dim=32, encoder_type='cnn', semantic_token_type='none')
    ch  = AWGNChannel(snr_db=10)
    dec = FlowDecoder(latent_dim=32, token_dim=0, image_size=8, n_channels=16, n_blocks=2)

    trainer = Trainer(enc, ch, dec, MINIMAL_CFG['loss'], MINIMAL_CFG, drive_root='/tmp')
    loader  = _make_dummy_loader(image_size=8, batch_size=2)

    metrics = trainer._validate(loader)
    assert 'clip'  in metrics
    assert 'psnr'  in metrics


@pytest.mark.skipif(not HAS_TORCHDIFFEQ, reason="torchdiffeq not installed")
def test_trainer_run_no_wandb(tmp_path):
    enc = Encoder(latent_dim=32, encoder_type='cnn', semantic_token_type='none')
    ch  = AWGNChannel(snr_db=10)
    dec = FlowDecoder(latent_dim=32, token_dim=0, image_size=8, n_channels=16, n_blocks=2)

    cfg = {**MINIMAL_CFG}
    cfg['logging'] = {**MINIMAL_CFG['logging'],
                      'use_wandb': False,
                      'checkpoint_dir': str(tmp_path / 'ckpts')}
    trainer = Trainer(enc, ch, dec, cfg['loss'], cfg, drive_root=str(tmp_path))
    loader  = _make_dummy_loader(image_size=8, batch_size=2, n_batches=1)

    # Patch wandb so the import inside run() doesn't fail if wandb isn't installed
    import sys
    import types
    if 'wandb' not in sys.modules:
        sys.modules['wandb'] = types.ModuleType('wandb')

    # run must complete without raising
    trainer.run(loader, loader, n_epochs=1)
