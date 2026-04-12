import os
import sys
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.seed import set_seed
from utils.device import get_device
from data.datasets import SemCommDataset
from data.augmentations import get_transforms
from models.encoder import Encoder
from data.synthetic_channel import AWGNChannel, RayleighChannel, BurstErasureChannel
from models.flow_decoder import FlowDecoder
from models.diffusion_decoder import DiffusionDecoder
from training.trainer import Trainer

def build_channel(channel_cfg):
    ctype = channel_cfg.get('type', 'awgn').lower()
    if ctype == 'awgn':
        return AWGNChannel(snr_db=channel_cfg.get('snr_db'), 
                           snr_range=channel_cfg.get('snr_range', (0, 20)),
                           return_snr=channel_cfg.get('snr_conditioning', False))
    elif ctype == 'rayleigh':
        return RayleighChannel(snr_db=channel_cfg.get('snr_db'), 
                               snr_range=channel_cfg.get('snr_range', (0, 20)),
                               return_snr=channel_cfg.get('snr_conditioning', False))
    elif ctype == 'burst':
        return BurstErasureChannel(erasure_prob=channel_cfg.get('erasure_prob', 0.2),
                                   snr_db=channel_cfg.get('snr_db', 10),
                                   return_snr=channel_cfg.get('snr_conditioning', False))
    else:
        raise ValueError(f"Unknown channel type {ctype}")

def build_decoder(decoder_cfg, encoder_cfg, channel_cfg):
    dtype = decoder_cfg.get('type', 'flow').lower()
    if dtype == 'flow':
        return FlowDecoder(latent_dim=encoder_cfg['latent_dim'],
                           token_dim=512 if encoder_cfg.get('semantic_token_type') == 'clip' else 0,
                           image_size=32, # assuming cifar
                           n_channels=decoder_cfg.get('n_channels', 128),
                           n_blocks=decoder_cfg.get('n_blocks', 5),
                           path_type=decoder_cfg.get('path_type', 'linear'),
                           use_guidance=decoder_cfg.get('use_guidance', False),
                           snr_conditioning=decoder_cfg.get('snr_conditioning', False))
    elif dtype == 'diffusion':
        return DiffusionDecoder(latent_dim=encoder_cfg['latent_dim'],
                                token_dim=512 if encoder_cfg.get('semantic_token_type') == 'clip' else 0,
                                image_size=32,
                                conditioning=decoder_cfg.get('conditioning', 'concat'),
                                base_channels=decoder_cfg.get('base_channels', 128),
                                snr_conditioning=decoder_cfg.get('snr_conditioning', False))
    else:
        raise ValueError(f"Unknown decoder type {dtype}")

def main():
    parser = argparse.ArgumentParser(description="Run Semantic Communication Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/data", help="Root directory for datasets")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/outputs", help="Root directory for outputs")
    parser.add_argument("--eval_only", action="store_true", help="Skip training, directly run evaluation on best.pt")
    args = parser.parse_args()

    # Setup Environment
    device = get_device()
    print(f"Using device: {device}")

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))

    # Path overrides for Kaggle / Cloud
    os.makedirs(args.data_dir, exist_ok=True)
    cfg['dataset']['root'] = args.data_dir
    clip_cache = os.path.join(args.data_dir, f"{cfg['dataset']['name']}_clip_train.pt")
    cfg['dataset']['clip_cache_path'] = clip_cache
    
    exp_dir = os.path.join(args.output_dir, cfg['experiment_id'])
    cfg['logging']['checkpoint_dir'] = os.path.join(exp_dir, "checkpoints")

    # 1. Precompute CLIP tokens if needed
    if cfg['encoder'].get('semantic_token_type') == 'clip':
        if not os.path.exists(clip_cache):
            print(f"Precomputing CLIP tokens to {clip_cache}...")
            SemCommDataset.precompute_clip_tokens(
                root=args.data_dir,
                dataset_name=cfg['dataset']['name'],
                image_size=cfg['dataset']['image_size'],
                cache_path=clip_cache,
                device=device
            )
    
    # 2. Setup DataLoaders
    train_trans, val_trans = get_transforms(cfg['dataset']['image_size'])
    
    train_ds = SemCommDataset(args.data_dir, 'train', cfg['dataset']['name'], 
                              cfg['dataset']['image_size'], clip_cache_path=clip_cache, 
                              transform=train_trans.get(cfg['dataset'].get('augmentation', 'basic'), train_trans['basic']))
                              
    val_ds = SemCommDataset(args.data_dir, 'val', cfg['dataset']['name'], 
                            cfg['dataset']['image_size'], clip_cache_path=clip_cache, 
                            transform=val_trans)

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], 
                              shuffle=True, num_workers=cfg['training'].get('num_workers', 2), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], 
                            shuffle=False, num_workers=cfg['training'].get('num_workers', 2), pin_memory=True)

    # 3. Build Models
    encoder = Encoder(**cfg['encoder'], device=device)
    channel = build_channel(cfg['channel'])
    decoder = build_decoder(cfg['decoder'], cfg['encoder'], cfg['channel'])

    # 4. Train
    if not args.eval_only:
        print(f"Starting training for {cfg['experiment_id']}...")
        trainer = Trainer(encoder, channel, decoder, cfg['loss'], cfg, drive_root=args.output_dir)
        trainer.run(train_loader, val_loader, cfg['training']['n_epochs'])
        print("Training complete.")

    # 5. Evaluate
    print(f"Starting evaluation for {cfg['experiment_id']}...")
    from evaluation.evaluate import run_evaluation
    
    best_ckpt = os.path.join(cfg['logging']['checkpoint_dir'], 'best.pt')
    if not os.path.exists(best_ckpt):
        # Fallback to last epoch if best.pt wasn't saved (e.g. valid loader mock)
        last_epoch = cfg['training']['n_epochs']
        best_ckpt = os.path.join(cfg['logging']['checkpoint_dir'], f'epoch_{last_epoch}.pt')
        
    if os.path.exists(best_ckpt):
        metrics = run_evaluation(cfg, best_ckpt, val_loader, device, encoder, channel, decoder)
        print(f"Evaluation metrics: {metrics}")
    else:
        print(f"Warning: Could not find checkpoint at {best_ckpt} for evaluation.")

if __name__ == "__main__":
    main()
