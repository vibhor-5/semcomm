import copy, yaml, os

def write_config(cfg: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

base_flow = {
    'experiment_id': 'B1_linear',
    'name': 'Flow Path Compare — Linear',
    'seed': 42,
    'dataset': {
        'name': 'cifar10',
        'root': '/content/drive/MyDrive/semcomm/data/cifar10',
        'image_size': 32,
        'augmentation': 'basic',
        'clip_cache_path': '/content/drive/MyDrive/semcomm/data/cifar10_clip_train.pt'
    },
    'encoder': {
        'latent_dim': 128,
        'encoder_type': 'cnn',
        'semantic_token_type': 'clip',
        'quant_bits': 8
    },
    'channel': {
        'type': 'awgn',
        'snr_db': 10,
        'snr_conditioning': False
    },
    'decoder': {
        'type': 'flow',
        'n_channels': 128,
        'n_blocks': 5,
        'path_type': 'linear',
        'use_guidance': False,
        'snr_conditioning': False
    },
    'loss': {
        'flow_weight': 1.0,
        'clip_weight': 0.0,
        'vgg_weight': 0.0
    },
    'training': {
        'n_epochs': 100,
        'batch_size': 64,
        'lr': 1.0e-4,
        'weight_decay': 1.0e-5,
        'gradient_clip': 1.0,
        'scheduler': 'cosine',
        'mixed_precision': True,
        'num_workers': 2
    },
    'evaluation': {
        'metrics': ['clip', 'lpips', 'fid', 'psnr', 'ssim'],
        'eval_every_n_epochs': 10,
        'sampling_steps': 50,
        'guidance_scale': 1.0,
        'n_fid_samples': 5000
    },
    'logging': {
        'use_wandb': True,
        'project': 'semcomm_freegpu',
        'save_every_n_epochs': 5,
        'checkpoint_dir': '/content/drive/MyDrive/semcomm/outputs/B1_linear/checkpoints'
    }
}

# Dump Base
write_config(base_flow, 'configs/flow/B1_linear.yaml')

# B1: two path types
for path_type in ['linear', 'ot']:
    cfg = copy.deepcopy(base_flow)
    cfg['experiment_id'] = f'B1_{path_type}'
    cfg['decoder']['path_type'] = path_type
    cfg['logging']['checkpoint_dir'] = cfg['logging']['checkpoint_dir'].replace('B1_linear', f'B1_{path_type}')
    write_config(cfg, f'configs/flow/B1_{path_type}.yaml')

# B2: n_blocks sweep
for n in [3, 5, 8]:
    cfg = copy.deepcopy(base_flow)
    cfg['experiment_id'] = f'B2_nblocks{n}'
    cfg['decoder']['n_blocks'] = n
    cfg['training']['n_epochs'] = 60
    cfg['logging']['checkpoint_dir'] = cfg['logging']['checkpoint_dir'].replace('B1_linear', f'B2_nblocks{n}')
    write_config(cfg, f'configs/flow/B2_nblocks{n}.yaml')

print("Configs generated.")
