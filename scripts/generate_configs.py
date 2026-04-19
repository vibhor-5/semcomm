"""
Generates ALL experiment YAML configs for the semcomm project.
Run from the repo root:
    python scripts/generate_configs.py

This will populate:
    configs/baselines/  (A1, A2, A3)
    configs/flow/       (B1–B10)
    configs/diffusion/  (C1–C10)
    configs/comparison/ (D1–D5, E1–E4)
"""

import copy
import os
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def write_config(cfg: dict, rel_path: str):
    full_path = os.path.join(ROOT, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  wrote {rel_path}")


# -----------------------------------------------------------------------
# Base configs
# -----------------------------------------------------------------------

BASE_FLOW = {
    "experiment_id": "B1_linear",
    "name": "Flow Path Compare — Linear",
    "seed": 42,
    "dataset": {
        "name": "cifar10",
        "root": "/content/drive/MyDrive/semcomm/data/cifar10",
        "image_size": 32,
        "augmentation": "basic",
        "clip_cache_path": "/content/drive/MyDrive/semcomm/data/cifar10_clip_train.pt",
    },
    "encoder": {
        "latent_dim": 128,
        "encoder_type": "cnn",
        "semantic_token_type": "clip",
        "quant_bits": 8,
    },
    "channel": {
        "type": "awgn",
        "snr_db": 10,
        "snr_conditioning": False,
    },
    "decoder": {
        "type": "flow",
        "n_channels": 128,
        "n_blocks": 5,
        "path_type": "linear",
        "use_guidance": False,
        "snr_conditioning": False,
    },
    "loss": {
        "flow_weight": 1.0,
        "clip_weight": 0.0,
        "vgg_weight": 0.0,
    },
    "training": {
        "n_epochs": 100,
        "batch_size": 64,
        "lr": 1.0e-4,
        "weight_decay": 1.0e-5,
        "gradient_clip": 1.0,
        "scheduler": "cosine",
        "mixed_precision": True,
        "num_workers": 2,
    },
    "evaluation": {
        "metrics": ["clip", "lpips", "fid", "psnr", "ssim"],
        "eval_every_n_epochs": 10,
        "sampling_steps": 50,
        "guidance_scale": 1.0,
        "n_fid_samples": 5000,
    },
    "logging": {
        "use_wandb": True,
        "project": "semcomm_freegpu",
        "save_every_n_epochs": 5,
        "checkpoint_dir": "/content/drive/MyDrive/semcomm/outputs/B1_linear/checkpoints",
    },
}

BASE_DIFFUSION = copy.deepcopy(BASE_FLOW)
BASE_DIFFUSION["decoder"] = {
    "type": "diffusion",
    "conditioning": "cross_attention",
    "base_channels": 128,
    "n_levels": 3,
    "timesteps": 1000,
    "use_vae_latent": False,
    "snr_conditioning": False,
}

BASE_BASELINE = copy.deepcopy(BASE_FLOW)


def _ckpt(exp_id: str):
    return f"/content/drive/MyDrive/semcomm/outputs/{exp_id}/checkpoints"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def flow_cfg(exp_id: str, name: str, overrides: dict) -> dict:
    cfg = copy.deepcopy(BASE_FLOW)
    cfg["experiment_id"] = exp_id
    cfg["name"] = name
    cfg["logging"]["checkpoint_dir"] = _ckpt(exp_id)
    _deep_update(cfg, overrides)
    return cfg


def diff_cfg(exp_id: str, name: str, overrides: dict) -> dict:
    cfg = copy.deepcopy(BASE_DIFFUSION)
    cfg["experiment_id"] = exp_id
    cfg["name"] = name
    cfg["logging"]["checkpoint_dir"] = _ckpt(exp_id)
    _deep_update(cfg, overrides)
    return cfg


def _deep_update(base: dict, updates: dict):
    for k, v in updates.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# -----------------------------------------------------------------------
# Group A — Baselines
# -----------------------------------------------------------------------
print("Generating Group A configs...")

# A1 — Eval calibration (CPU only, no model training)
cfg = copy.deepcopy(BASE_BASELINE)
cfg["experiment_id"] = "A1_eval_calibration"
cfg["name"] = "Evaluation Pipeline Calibration"
cfg["training"]["n_epochs"] = 0  # no training
cfg["logging"]["checkpoint_dir"] = _ckpt("A1_eval_calibration")
write_config(cfg, "configs/baselines/A1_eval_calibration.yaml")

# A2 — DeepJSCC (k=64 and k=128)
for k in [64, 128]:
    eid = f"A2_deepjscc_k{k}"
    cfg = copy.deepcopy(BASE_BASELINE)
    cfg["experiment_id"] = eid
    cfg["name"] = f"DeepJSCC Baseline k={k}"
    cfg["encoder"]["latent_dim"] = k
    cfg["encoder"]["semantic_token_type"] = "none"
    cfg["decoder"] = {"type": "deepjscc", "latent_dim": k}
    cfg["training"]["n_epochs"] = 50
    cfg["training"]["batch_size"] = 128
    cfg["training"]["lr"] = 1.0e-3
    cfg["logging"]["checkpoint_dir"] = _ckpt(eid)
    write_config(cfg, f"configs/baselines/{eid}.yaml")

# A3 — JPEG baseline (CPU only)
cfg = copy.deepcopy(BASE_BASELINE)
cfg["experiment_id"] = "A3_jpeg"
cfg["name"] = "JPEG Traditional Compression Baseline"
cfg["training"]["n_epochs"] = 0
cfg["logging"]["checkpoint_dir"] = _ckpt("A3_jpeg")
write_config(cfg, "configs/baselines/A3_jpeg.yaml")

# -----------------------------------------------------------------------
# Group B — Flow-Matching
# -----------------------------------------------------------------------
print("Generating Group B configs...")

# B1 — Flow path: linear vs OT
for path_type in ["linear", "ot"]:
    eid = f"B1_{path_type}"
    cfg = flow_cfg(
        eid,
        f"Flow Path Compare — {path_type.upper()}",
        {"decoder": {"path_type": path_type}},
    )
    write_config(cfg, f"configs/flow/{eid}.yaml")

# B2 — CNF depth ablation
for n in [3, 5, 8]:
    eid = f"B2_nblocks{n}"
    cfg = flow_cfg(
        eid,
        f"CNF Depth Ablation n_blocks={n}",
        {"decoder": {"n_blocks": n}, "training": {"n_epochs": 60}},
    )
    write_config(cfg, f"configs/flow/{eid}.yaml")

# B3 — Semantic conditioning ablation
B3_CONDS = {
    "none": {"semantic_token_type": "none", "token_dim": 0},
    "clip": {},
    "clip_noisy": {},  # handled in training code by passing noisy latent + clip token
    "clip_class": {"semantic_token_type": "class_onehot"},
}
for variant, enc_override in B3_CONDS.items():
    eid = f"B3_{variant}"
    cfg = flow_cfg(
        eid,
        f"Semantic Conditioning — {variant}",
        {"encoder": enc_override, "training": {"n_epochs": 80}},
    )
    write_config(cfg, f"configs/flow/{eid}.yaml")

# B4 — Bitrate sweep
for k in [32, 64, 128, 256]:
    eid = f"B4_k{k}"
    cfg = flow_cfg(
        eid,
        f"Rate-Semantic Curve Flow k={k}",
        {"encoder": {"latent_dim": k}, "training": {"n_epochs": 100}},
    )
    write_config(cfg, f"configs/flow/{eid}.yaml")

# B5 — SNR sweep (eval only, no training)
eid = "B5_snr_sweep"
cfg = flow_cfg(
    eid,
    "Channel Robustness SNR Sweep (Flow — eval only)",
    {"training": {"n_epochs": 0}},
)
write_config(cfg, "configs/flow/B5_snr_sweep.yaml")

# B6 — Channel-aware training
for variant in ["random_snr", "random_snr_token"]:
    use_token = variant == "random_snr_token"
    eid = f"B6_{variant}"
    cfg = flow_cfg(
        eid,
        f"Channel-Aware Flow Training — {variant}",
        {
            "channel": {
                "snr_db": None,
                "snr_range": [0, 20],
                "snr_conditioning": use_token,
            },
            "decoder": {"snr_conditioning": use_token},
            "training": {"n_epochs": 100},
        },
    )
    write_config(cfg, f"configs/flow/{eid}.yaml")

# B7 — Guidance weight sweep (eval only)
for w in [1.0, 1.5, 2.0, 3.0]:
    eid = f'B7_w{str(w).replace(".", "p")}'
    cfg = flow_cfg(
        eid,
        f"Guidance Weight Sweep w={w} (Flow — eval only)",
        {
            "decoder": {"use_guidance": True},
            "evaluation": {"guidance_scale": w},
            "training": {"n_epochs": 0},
        },
    )
    write_config(cfg, f"configs/flow/{eid}.yaml")

# B8 — TinyImageNet scale-up
eid = "B8_tinyimagenet"
cfg = flow_cfg(
    eid,
    "Scale-Up TinyImageNet 64×64 (Flow)",
    {
        "dataset": {
            "name": "tinyimagenet",
            "image_size": 64,
            "root": "/content/drive/MyDrive/semcomm/data/tinyimagenet",
            "clip_cache_path": "/content/drive/MyDrive/semcomm/data/tinyimagenet_clip_train.pt",
        },
        "decoder": {"n_channels": 128, "n_blocks": 5},
        "training": {"n_epochs": 80},
    },
)
write_config(cfg, "configs/flow/B8_tinyimagenet.yaml")

# B9 — ODE steps sweep (eval only)
for steps in [5, 10, 25, 50, 100, 200]:
    eid = f"B9_steps{steps}"
    cfg = flow_cfg(
        eid,
        f"ODE Steps vs Quality steps={steps} (eval only)",
        {
            "evaluation": {"sampling_steps": steps},
            "training": {"n_epochs": 0},
        },
    )
    write_config(cfg, f"configs/flow/{eid}.yaml")

# B10 — Fading channel robustness (eval only)
for ch_type in ["awgn", "rayleigh", "burst"]:
    eid = f"B10_{ch_type}"
    cfg = flow_cfg(
        eid,
        f"Fading Channel Robustness — {ch_type} (eval only)",
        {
            "channel": {
                "type": ch_type,
                "erasure_prob": 0.2 if ch_type == "burst" else None,
            },
            "training": {"n_epochs": 0},
        },
    )
    # Remove None values
    cfg["channel"] = {k: v for k, v in cfg["channel"].items() if v is not None}
    write_config(cfg, f"configs/flow/{eid}.yaml")

# -----------------------------------------------------------------------
# Group C — Diffusion
# -----------------------------------------------------------------------
print("Generating Group C configs...")

# C1 — Conditioning strategy sweep
for cond in ["concat", "cross_attention", "adain", "none"]:
    eid = f"C1_{cond}"
    cfg = diff_cfg(
        eid,
        f"Conditioning Strategy — {cond}",
        {
            "decoder": {"conditioning": cond},
            "training": {"n_epochs": 80},
        },
    )
    write_config(cfg, f"configs/diffusion/{eid}.yaml")

# C2 — Diffusion steps vs quality (eval only)
for steps in [5, 10, 25, 50, 100, 200]:
    eid = f"C2_steps{steps}"
    cfg = diff_cfg(
        eid,
        f"Diffusion Steps vs Quality steps={steps} (eval only)",
        {
            "evaluation": {"sampling_steps": steps},
            "training": {"n_epochs": 0},
        },
    )
    write_config(cfg, f"configs/diffusion/{eid}.yaml")

# C3 — Bitrate sweep
for k in [32, 64, 128, 256]:
    eid = f"C3_k{k}"
    cfg = diff_cfg(
        eid,
        f"Rate-Semantic Curve Diffusion k={k}",
        {
            "encoder": {"latent_dim": k},
            "training": {"n_epochs": 100},
        },
    )
    write_config(cfg, f"configs/diffusion/{eid}.yaml")

# C4 — Guidance sweep (eval only)
for w in [1.0, 1.5, 2.0, 3.0, 5.0]:
    eid = f'C4_w{str(w).replace(".", "p")}'
    cfg = diff_cfg(
        eid,
        f"Guidance Weight Sweep w={w} (Diffusion — eval only)",
        {
            "evaluation": {"guidance_scale": w},
            "training": {"n_epochs": 0},
        },
    )
    write_config(cfg, f"configs/diffusion/{eid}.yaml")

# C5 — SNR sweep (eval only)
eid = "C5_snr_sweep"
cfg = diff_cfg(
    eid,
    "Channel Robustness SNR Sweep (Diffusion — eval only)",
    {"training": {"n_epochs": 0}},
)
write_config(cfg, "configs/diffusion/C5_snr_sweep.yaml")

# C6 — Channel-aware diffusion
for variant in ["random_snr", "random_snr_token"]:
    use_token = variant == "random_snr_token"
    eid = f"C6_{variant}"
    cfg = diff_cfg(
        eid,
        f"Channel-Aware Diffusion Training — {variant}",
        {
            "channel": {
                "snr_db": None,
                "snr_range": [0, 20],
                "snr_conditioning": use_token,
            },
            "decoder": {"snr_conditioning": use_token},
            "training": {"n_epochs": 100},
        },
    )
    write_config(cfg, f"configs/diffusion/{eid}.yaml")

# C7 — Latent vs pixel space
for space in ["pixel", "vae_latent"]:
    use_vae = space == "vae_latent"
    eid = f"C7_{space}"
    cfg = diff_cfg(
        eid,
        f"Diffusion Space — {space}",
        {
            "decoder": {"use_vae_latent": use_vae},
            "training": {"n_epochs": 80},
        },
    )
    write_config(cfg, f"configs/diffusion/{eid}.yaml")

# C8 — Semantic loss augmentation
LOSS_VARIANTS = {
    "mse_only": {"clip_weight": 0.0, "vgg_weight": 0.0},
    "mse_clip": {"clip_weight": 0.1, "vgg_weight": 0.0},
    "mse_vgg": {"clip_weight": 0.0, "vgg_weight": 0.1},
    "mse_clip_vgg": {"clip_weight": 0.1, "vgg_weight": 0.1},
}
for variant, loss_override in LOSS_VARIANTS.items():
    eid = f"C8_{variant}"
    cfg = diff_cfg(
        eid,
        f"Semantic Loss Augmentation — {variant}",
        {"loss": loss_override, "training": {"n_epochs": 80}},
    )
    write_config(cfg, f"configs/diffusion/{eid}.yaml")

# C9 — TinyImageNet scale-up
eid = "C9_tinyimagenet"
cfg = diff_cfg(
    eid,
    "Scale-Up TinyImageNet 64×64 (Diffusion)",
    {
        "dataset": {
            "name": "tinyimagenet",
            "image_size": 64,
            "root": "/content/drive/MyDrive/semcomm/data/tinyimagenet",
            "clip_cache_path": "/content/drive/MyDrive/semcomm/data/tinyimagenet_clip_train.pt",
        },
        "training": {"n_epochs": 60},
    },
)
write_config(cfg, "configs/diffusion/C9_tinyimagenet.yaml")

# C10 — Fading channel robustness (eval only)
for ch_type in ["awgn", "rayleigh", "burst"]:
    eid = f"C10_{ch_type}"
    cfg = diff_cfg(
        eid,
        f"Fading Channel Robustness — {ch_type} (Diffusion eval only)",
        {
            "channel": {
                "type": ch_type,
                "erasure_prob": 0.2 if ch_type == "burst" else None,
            },
            "training": {"n_epochs": 0},
        },
    )
    cfg["channel"] = {k: v for k, v in cfg["channel"].items() if v is not None}
    write_config(cfg, f"configs/diffusion/{eid}.yaml")

# -----------------------------------------------------------------------
# Group D — Comparisons (no new training — eval / aggregation only)
# -----------------------------------------------------------------------
print("Generating Group D configs...")

for d_id, d_name in [
    ("D1_rate_curves", "Rate-Semantic Curve All Models"),
    ("D2_speed_quality", "Speed vs Quality Flow vs Diffusion"),
    ("D3_snr_robustness", "Channel Robustness All Models"),
    ("D4_recon_gallery", "Reconstruction Gallery"),
    ("D5_clip_dist", "CLIP Score Distribution"),
]:
    cfg = copy.deepcopy(BASE_FLOW)
    cfg["experiment_id"] = d_id
    cfg["name"] = d_name
    cfg["training"]["n_epochs"] = 0
    cfg["logging"]["checkpoint_dir"] = _ckpt(d_id)
    write_config(cfg, f"configs/comparison/{d_id}.yaml")

# -----------------------------------------------------------------------
# Group E — Ablations
# -----------------------------------------------------------------------
print("Generating Group E configs...")

# E1 — Encoder architecture
for enc_type in ["cnn", "resnet18", "mobilenetv2"]:
    eid = f"E1_{enc_type}"
    cfg = flow_cfg(
        eid,
        f"Encoder Architecture — {enc_type}",
        {
            "encoder": {"encoder_type": enc_type},
            "training": {"n_epochs": 50},
        },
    )
    write_config(cfg, f"configs/comparison/{eid}.yaml")

# E2 — Semantic token type
for tok_type in ["clip", "learned", "class_onehot", "none"]:
    eid = f"E2_{tok_type}"
    cfg = flow_cfg(
        eid,
        f"Semantic Token Type — {tok_type}",
        {
            "encoder": {"semantic_token_type": tok_type},
            "training": {"n_epochs": 60},
        },
    )
    write_config(cfg, f"configs/comparison/{eid}.yaml")

# E3 — Quantisation method
for quant_method in ["uniform8", "uniform4", "vqvae", "none"]:
    eid = f"E3_{quant_method}"
    quant_bits = {"uniform8": 8, "uniform4": 4, "vqvae": 0, "none": 0}[quant_method]
    cfg = flow_cfg(
        eid,
        f"Quantisation Method — {quant_method}",
        {
            "encoder": {"quant_bits": quant_bits, "quant_method": quant_method},
            "training": {"n_epochs": 60},
        },
    )
    write_config(cfg, f"configs/comparison/{eid}.yaml")

# E4 — Data augmentation impact
for aug in ["none", "basic", "full", "mixup"]:
    eid = f"E4_{aug}"
    cfg = flow_cfg(
        eid,
        f"Data Augmentation Impact — {aug}",
        {
            "dataset": {"augmentation": aug},
            "training": {"n_epochs": 60},
        },
    )
    write_config(cfg, f"configs/comparison/{eid}.yaml")

print("\nDone! All configs generated.")
