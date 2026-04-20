"""
run_experiment.py — Generic experiment runner for semcomm.

Works on Kaggle, Google Colab, and locally without any path changes.
All outputs are stored INSIDE the cloned repo directory by default.

Usage (Kaggle / Colab / local):
    python scripts/run_experiment.py --config configs/flow/B1_linear.yaml

    # Override dirs if needed:
    python scripts/run_experiment.py \\
        --config configs/flow/B1_linear.yaml \\
        --data_dir ./data \\
        --output_dir ./outputs

    # Resume after a disconnect (finds the latest saved checkpoint automatically):
    python scripts/run_experiment.py --config configs/flow/B1_linear.yaml --resume

    # Eval only (no training):
    python scripts/run_experiment.py --config configs/flow/B1_linear.yaml --eval_only
"""

import os
import sys
import csv
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

# ── repo root is the parent of this script's directory ──────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from utils.seed import set_seed  # noqa: E402
from utils.device import get_device  # noqa: E402
from data.datasets import SemCommDataset  # noqa: E402
from data.augmentations import get_transforms  # noqa: E402
from models.encoder import Encoder  # noqa: E402
from data.synthetic_channel import (  # noqa: E402
    AWGNChannel,
    RayleighChannel,
    BurstErasureChannel,
)
from models.flow_decoder import FlowDecoder  # noqa: E402
from models.diffusion_decoder import DiffusionDecoder  # noqa: E402
from training.trainer import Trainer  # noqa: E402


# ── model builders ───────────────────────────────────────────────────────────


def build_channel(channel_cfg):
    ctype = channel_cfg.get("type", "awgn").lower()
    snr_db = channel_cfg.get("snr_db")
    snr_range = channel_cfg.get("snr_range", (0, 20))
    return_snr = channel_cfg.get("snr_conditioning", False)
    if ctype == "awgn":
        return AWGNChannel(snr_db=snr_db, snr_range=snr_range, return_snr=return_snr)
    elif ctype == "rayleigh":
        return RayleighChannel(
            snr_db=snr_db, snr_range=snr_range, return_snr=return_snr
        )
    elif ctype == "burst":
        return BurstErasureChannel(
            erasure_prob=channel_cfg.get("erasure_prob", 0.2),
            snr_db=snr_db if snr_db is not None else 10,
            return_snr=return_snr,
        )
    raise ValueError(f"Unknown channel type '{ctype}'")


def build_decoder(decoder_cfg, encoder_cfg, dataset_cfg):
    dtype = decoder_cfg.get("type", "flow").lower()
    image_size = dataset_cfg.get("image_size", 32)
    token_dim = 512 if encoder_cfg.get("semantic_token_type") == "clip" else 0
    if dtype == "flow":
        return FlowDecoder(
            latent_dim=encoder_cfg["latent_dim"],
            token_dim=token_dim,
            image_size=image_size,
            n_channels=decoder_cfg.get("n_channels", 128),
            n_blocks=decoder_cfg.get("n_blocks", 5),
            path_type=decoder_cfg.get("path_type", "linear"),
            use_guidance=decoder_cfg.get("use_guidance", False),
            snr_conditioning=decoder_cfg.get("snr_conditioning", False),
        )
    elif dtype == "diffusion":
        return DiffusionDecoder(
            latent_dim=encoder_cfg["latent_dim"],
            token_dim=token_dim,
            image_size=image_size,
            conditioning=decoder_cfg.get("conditioning", "concat"),
            base_channels=decoder_cfg.get("base_channels", 128),
            snr_conditioning=decoder_cfg.get("snr_conditioning", False),
        )
    elif dtype == "deepjscc":
        from models.baselines.deepjscc import DeepJSCCDecoder

        return DeepJSCCDecoder(latent_dim=encoder_cfg.get("latent_dim", 128))
    raise ValueError(f"Unknown decoder type '{dtype}'")


# ── resume helper ────────────────────────────────────────────────────────────


def find_latest_checkpoint(ckpt_dir: str):
    """Return (path, epoch) of the most recent epoch_N.pt, or (None, 0)."""
    if not os.path.isdir(ckpt_dir):
        return None, 0
    candidates = []
    for fname in os.listdir(ckpt_dir):
        if fname.startswith("epoch_") and fname.endswith(".pt"):
            try:
                ep = int(fname.replace("epoch_", "").replace(".pt", ""))
                candidates.append((ep, os.path.join(ckpt_dir, fname)))
            except ValueError:
                pass
    if not candidates:
        return None, 0
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][0]


# ── results CSV ──────────────────────────────────────────────────────────────


def append_summary_csv(output_dir: str, row: dict):
    """Append one result row to outputs/summary_table.csv."""
    csv_path = os.path.join(output_dir, "summary_table.csv")
    fieldnames = [
        "experiment_id",
        "clip",
        "lpips",
        "fid",
        "psnr",
        "ssim",
        "bpp",
        "sampling_time_ms",
        "n_epochs_trained",
        "gpu",
        "date",
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results appended to {csv_path}")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Semcomm experiment runner — works on Kaggle, Colab, and locally."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config (relative to repo root)",
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(REPO_ROOT, "data"),
        help="Dataset root  [default: <repo>/data]",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(REPO_ROOT, "outputs"),
        help="Outputs root  [default: <repo>/outputs]",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; load best.pt and evaluate",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the latest saved checkpoint"
    )
    args = parser.parse_args()

    # ── resolve config path (accept relative-to-cwd or relative-to-repo) ────
    if os.path.isabs(args.config):
        config_path = args.config
    elif os.path.exists(args.config):
        config_path = os.path.abspath(args.config)
    else:
        config_path = os.path.join(REPO_ROOT, args.config)

    device = get_device()

    from utils.device import print_device_info

    print(f"\n{'='*60}")
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print_device_info()
    print(f"  Config     : {config_path}")
    print(f"  Data root  : {args.data_dir}")
    print(f"  Output root: {args.output_dir}")
    print(f"{'='*60}\n")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    # ── override ALL file paths at runtime ───────────────────────────────────
    # Config files store placeholder paths; we always override them here.
    os.makedirs(args.data_dir, exist_ok=True)
    cfg["dataset"]["root"] = args.data_dir

    clip_cache = os.path.join(
        args.data_dir,
        f"{cfg['dataset']['name']}_clip_train.pt",
    )
    cfg["dataset"]["clip_cache_path"] = clip_cache

    exp_dir = os.path.join(args.output_dir, cfg["experiment_id"])
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    cfg["logging"]["checkpoint_dir"] = ckpt_dir

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "figures"), exist_ok=True)

    # ── precompute CLIP token cache (only once) ───────────────────────────────
    if cfg["encoder"].get("semantic_token_type") == "clip":
        if not os.path.exists(clip_cache):
            print(f"Precomputing CLIP tokens → {clip_cache}  (one-time, ~5 min on T4)")
            SemCommDataset.precompute_clip_tokens(
                root=args.data_dir,
                dataset_name=cfg["dataset"]["name"],
                image_size=cfg["dataset"]["image_size"],
                cache_path=clip_cache,
                device=device,
            )

    # ── data loaders ─────────────────────────────────────────────────────────
    train_trans, val_trans = get_transforms(cfg["dataset"]["image_size"])
    aug_name = cfg["dataset"].get("augmentation", "basic")
    val_split = "test" if cfg["dataset"]["name"] in ("cifar10", "cifar100") else "val"

    train_ds = SemCommDataset(
        args.data_dir,
        "train",
        cfg["dataset"]["name"],
        cfg["dataset"]["image_size"],
        clip_cache_path=clip_cache,
        transform=train_trans.get(aug_name, train_trans["basic"]),
    )
    val_ds = SemCommDataset(
        args.data_dir,
        val_split,
        cfg["dataset"]["name"],
        cfg["dataset"]["image_size"],
        clip_cache_path=clip_cache,
        transform=val_trans,
    )

    num_workers = cfg["training"].get("num_workers", 2)
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── build models ─────────────────────────────────────────────────────────
    encoder = Encoder(**cfg["encoder"], device=device)
    channel = build_channel(cfg["channel"])
    decoder = build_decoder(cfg["decoder"], cfg["encoder"], cfg["dataset"])

    # ── train ────────────────────────────────────────────────────────────────
    n_epochs = cfg["training"]["n_epochs"]
    start_epoch = 0

    if not args.eval_only and n_epochs > 0:
        trainer = Trainer(
            encoder, channel, decoder, cfg["loss"], cfg, drive_root=args.output_dir
        )

        if args.resume:
            resume_path, start_epoch = find_latest_checkpoint(ckpt_dir)
            if resume_path:
                from utils.checkpoint import load_checkpoint

                loaded_ep, _ = load_checkpoint(
                    resume_path, decoder, trainer.optimizer, trainer.scaler
                )
                ckpt = torch.load(resume_path, map_location="cpu")
                if "encoder_state" in ckpt:
                    encoder.load_state_dict(ckpt["encoder_state"])
                start_epoch = loaded_ep
                print(f"Resumed from epoch {start_epoch}  ({resume_path})")
            else:
                print("No checkpoint found — starting from scratch.")

        remaining = n_epochs - start_epoch
        if remaining > 0:
            print(
                f"Training {cfg['experiment_id']}  "
                f"(epochs {start_epoch + 1} – {n_epochs})"
            )
            trainer.run(train_loader, val_loader, n_epochs=remaining)
            print("Training complete.\n")
        else:
            print(f"Already trained {n_epochs} epochs — skipping training.")

    # ── evaluate ─────────────────────────────────────────────────────────────
    from evaluation.evaluate import run_evaluation
    import datetime

    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    if not os.path.exists(best_ckpt):
        best_ckpt, _ = find_latest_checkpoint(ckpt_dir)

    # Proceed to evaluation if we found a checkpoint OR if n_epochs is 0
    # (n_epochs=0 indicates a baseline experiment like JPEG that doesn't train).
    if (best_ckpt and os.path.exists(best_ckpt)) or (n_epochs == 0):
        if best_ckpt and os.path.exists(best_ckpt):
            print(f"Evaluating checkpoint: {best_ckpt}")
        else:
            print(f"Evaluating baseline (no checkpoint): {cfg['experiment_id']}")
            best_ckpt = None

        metrics = run_evaluation(
            cfg,
            best_ckpt,
            val_loader,
            device,
            encoder,
            channel,
            decoder,
        )
        print(f"\nResults for {cfg['experiment_id']}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<22}: {v:.4f}")
            else:
                print(f"  {k:<22}: {v}")

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        append_summary_csv(
            args.output_dir,
            {
                "experiment_id": cfg["experiment_id"],
                "clip": metrics.get("clip", ""),
                "lpips": metrics.get("lpips", ""),
                "fid": metrics.get("fid", ""),
                "psnr": metrics.get("psnr", ""),
                "ssim": metrics.get("ssim", ""),
                "bpp": metrics.get("bpp", ""),
                "sampling_time_ms": "",
                "n_epochs_trained": n_epochs,
                "gpu": gpu_name,
                "date": datetime.date.today().isoformat(),
            },
        )
    else:
        print(f"No checkpoint found in {ckpt_dir} — skipping evaluation.")


if __name__ == "__main__":
    main()
