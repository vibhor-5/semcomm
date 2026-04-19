"""
run_session.py — Runs all experiments for a given session number from the
Experiment Catalogue, in order. Designed for Kaggle/Colab where you have
a fixed GPU time budget per session.

Usage:
    # Run session 3 (B1 flow path comparison):
    python scripts/run_session.py --session 3

    # Resume a session that was interrupted:
    python scripts/run_session.py --session 3 --resume

    # Just list what a session would run (dry run):
    python scripts/run_session.py --session 3 --dry_run

    # Override output root (defaults to <repo>/outputs):
    python scripts/run_session.py --session 3 --output_dir ./outputs

Sessions map directly to the Experiment Catalogue table:
    Session 1  → A1, A3        (baselines, CPU)
    Session 2  → A2            (DeepJSCC)
    Session 3  → B1            (flow path compare)
    Session 4  → B2            (depth ablation)
    Session 5  → B3            (conditioning ablation)
    Session 6  → B4            (rate sweep, all k)
    Session 7  → B5, B6        (SNR sweep + channel-aware)
    Session 8  → B7, B9, B10   (guidance, ODE steps, fading)
    Session 9  → B8            (TinyImageNet, flow)
    Session 10 → C1            (diffusion conditioning)
    Session 11 → C3            (diffusion rate sweep)
    Session 12 → C2, C4, C5, C10  (eval-only diffusion)
    Session 13 → C6, C7        (channel-aware + latent space)
    Session 14 → C8            (semantic loss)
    Session 15 → C9            (TinyImageNet, diffusion)
    Session 16 → D1-D5         (head-to-head plots)
    Session 17 → E1, E2        (encoder + token ablations)
    Session 18 → E3, E4        (quant + augmentation ablations)
"""

import os
import sys
import argparse
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# ── session → list of config paths (relative to repo root) ──────────────────

SESSION_MAP = {
    1: [
        "configs/baselines/A1_eval_calibration.yaml",
        "configs/baselines/A3_jpeg.yaml",
    ],
    2: [
        "configs/baselines/A2_deepjscc_k64.yaml",
        "configs/baselines/A2_deepjscc_k128.yaml",
    ],
    3: [
        "configs/flow/B1_linear.yaml",
        "configs/flow/B1_ot.yaml",
    ],
    4: [
        "configs/flow/B2_nblocks3.yaml",
        "configs/flow/B2_nblocks5.yaml",
        "configs/flow/B2_nblocks8.yaml",
    ],
    5: [
        "configs/flow/B3_none.yaml",
        "configs/flow/B3_clip.yaml",
        "configs/flow/B3_clip_noisy.yaml",
        "configs/flow/B3_clip_class.yaml",
    ],
    6: [
        "configs/flow/B4_k32.yaml",
        "configs/flow/B4_k64.yaml",
        "configs/flow/B4_k128.yaml",
        "configs/flow/B4_k256.yaml",
    ],
    7: [
        "configs/flow/B5_snr_sweep.yaml",
        "configs/flow/B6_random_snr.yaml",
        "configs/flow/B6_random_snr_token.yaml",
    ],
    8: [
        "configs/flow/B7_w1p0.yaml",
        "configs/flow/B7_w1p5.yaml",
        "configs/flow/B7_w2p0.yaml",
        "configs/flow/B7_w3p0.yaml",
        "configs/flow/B9_steps5.yaml",
        "configs/flow/B9_steps10.yaml",
        "configs/flow/B9_steps25.yaml",
        "configs/flow/B9_steps50.yaml",
        "configs/flow/B9_steps100.yaml",
        "configs/flow/B9_steps200.yaml",
        "configs/flow/B10_awgn.yaml",
        "configs/flow/B10_rayleigh.yaml",
        "configs/flow/B10_burst.yaml",
    ],
    9: [
        "configs/flow/B8_tinyimagenet.yaml",
    ],
    10: [
        "configs/diffusion/C1_concat.yaml",
        "configs/diffusion/C1_cross_attention.yaml",
        "configs/diffusion/C1_adain.yaml",
        "configs/diffusion/C1_none.yaml",
    ],
    11: [
        "configs/diffusion/C3_k32.yaml",
        "configs/diffusion/C3_k64.yaml",
        "configs/diffusion/C3_k128.yaml",
        "configs/diffusion/C3_k256.yaml",
    ],
    12: [
        "configs/diffusion/C2_steps5.yaml",
        "configs/diffusion/C2_steps10.yaml",
        "configs/diffusion/C2_steps25.yaml",
        "configs/diffusion/C2_steps50.yaml",
        "configs/diffusion/C2_steps100.yaml",
        "configs/diffusion/C2_steps200.yaml",
        "configs/diffusion/C4_w1p0.yaml",
        "configs/diffusion/C4_w1p5.yaml",
        "configs/diffusion/C4_w2p0.yaml",
        "configs/diffusion/C4_w3p0.yaml",
        "configs/diffusion/C4_w5p0.yaml",
        "configs/diffusion/C5_snr_sweep.yaml",
        "configs/diffusion/C10_awgn.yaml",
        "configs/diffusion/C10_rayleigh.yaml",
        "configs/diffusion/C10_burst.yaml",
    ],
    13: [
        "configs/diffusion/C6_random_snr.yaml",
        "configs/diffusion/C6_random_snr_token.yaml",
        "configs/diffusion/C7_pixel.yaml",
        "configs/diffusion/C7_vae_latent.yaml",
    ],
    14: [
        "configs/diffusion/C8_mse_only.yaml",
        "configs/diffusion/C8_mse_clip.yaml",
        "configs/diffusion/C8_mse_vgg.yaml",
        "configs/diffusion/C8_mse_clip_vgg.yaml",
    ],
    15: [
        "configs/diffusion/C9_tinyimagenet.yaml",
    ],
    16: [
        "configs/comparison/D1_rate_curves.yaml",
        "configs/comparison/D2_speed_quality.yaml",
        "configs/comparison/D3_snr_robustness.yaml",
        "configs/comparison/D4_recon_gallery.yaml",
        "configs/comparison/D5_clip_dist.yaml",
    ],
    17: [
        "configs/comparison/E1_cnn.yaml",
        "configs/comparison/E1_resnet18.yaml",
        "configs/comparison/E1_mobilenetv2.yaml",
        "configs/comparison/E2_clip.yaml",
        "configs/comparison/E2_learned.yaml",
        "configs/comparison/E2_class_onehot.yaml",
        "configs/comparison/E2_none.yaml",
    ],
    18: [
        "configs/comparison/E3_uniform8.yaml",
        "configs/comparison/E3_uniform4.yaml",
        "configs/comparison/E3_vqvae.yaml",
        "configs/comparison/E3_none.yaml",
        "configs/comparison/E4_none.yaml",
        "configs/comparison/E4_basic.yaml",
        "configs/comparison/E4_full.yaml",
        "configs/comparison/E4_mixup.yaml",
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Run all experiments for a specific session."
    )
    parser.add_argument(
        "--session",
        type=int,
        required=True,
        help="Session number (1-18) matching the Experiment Catalogue",
    )
    parser.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "outputs"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume to each experiment (continue from last checkpoint)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would run without executing anything",
    )
    args = parser.parse_args()

    configs = SESSION_MAP.get(args.session)
    if configs is None:
        print(f"Unknown session {args.session}. Valid sessions: 1–{max(SESSION_MAP)}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  SESSION {args.session}  ({len(configs)} experiments)")
    print(f"{'='*60}")
    for c in configs:
        print(f"  • {c}")
    print()

    if args.dry_run:
        print("Dry run — nothing executed.")
        return

    runner = os.path.join(REPO_ROOT, "scripts", "run_experiment.py")
    failed = []

    for i, cfg_path in enumerate(configs, 1):
        abs_cfg = os.path.join(REPO_ROOT, cfg_path)
        if not os.path.exists(abs_cfg):
            print(f"[{i}/{len(configs)}] SKIP — config not found: {abs_cfg}")
            failed.append(cfg_path)
            continue

        print(f"\n[{i}/{len(configs)}] Running: {cfg_path}")
        print("-" * 60)

        cmd = [
            sys.executable,
            runner,
            "--config",
            abs_cfg,
            "--data_dir",
            args.data_dir,
            "--output_dir",
            args.output_dir,
        ]
        if args.resume:
            cmd.append("--resume")

        ret = subprocess.run(cmd, cwd=REPO_ROOT)
        if ret.returncode != 0:
            print(
                f"\n[WARNING] Experiment failed (exit code {ret.returncode}): {cfg_path}"
            )
            failed.append(cfg_path)

    print(f"\n{'='*60}")
    print(f"Session {args.session} complete.")
    print(f"  Passed : {len(configs) - len(failed)}/{len(configs)}")
    if failed:
        print(f"  Failed : {failed}")
    print(f"  Results: {os.path.join(args.output_dir, 'summary_table.csv')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
