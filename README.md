# Semantic Communication Research — Free GPU Edition

[![Unit Tests](https://github.com/vibhor/semcomm/actions/workflows/ci.yml/badge.svg)](https://github.com/vibhor/semcomm/actions/workflows/ci.yml)

> **Hardware target:** Google Colab T4 (15 GB VRAM) · Kaggle T4 / P100 (16 GB VRAM)  
> **Training budget:** ≤ 8 hours per session · fp16 mixed precision throughout

A research codebase for **Semantic Communication** experiments comparing
Flow-Matching and Diffusion-based generative decoders against the DeepJSCC and
JPEG baselines on CIFAR-10 / TinyImageNet.

---

## Repository Structure

```
semcomm/
├── configs/                  # One YAML per experiment
│   ├── baselines/            # A1, A2 (DeepJSCC), A3 (JPEG)
│   ├── flow/                 # B1–B10 (Flow-Matching)
│   ├── diffusion/            # C1–C10 (Diffusion)
│   └── comparison/           # D1–D5 (head-to-head), E1–E4 (ablations)
├── data/
│   ├── datasets.py           # SemCommDataset (CIFAR-10/100, TinyImageNet, COCO subset)
│   ├── augmentations.py      # Transform pipelines + MixUp collate
│   └── synthetic_channel.py  # AWGN, Rayleigh, BurstErasure channel models
├── models/
│   ├── encoder.py            # CNN / ResNet-18 / MobileNetV2 + CLIP token + quantisation
│   ├── flow_decoder.py       # CNF decoder (linear / OT path, CFG, SNR conditioning)
│   ├── diffusion_decoder.py  # DDPM U-Net decoder (concat / cross-attn / AdaIN)
│   └── baselines/
│       └── deepjscc.py       # DeepJSCC baseline (frozen after A2)
├── losses/
│   ├── semantic_loss.py      # CLIPLoss (1 − cosine similarity)
│   └── perceptual_loss.py    # VGGPerceptualLoss (relu2_2 features)
├── training/
│   ├── trainer.py            # Trainer: fp16 loop, validation, checkpointing, W&B
│   ├── train_flow.py         # Entry point for B-group experiments
│   └── train_diffusion.py    # Entry point for C-group experiments
├── evaluation/
│   ├── metrics.py            # CLIP sim, LPIPS, FID, PSNR, SSIM, BPP, timing
│   ├── evaluate.py           # run_evaluation() — loads checkpoint, computes metrics
│   └── visualise.py          # Publication-ready plots (PDF + PNG)
├── utils/
│   ├── seed.py               # set_seed()
│   ├── device.py             # get_device() — CUDA / MPS / CPU
│   └── checkpoint.py         # save_checkpoint / load_checkpoint
├── scripts/
│   ├── generate_configs.py   # Generate ALL experiment YAML configs
│   └── run_experiment.py     # Generic CLI runner (train + eval)
├── tests/                    # Pytest suite (CPU-only, no dataset download)
├── notebooks/                # One Jupyter notebook per session
└── requirements.txt
```

---

## Quick Start

### 1 — Install dependencies

```bash
# GPU environment (Colab / Kaggle)
pip install -r requirements.txt

# CPU-only (local dev / CI)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy Pillow matplotlib seaborn tqdm pyyaml einops torchdiffeq pot lpips torchmetrics[image]
```

### 2 — Generate all experiment configs

```bash
python scripts/generate_configs.py
```

This creates 60+ YAML files under `configs/`.

### 3 — Run an experiment

```bash
python scripts/run_experiment.py \
    --config configs/flow/B1_linear.yaml \
    --data_dir /content/drive/MyDrive/semcomm/data \
    --output_dir /content/drive/MyDrive/semcomm/outputs
```

Add `--eval_only` to skip training and evaluate the saved `best.pt` checkpoint.

### 4 — Run tests

```bash
pytest tests/ -v --tb=short -x
```

---

## Experiment Groups

| Group | ID Range | Description | GPU Time |
|-------|----------|-------------|----------|
| A — Baselines | A1–A3 | DeepJSCC, JPEG, eval calibration | ~1.5 hr |
| B — Flow | B1–B10 | Flow-matching decoder ablations + sweeps | ~35 hr |
| C — Diffusion | C1–C10 | Diffusion decoder ablations + sweeps | ~55 hr |
| D — Compare | D1–D5 | Head-to-head plots (eval only) | ~3 hr |
| E — Ablations | E1–E4 | Encoder / token / quantisation / augment | ~25 hr |

**Total estimated GPU hours: ~115 hr** (~4 Kaggle weeks at 30 hr/week).

---

## Key Design Decisions

- **fp16 mixed precision** via `torch.cuda.amp.GradScaler` throughout training.
- **Checkpoints every 5 epochs** to survive free-tier session disconnects.
- **CLIP tokens pre-computed and cached** to disk — avoids re-running ViT-B/32 per step.
- **Straight-through estimator** for gradient flow through scalar quantisation.
- **OT path** for flow-matching uses `pot.emd` on mini-batches (64 samples).
- **Classifier-free guidance** with 10 % null-token dropout during training.
- **DiffusionDecoder** uses `diffusers.UNet2DConditionModel` for proven architecture.

---

## Metrics

| Metric | Higher / Lower | Notes |
|--------|----------------|-------|
| CLIP Similarity | ↑ | Frozen ViT-B/32 cosine similarity |
| LPIPS | ↓ | AlexNet perceptual similarity |
| FID | ↓ | Needs ≥ 2,048 images |
| PSNR (dB) | ↑ | Computed analytically from MSE |
| SSIM | ↑ | Structural similarity |
| BPP | — | `latent_dim × quant_bits / (H × W)` |
| ms / image | ↓ | Measured on T4 via CUDA events |

---

## Datasets

| Dataset | Size | Resolution | Used In |
|---------|------|------------|---------|
| CIFAR-10 | 163 MB | 32 × 32 | All A/B/C/D/E experiments |
| CIFAR-100 | 169 MB | 32 × 32 | More-class ablations |
| TinyImageNet | 237 MB | 64 × 64 | B8, C9 scale-up |
| COCO val2017 (2k) | ~800 MB | → 64 × 64 | E4 OOD evaluation |

---

## References

- Bourtsoulatze et al. (2019) — DeepJSCC
- Song et al. (2025) — SEDIC (published numbers only)
- Grassucci et al. (2025) — Q-GESCO (published numbers only)
- Lipman et al. (2022) — Flow Matching for Generative Modeling
- Ho et al. (2020) — Denoising Diffusion Probabilistic Models
- Song et al. (2020) — DDIM
