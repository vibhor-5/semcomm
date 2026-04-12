# Semantic Communication Research: Experiment Catalogue
### (Free GPU Edition — Google Colab T4 / Kaggle T4/P100)

> **Hardware context:** All experiments run on free-tier cloud GPUs.
> - **Google Colab Free:** NVIDIA T4 (15 GB VRAM), ~12-hour session limit, ~13 GB RAM. Mount Google Drive for persistence.
> - **Kaggle Free:** NVIDIA T4 or P100 (16 GB VRAM), 30 GPU hours/week, 12-hour sessions. Save outputs as Kaggle Dataset versions.
> - **Strategy:** Every training run fits in under 8 hours (leaves buffer before session cut-off). Save checkpoints every 5 epochs. Use fp16 mixed precision throughout.

---

## Hardware Budget

| Platform | GPU | VRAM | Session Limit | Weekly GPU Limit | Persistence |
|----------|-----|------|--------------|-----------------|-------------|
| Colab Free | T4 | 15 GB | ~12 hr | Unmetered (but disconnects) | Google Drive |
| Kaggle Free | T4 / P100 | 16 GB | 12 hr | 30 GPU-hr | Kaggle Datasets |

**Rules of thumb for this project:**
- Max image size for training: **64×64** (good batch sizes fit in VRAM)
- Max batch size: **64** for generative decoders, **128** for plain CNNs
- Use **fp16 mixed precision** everywhere via `torch.cuda.amp.GradScaler`
- Save a checkpoint **every 5 epochs** — free sessions disconnect without warning
- Every run must finish in **≤ 8 hours** (leaves ~4 hr buffer)
- Store checkpoints in Google Drive (Colab) or output as Kaggle Dataset (Kaggle)

---

## Research Directions Overview

| # | Direction | Role | Priority |
|---|-----------|------|----------|
| 1 | Flow-Matching Generative Semantic Image Comm | **Primary research** | High |
| 2 | Diffusion-Based Generative Semantic Image Comm | **Primary research** | High |
| 3 | DeepJSCC (Deep Joint Source-Channel Coding) | **Baseline only — not extended** | — |

---

## Baseline Systems (Reference Only — Not Extended)

Run once, record numbers, and freeze. Do not tune or modify further.

| Baseline | Paper | Handling on Free GPU |
|----------|-------|---------------------|
| DeepJSCC (CNN autoencoder) | Bourtsoulatze et al. 2019 | Train from scratch — fast (~1 hr on T4) |
| SEDIC | Song et al., arXiv 2503.00399, 2025 | Use **published numbers only** — too costly to retrain |
| Q-GESCO | Grassucci et al. 2025 | Use **published numbers only** — too costly to retrain |
| JPEG + ideal channel | Standard | CPU only, no GPU needed |
| BPG + ideal channel | Standard | CPU only, no GPU needed |

> SEDIC and Q-GESCO require multi-GPU days of training. On free GPUs, use their reported metrics for comparison and only verify your own evaluation code is consistent using JPEG as a sanity check.

---

## Datasets

Chosen for fast download, small disk footprint, and viability on free-tier storage (~20 GB Colab disk / Kaggle dataset limits).

| Dataset | Disk Size | Resolution | Role | How to Load |
|---------|----------|------------|------|-------------|
| CIFAR-10 | 163 MB | 32×32 | **Primary for all experiments** | `torchvision.datasets.CIFAR10(download=True)` |
| CIFAR-100 | 169 MB | 32×32 | More-class ablations | `torchvision.datasets.CIFAR100(download=True)` |
| TinyImageNet | 237 MB | 64×64 | Scale-up experiments (B8, C9) | Download once, save to Drive or Kaggle Dataset |
| COCO val2017 (2k subset) | ~800 MB | → 64×64 | Out-of-distribution semantic eval only | Sample 2k images, save to Drive |
| Synthetic Channel Data | None | N/A | Generated on-the-fly in code | — |

**Splits:** CIFAR standard (45k train / 5k val / 10k test). TinyImageNet standard (80k/10k/10k). COCO subset: 1,600 train / 200 val / 200 test, seed=42.

**Colab tip:** After first download, copy datasets to your Google Drive folder and load from there in future sessions — avoids re-downloading every session.

**Kaggle tip:** Upload CIFAR-10 and TinyImageNet as private Kaggle Datasets once; attach them to every notebook so they appear pre-loaded.

---

## Evaluation Metrics

| Metric | Type | Notes |
|--------|------|-------|
| CLIP Similarity | Semantic | Frozen `ViT-B/32` — fits in T4 VRAM alongside models |
| LPIPS | Perceptual | AlexNet-based, lightweight — runs in eval loop |
| FID | Perceptual | Needs ≥2,048 images; use full CIFAR-10 test (10k real) + 5k generated |
| PSNR | Pixel-level | Computed in eval loop, free |
| SSIM | Pixel-level | Computed in eval loop, free |
| Bits-per-pixel (bpp) | Rate | `latent_dim × quant_bits / (H × W)` — analytic |
| Sampling Time (ms/image) | Efficiency | Measure on GPU with `torch.cuda.Event` timing |
| SNR Threshold (dB) | Robustness | Minimum SNR for CLIP ≥ 0.80 |

---

## Experiment Group A: Baselines

### A1 — Evaluation Pipeline Calibration
- **Goal:** Confirm CLIP similarity, LPIPS, and PSNR functions are correctly implemented before using them to judge research models.
- **Method:** Pass identical image pairs (CLIP → ~1.0, PSNR → very high). Then pass JPEG-compressed pairs at known quality levels and verify PSNR matches reference values.
- **Dataset:** CIFAR-10 test (10k images).
- **GPU needed:** No — CPU only.
- **Time:** 15 minutes.

### A2 — DeepJSCC Baseline (CNN Autoencoder)
- **Goal:** Train a vanilla CNN autoencoder (no semantic loss) on CIFAR-10 to produce the JSCC reference curve.
- **Dataset:** CIFAR-10 full (32×32).
- **Model:** 4-layer strided CNN encoder → latent k → 4-layer transposed CNN decoder.
- **Latent sizes:** k ∈ {64, 128}.
- **Training:** 50 epochs, batch=128, Adam lr=1e-3, fp16. ~25 min per k on T4.
- **Metrics:** Classification accuracy (frozen ResNet-18 classifier on reconstructions) at SNR ∈ {0, 5, 10, 20} dB. PSNR, SSIM, CLIP vs bpp.
- **Time:** ~1 hour total (both k values in one session).

### A3 — Traditional Compression Baseline
- **Goal:** JPEG reference curves.
- **Dataset:** CIFAR-10 test (10k images).
- **Method:** JPEG quality ∈ {5, 10, 25, 50, 75} via Pillow. Record bpp, PSNR, LPIPS, CLIP per quality setting.
- **GPU needed:** No — CPU only.
- **Time:** 15–20 minutes.

---

## Experiment Group B: Flow-Matching Decoder (Primary Research)

**Default setup for all B experiments unless stated:**
- Dataset: CIFAR-10 (32×32)
- Device: CUDA T4, fp16 mixed precision
- Batch size: 64
- Checkpoint every 5 epochs to Drive/Kaggle

### B1 — Flow Path Comparison: Linear vs Optimal Transport (OT)
- **Goal:** Determine which interpolation path gives better flow-matching training dynamics and final quality.
- **Dataset:** CIFAR-10.
- **Architecture:** CNN encoder (k=128) → frozen CLIP ViT-B/32 semantic token (512-d) → CNF decoder (5 ResNet blocks, 128 hidden channels).
- **Variables:** Path type ∈ {linear, ot}.
- **Fixed:** lr=1e-4, batch=64, epochs=100, SNR=10 dB, no guidance.
- **Metrics:** CLIP, LPIPS, training loss at epochs {10, 50, 100}.
- **Time:** ~1.5 hr per run × 2 = **3 hr total**. One Kaggle session.
- **Expected outcome:** OT path converges ~20% faster; final CLIP within 2–4%.

### B2 — CNF Architecture Depth Ablation
- **Goal:** Select the number of flow layers that best balances quality and decoding speed on T4.
- **Dataset:** CIFAR-10.
- **Variables:** n_blocks ∈ {3, 5, 8}.
- **Fixed:** Best path from B1, lr=1e-4, batch=64, epochs=60, k=128.
- **Metrics:** CLIP, LPIPS, sampling time (ms/image on T4).
- **Time:** ~1 hr per run × 3 = **3 hr total**. One session.

### B3 — Semantic Conditioning Ablation
- **Goal:** Quantify the contribution of the CLIP token to reconstruction quality.
- **Dataset:** CIFAR-10.
- **Variables:** Conditioning ∈ {none (unconditional CNF), CLIP token only, CLIP token + noisy latent, CLIP token + noisy latent + class label}.
- **Fixed:** Best architecture from B1/B2, k=128, SNR=10 dB, epochs=80, batch=64.
- **Metrics:** CLIP, LPIPS, FID (5k generated images).
- **Time:** ~1.5 hr per run × 4 = **6 hr total**. One Kaggle session.

### B4 — Bitrate Sweep (Rate–Semantic Curve, Flow)
- **Goal:** Core result — CLIP vs bpp curve for the flow decoder across bitrate levels.
- **Dataset:** CIFAR-10.
- **Variables:** k ∈ {32, 64, 128, 256} (bpp = k×8/(32×32) = {0.25, 0.5, 1.0, 2.0}).
- **Fixed:** Best config from B1–B3, OT path, SNR=10 dB, epochs=100, batch=64.
- **Metrics:** CLIP, LPIPS, PSNR vs bpp.
- **Time:** ~2 hr per k × 4 = **8 hr total**. One 12-hr Kaggle session (run all k sequentially).

### B5 — Channel Robustness: AWGN SNR Sweep (Flow)
- **Goal:** Evaluate the flow decoder across noise levels. No new training — eval only.
- **Dataset:** CIFAR-10 test. Load best B4 checkpoint (k=128).
- **Variables:** SNR ∈ {0, 3, 5, 7, 10, 15, 20} dB.
- **Metrics:** CLIP vs SNR, LPIPS vs SNR, SNR threshold for CLIP ≥ 0.80.
- **Time:** **30 min** (eval only).

### B6 — Channel-Aware Flow Training (Multi-SNR)
- **Goal:** Train the flow decoder with randomly sampled SNR per batch and an SNR conditioning token, to improve robustness beyond single-SNR training.
- **Dataset:** CIFAR-10.
- **Variables:** Regime ∈ {random SNR ∈ [0,20] dB (no token), random SNR + SNR embedding token}.
- **Fixed:** Best architecture from B1/B2, k=128, epochs=100, batch=64.
- **Metrics:** CLIP vs SNR evaluated at {0, 5, 10, 20} dB. Compare to B5.
- **Time:** ~2 hr per regime × 2 = **4 hr total**. One session.

### B7 — Classifier-Free Guidance Weight Sweep (Flow)
- **Goal:** Find the optimal guidance scale w for the conditional flow decoder.
- **Dataset:** CIFAR-10 (eval only — vary w at sample time, no retraining).
- **Variables:** w ∈ {1.0, 1.5, 2.0, 3.0}. Training uses 10% null-token dropout.
- **Fixed:** Best model from B6, k=128, SNR=10 dB.
- **Metrics:** CLIP, FID (5k), LPIPS.
- **Time:** **30 min** (eval only with different w values).

### B8 — Scale-Up: TinyImageNet 64×64 (Flow)
- **Goal:** Validate best flow configuration on higher resolution and more diverse data than CIFAR-10.
- **Dataset:** TinyImageNet (100k images, 64×64).
- **Fixed:** Best config from B1–B7. Increase image_size=64, n_channels=128, n_blocks=5.
- **Training:** 80 epochs, batch=64, lr=1e-4, fp16.
- **Metrics:** CLIP, LPIPS, FID, PSNR vs bpp.
- **Time:** **~5–6 hr**. One Kaggle session.

### B9 — ODE Step Count vs Quality and Speed (Flow)
- **Goal:** Find the minimum ODE steps that preserve quality — crucial for practical decoding speed.
- **Dataset:** CIFAR-10 test (eval only — load best B4 checkpoint).
- **Variables:** Steps ∈ {5, 10, 25, 50, 100, 200}.
- **Metrics:** CLIP, LPIPS, FID (5k), ms/image on T4.
- **Time:** **45 min** (eval only).
- **Expected outcome:** 25–50 steps reaches ≥95% of 200-step quality at ~4× speed — key advantage over diffusion.

### B10 — Fading Channel Robustness (Flow)
- **Goal:** Test flow decoder under more realistic channel conditions than AWGN.
- **Dataset:** CIFAR-10 test (eval only — load best B6 checkpoint).
- **Variables:** Channel ∈ {AWGN 10 dB, Rayleigh fading 10 dB avg SNR, burst erasure 20%}.
- **Metrics:** CLIP, LPIPS per channel type.
- **Time:** **30 min**.

---

## Experiment Group C: Diffusion Decoder (Primary Research)

**Default setup for all C experiments:**
- Dataset: CIFAR-10 (32×32)
- Device: CUDA T4, fp16
- Batch size: 64
- Use DDIM by default — faster than DDPM, essential on free GPU time budget

### C1 — Conditioning Strategy Sweep
- **Goal:** Find the best way to inject the CLIP semantic token into the diffusion U-Net.
- **Variables:** Conditioning ∈ {concatenate to input channels, cross-attention (token as K/V), AdaIN (adaptive instance norm), none}.
- **Fixed:** U-Net with 3 encoder levels and 128 base channels, DDIM 50 steps, k=128, SNR=10 dB, epochs=80, batch=64.
- **Metrics:** CLIP, LPIPS, FID (5k).
- **Time:** ~1.5 hr per run × 4 = **6 hr total**. One Kaggle session.

### C2 — Diffusion Steps vs Quality (Sampling Efficiency)
- **Goal:** Find minimum inference steps on T4. Directly comparable to B9.
- **Dataset:** CIFAR-10 test (eval only — load best C1 checkpoint).
- **Variables:** Steps ∈ {5, 10, 25, 50, 100, 200} with DDIM and DPM-Solver++.
- **Metrics:** CLIP, LPIPS, FID (5k), ms/image on T4.
- **Time:** **45 min**.

### C3 — Bitrate Sweep (Rate–Semantic Curve, Diffusion)
- **Goal:** Core result — CLIP vs bpp for the diffusion decoder. Directly comparable to B4.
- **Dataset:** CIFAR-10.
- **Variables:** k ∈ {32, 64, 128, 256}.
- **Fixed:** Best conditioning from C1, DDIM 50 steps, SNR=10 dB, epochs=100, batch=64.
- **Metrics:** CLIP, LPIPS, PSNR vs bpp.
- **Time:** ~2 hr per k × 4 = **8 hr total**. One 12-hr Kaggle session.

### C4 — Guidance Weight Sweep (Diffusion)
- **Goal:** Comparable to B7. Find optimal guidance scale for the diffusion decoder.
- **Dataset:** CIFAR-10 test (eval only).
- **Variables:** w ∈ {1.0, 1.5, 2.0, 3.0, 5.0}. Null-token training probability=10%.
- **Fixed:** Best C3 model, k=128, SNR=10 dB.
- **Metrics:** CLIP, FID (5k), LPIPS.
- **Time:** **30 min** (eval only).

### C5 — Channel Robustness: AWGN SNR Sweep (Diffusion)
- **Goal:** Comparable to B5.
- **Dataset:** CIFAR-10 test (eval only — load best C3 checkpoint, k=128).
- **Variables:** SNR ∈ {0, 3, 5, 7, 10, 15, 20} dB.
- **Metrics:** CLIP vs SNR, SNR threshold for CLIP ≥ 0.80.
- **Time:** **30 min**.

### C6 — Channel-Aware Diffusion Training (Multi-SNR)
- **Goal:** Comparable to B6.
- **Dataset:** CIFAR-10.
- **Variables:** Regime ∈ {random SNR ∈ [0,20] dB, random SNR + SNR conditioning token}.
- **Fixed:** Best C1 architecture, k=128, epochs=100, batch=64.
- **Metrics:** CLIP vs SNR at {0, 5, 10, 20} dB.
- **Time:** ~2 hr per regime × 2 = **4 hr total**.

### C7 — Latent-Space vs Pixel-Space Diffusion
- **Goal:** Test whether a small VAE bottleneck speeds up diffusion training on T4.
- **Dataset:** CIFAR-10.
- **Variables:** Diffusion space ∈ {pixel (3×32×32), small VAE latent (4×8×8 trained jointly)}.
- **Fixed:** Cross-attention conditioning, DDIM 50 steps, k=128, SNR=10 dB, batch=64.
- **Metrics:** CLIP, FID, LPIPS, training time per epoch.
- **Time:** ~2 hr per variant × 2 = **4 hr total**.
- **Note:** Do NOT use Stable Diffusion's VAE — too large. Train a micro-VAE (3 conv layers each side, 4 latent channels) jointly with the diffusion model.

### C8 — Semantic Loss Augmentation (Diffusion)
- **Goal:** Does adding CLIP or VGG loss alongside diffusion MSE loss improve semantic fidelity?
- **Dataset:** CIFAR-10.
- **Variables:** Loss ∈ {MSE only, MSE + CLIP loss (λ=0.1), MSE + VGG perceptual (λ=0.1), MSE + CLIP + VGG}.
- **Fixed:** Best C1/C7 config, k=128, SNR=10 dB, epochs=80, batch=64.
- **Metrics:** CLIP, LPIPS, FID, PSNR.
- **Time:** ~1.5 hr per variant × 4 = **6 hr total**. One session.

### C9 — Scale-Up: TinyImageNet 64×64 (Diffusion)
- **Goal:** Comparable to B8. Validate diffusion decoder on 64×64 data.
- **Dataset:** TinyImageNet (100k images, 64×64).
- **Fixed:** Best config from C1–C8. batch=64, epochs=60, lr=1e-4, fp16.
- **Metrics:** CLIP, LPIPS, FID, PSNR vs bpp.
- **Time:** **~6–8 hr**. One Kaggle session.

### C10 — Fading Channel Robustness (Diffusion)
- **Goal:** Comparable to B10.
- **Dataset:** CIFAR-10 test (eval only — load best C6 checkpoint).
- **Variables:** Channel ∈ {AWGN 10 dB, Rayleigh fading 10 dB avg SNR, burst erasure 20%}.
- **Metrics:** CLIP, LPIPS per channel type.
- **Time:** **30 min**.

---

## Experiment Group D: Head-to-Head Comparisons

All D experiments use pre-trained checkpoints already saved from Groups B and C. No new training.

### D1 — Rate–Semantic Curve: All Models
- **Models:** Best flow (B4), best diffusion (C3), DeepJSCC (A2), JPEG (A3). SEDIC and Q-GESCO numbers overlaid from their papers.
- **Dataset:** CIFAR-10 test.
- **Output:** CLIP vs bpp, LPIPS vs bpp, PSNR vs bpp — three publication-ready figures.
- **Time:** 1 hr (aggregating saved results + plotting).

### D2 — Speed vs Quality: Flow vs Diffusion on T4
- **Goal:** Which model is more practically usable given T4 throughput?
- **Models:** Flow (B9 results), Diffusion (C2 results).
- **Metrics:** CLIP and LPIPS vs ms/image on T4.
- **Output:** Pareto plot (quality vs latency).
- **Time:** 30 min.

### D3 — Channel Robustness: All Models
- **Models:** Best channel-aware flow (B6), best channel-aware diffusion (C6), DeepJSCC (A2).
- **Metrics:** CLIP vs SNR (0–20 dB) on one shared plot.
- **Time:** 30 min.

### D4 — Reconstruction Gallery
- **Dataset:** 24 CIFAR-10 test images (4 per class × 6 classes).
- **Models:** Flow, Diffusion, DeepJSCC, JPEG at k=128 / SNR=10 dB.
- **Output:** Image grid (4 models × 24 images), saved as PDF and PNG.
- **Time:** 30 min.

### D5 — CLIP Score Distribution
- **Dataset:** Full CIFAR-10 test set (10k images).
- **Models:** Best flow, best diffusion, DeepJSCC, JPEG.
- **Output:** Overlapping histogram of per-image CLIP scores + summary stats (mean, std, P5, P95).
- **Time:** 30 min.

---

## Experiment Group E: Ablations

### E1 — Encoder Architecture
- **Variables:** Encoder ∈ {4-layer strided CNN, ResNet-18 (pretrained, fine-tuned), MobileNetV2 (lightweight pretrained)}.
- **Fixed:** Best flow decoder (B4), k=128, SNR=10 dB, epochs=50, batch=64.
- **Metrics:** CLIP, LPIPS, encoding time (ms/image on T4).
- **Time:** ~1.5 hr × 3 = **4.5 hr total**.

### E2 — Semantic Token Type
- **Variables:** Token ∈ {CLIP ViT-B/32 (512-d, frozen), learned transformer token (128-d, trained end-to-end), class one-hot (10-d for CIFAR-10), none}.
- **Fixed:** Best flow decoder, CNN encoder, k=128, SNR=10 dB, epochs=60, batch=64.
- **Metrics:** CLIP, LPIPS, FID (5k).
- **Time:** ~1.5 hr × 4 = **6 hr total**. One Kaggle session.

### E3 — Quantisation Method for Latent Code
- **Variables:** Quant ∈ {uniform 8-bit, uniform 4-bit, VQ-VAE (codebook size 512), no quantisation / analog}.
- **Fixed:** Best flow decoder, CNN encoder, SNR=10 dB, epochs=60, batch=64.
- **Metrics:** CLIP vs bpp (one rate–semantic curve per quantisation method).
- **Time:** ~2 hr × 4 = **8 hr total**. One Kaggle session.

### E4 — Data Augmentation Impact
- **Variables:** Augmentation ∈ {none, random flip+crop, flip+crop+colour jitter, flip+crop+MixUp}.
- **Fixed:** Best flow decoder, CNN encoder, k=128, SNR=10 dB, epochs=60, batch=64.
- **Metrics:** CLIP on CIFAR-10 val (in-distribution) and COCO 2k subset (out-of-distribution).
- **Time:** ~1.5 hr × 4 = **6 hr total**. One Kaggle session.

---

## Full Experiment Summary Table

| ID | Group | Short Name | Dataset | Key Variable | Primary Metric | Est. GPU Time |
|----|-------|-----------|---------|-------------|---------------|--------------|
| A1 | Baseline | Eval Calibration | CIFAR-10 | — | CLIP sanity | 15 min (CPU) |
| A2 | Baseline | DeepJSCC | CIFAR-10 | k ∈ {64,128} | Acc vs SNR | 1 hr |
| A3 | Baseline | JPEG | CIFAR-10 | quality | PSNR vs bpp | 20 min (CPU) |
| B1 | Flow | Path Compare | CIFAR-10 | linear vs OT | CLIP, loss | 3 hr |
| B2 | Flow | Depth Ablation | CIFAR-10 | n_blocks ∈ {3,5,8} | CLIP, ms/img | 3 hr |
| B3 | Flow | Conditioning | CIFAR-10 | token type | CLIP, FID | 6 hr |
| B4 | Flow | Rate–Semantic | CIFAR-10 | k ∈ {32,64,128,256} | CLIP vs bpp | 8 hr |
| B5 | Flow | SNR Sweep | CIFAR-10 test | SNR 0–20 dB | CLIP vs SNR | 30 min |
| B6 | Flow | Channel-Aware | CIFAR-10 | regime | CLIP vs SNR | 4 hr |
| B7 | Flow | Guidance Sweep | CIFAR-10 test | w ∈ {1.0–3.0} | CLIP, FID | 30 min |
| B8 | Flow | TinyImageNet | TinyImageNet 64×64 | — | CLIP, FID | 6 hr |
| B9 | Flow | ODE Steps | CIFAR-10 test | steps ∈ {5–200} | CLIP vs ms | 45 min |
| B10 | Flow | Fading Channel | CIFAR-10 test | channel type | CLIP, LPIPS | 30 min |
| C1 | Diffusion | Conditioning | CIFAR-10 | method | CLIP, FID | 6 hr |
| C2 | Diffusion | Steps vs Quality | CIFAR-10 test | steps, sampler | CLIP, ms | 45 min |
| C3 | Diffusion | Rate–Semantic | CIFAR-10 | k ∈ {32,64,128,256} | CLIP vs bpp | 8 hr |
| C4 | Diffusion | Guidance Sweep | CIFAR-10 test | w ∈ {1.0–5.0} | CLIP, FID | 30 min |
| C5 | Diffusion | SNR Sweep | CIFAR-10 test | SNR 0–20 dB | CLIP vs SNR | 30 min |
| C6 | Diffusion | Channel-Aware | CIFAR-10 | regime | CLIP vs SNR | 4 hr |
| C7 | Diffusion | Latent vs Pixel | CIFAR-10 | diffusion space | CLIP, FID | 4 hr |
| C8 | Diffusion | Semantic Loss | CIFAR-10 | loss terms | CLIP, LPIPS | 6 hr |
| C9 | Diffusion | TinyImageNet | TinyImageNet 64×64 | — | CLIP, FID | 8 hr |
| C10 | Diffusion | Fading Channel | CIFAR-10 test | channel type | CLIP, LPIPS | 30 min |
| D1 | Compare | Rate Curves | CIFAR-10 | all models | CLIP vs bpp | 1 hr |
| D2 | Compare | Speed vs Quality | CIFAR-10 | steps | CLIP vs ms | 30 min |
| D3 | Compare | SNR Robustness | CIFAR-10 | all models | CLIP vs SNR | 30 min |
| D4 | Compare | Recon Gallery | CIFAR-10 | all models | Visual | 30 min |
| D5 | Compare | CLIP Distribution | CIFAR-10 | all models | CLIP hist | 30 min |
| E1 | Ablation | Encoder Arch | CIFAR-10 | encoder type | CLIP, ms | 4.5 hr |
| E2 | Ablation | Token Type | CIFAR-10 | token | CLIP, FID | 6 hr |
| E3 | Ablation | Quantisation | CIFAR-10 | quant method | CLIP vs bpp | 8 hr |
| E4 | Ablation | Augmentation | CIFAR-10 + COCO | augment | CLIP in/out | 6 hr |

**Total estimated GPU hours: ~115 hr.** At 30 GPU hr/week on Kaggle this is ~4 weeks. Colab can supplement for shorter eval-only runs.

---

## Improvement Targets

| Experiment | Reference | Target |
|-----------|-----------|--------|
| B4 vs A2 (DeepJSCC) | Bourtsoulatze 2019 | +10% CLIP at matched bpp |
| B4 vs A3 (JPEG) | Standard | Higher CLIP at same bpp across all k values |
| B9 vs C2 | — | Flow achieves same CLIP in ≤50% diffusion decoding time on T4 |
| B6/C6 vs A2 | DeepJSCC | CLIP ≥ 0.80 at ≥5 dB lower SNR |
| C3 vs A3 (JPEG) | Standard | +3–5 dB PSNR at matched bpp |

---

## Session Planning

| Session | Platform | Experiments | Est. Duration |
|---------|---------|------------|--------------|
| 1 | Colab (CPU) | A1, A3 | 30 min |
| 2 | Kaggle GPU | A2 | 1 hr |
| 3 | Kaggle GPU | B1 (linear + OT) | 3 hr |
| 4 | Kaggle GPU | B2 (depth ablation) | 3 hr |
| 5 | Kaggle GPU | B3 (conditioning) | 6 hr |
| 6 | Kaggle GPU | B4 (rate sweep, all k) | 8 hr |
| 7 | Kaggle GPU | B5, B6 | 4.5 hr |
| 8 | Kaggle GPU | B7, B9, B10 | 2 hr |
| 9 | Kaggle GPU | B8 (TinyImageNet) | 6 hr |
| 10 | Kaggle GPU | C1 (conditioning) | 6 hr |
| 11 | Kaggle GPU | C3 (rate sweep, all k) | 8 hr |
| 12 | Kaggle GPU | C2, C4, C5, C10 | 3 hr |
| 13 | Kaggle GPU | C6, C7 | 8 hr |
| 14 | Kaggle GPU | C8 (semantic loss) | 6 hr |
| 15 | Kaggle GPU | C9 (TinyImageNet) | 8 hr |
| 16 | Kaggle GPU | D1–D5 | 3 hr |
| 17 | Kaggle GPU | E1, E2 | ~10 hr (split if needed) |
| 18 | Kaggle GPU | E3, E4 | ~14 hr (split across 2 sessions) |
