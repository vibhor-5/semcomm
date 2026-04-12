# Semantic Communication: Agent Implementation & Experiment Design Guide
### (Free GPU Edition — Google Colab T4 / Kaggle T4/P100)

> **Purpose:** This document gives a coding agent everything needed to implement, run, and log all experiments from the Experiment Catalogue on free cloud GPUs. Follow it top-to-bottom. Every code decision is tuned for a T4 (15–16 GB VRAM) with fp16, checkpointing to Google Drive or Kaggle Datasets, and sessions that fit within 12 hours.

---

## 1. Repository & Notebook Structure

Organise code as **Python modules** that are imported by Colab/Kaggle notebooks. This means one notebook per experiment group, with shared library code stored in your Drive or as a Kaggle utility script.

```
semcomm/                          # Clone this repo into Colab / Kaggle working dir
├── configs/                      # YAML configs, one per experiment
│   ├── baselines/  (A1, A2, A3)
│   ├── flow/       (B1–B10)
│   ├── diffusion/  (C1–C10)
│   └── comparison/ (D1–D5, E1–E4)
├── data/
│   ├── datasets.py               # Dataset classes and loaders
│   ├── augmentations.py          # Augmentation pipelines
│   └── synthetic_channel.py     # AWGN, Rayleigh, burst noise
├── models/
│   ├── encoder.py
│   ├── channel.py
│   ├── flow_decoder.py
│   ├── diffusion_decoder.py
│   └── baselines/
│       └── deepjscc.py           # Baseline only — do not modify
├── losses/
│   ├── semantic_loss.py          # CLIP loss
│   ├── perceptual_loss.py        # VGG perceptual loss
│   └── flow_loss.py              # Flow matching objective (also inside FlowDecoder)
├── training/
│   ├── trainer.py                # Generic trainer with fp16 + checkpointing
│   ├── train_flow.py             # Entry point for B-group experiments
│   └── train_diffusion.py       # Entry point for C-group experiments
├── evaluation/
│   ├── metrics.py
│   ├── evaluate.py
│   └── visualise.py
├── utils/
│   ├── seed.py
│   ├── device.py                 # Device detection (CUDA / CPU fallback)
│   └── checkpoint.py            # Drive/Kaggle checkpoint helpers
├── tests/
│   └── *.py
├── notebooks/
│   ├── session_01_baselines.ipynb
│   ├── session_02_flow_path.ipynb
│   └── ...                      # One notebook per session from catalogue
├── requirements.txt
└── README.md
```

**Notebook header (every notebook starts with this):**
```python
# Mount Drive (Colab only)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = '/content/drive/MyDrive/semcomm'
    IN_COLAB = True
except ImportError:
    DRIVE_ROOT = '/kaggle/working/semcomm'
    IN_COLAB = False

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                'semcomm/requirements.txt'])

import sys; sys.path.insert(0, 'semcomm/')
from utils.seed import set_seed
from utils.device import get_device

DEVICE = get_device()   # 'cuda' on T4/P100, 'cpu' fallback
set_seed(42)
print(f"Device: {DEVICE}")
```

---

## 2. Environment Setup

### 2.1 requirements.txt

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
scipy>=1.10.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
wandb>=0.15.0
git+https://github.com/openai/CLIP.git
lpips>=0.1.4
torchmetrics[image]>=1.0.0
diffusers>=0.25.0
accelerate>=0.25.0
einops>=0.7.0
torchdiffeq>=0.2.3
pot>=0.9.0
compressai>=1.2.4
```

Install at the start of each notebook session (Colab/Kaggle discards pip state between sessions).

### 2.2 Device Detection (`utils/device.py`)

```python
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Apple Silicon fallback (if ever running locally)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def print_device_info():
    dev = get_device()
    print(f"Device: {dev}")
    if dev.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 2.3 Seed (`utils/seed.py`)

```python
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # set True only if input size is fixed and speed matters
```

Call `set_seed(cfg['seed'])` at the top of every training and evaluation script.

### 2.4 Checkpoint Helpers (`utils/checkpoint.py`)

```python
import os, torch

def save_checkpoint(state: dict, path: str, is_best: bool = False):
    """Save checkpoint. Optionally also copy to best.pt."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), 'best.pt')
        torch.save(state, best_path)
    print(f"Saved checkpoint → {path}")

def load_checkpoint(path: str, model, optimizer=None, scaler=None):
    """Load checkpoint. Returns epoch and best_metric."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scaler and 'scaler_state' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state'])
    return ckpt.get('epoch', 0), ckpt.get('best_metric', 0.0)

def get_checkpoint_dir(cfg: dict, drive_root: str) -> str:
    """Returns path like /content/drive/MyDrive/semcomm/outputs/B1_linear/checkpoints/"""
    exp_id = cfg['experiment_id']
    return os.path.join(drive_root, 'outputs', exp_id, 'checkpoints')
```

---

## 3. Module APIs

### 3.1 `models/encoder.py`

```python
class Encoder(nn.Module):
    """
    Maps input image to (latent, semantic_token).

    Args:
        latent_dim (int): k — dimension of latent code
        encoder_type (str): 'cnn' | 'resnet18' | 'mobilenetv2'
        semantic_token_type (str): 'clip' | 'learned' | 'class_onehot' | 'none'
        num_classes (int): required if semantic_token_type == 'class_onehot'
        quant_bits (int): uniform scalar quantisation bits (default 8)

    Input:
        image (Tensor): [B, 3, H, W] in [-1, 1]

    Output:
        latent (Tensor): [B, k] — quantised
        token  (Tensor): [B, token_dim]
    """
    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]: ...
```

**CNN encoder implementation:**
```
Conv(3→64, stride=2) → BN → ReLU
Conv(64→128, stride=2) → BN → ReLU
Conv(128→256, stride=2) → BN → ReLU
Conv(256→512, stride=2) → BN → ReLU
AdaptiveAvgPool → Flatten → Linear(512, k)
→ Uniform quantisation (straight-through estimator for gradients)
```

**CLIP token:** load `clip.load('ViT-B/32', device=device)` once, store frozen. Call inside `@torch.no_grad()`. This uses ~300 MB VRAM — fine alongside a small training model.

**Quantisation — straight-through estimator:**
```python
def quantise(x: Tensor, bits: int = 8) -> Tensor:
    scale = 2**bits - 1
    x_scaled = (x.clamp(-1, 1) + 1) / 2 * scale
    x_q = x_scaled.round()
    # Straight-through: gradients pass as if identity
    x_q = x_scaled + (x_q - x_scaled).detach()
    return x_q / scale * 2 - 1
```

**Unit tests (`tests/test_encoder.py`):**
```python
def test_shapes():
    enc = Encoder(latent_dim=128, encoder_type='cnn', semantic_token_type='clip')
    img = torch.randn(4, 3, 32, 32)
    lat, tok = enc(img)
    assert lat.shape == (4, 128)
    assert tok.shape == (4, 512)

def test_no_nan():
    enc = Encoder(latent_dim=128, encoder_type='cnn', semantic_token_type='clip')
    lat, tok = enc(torch.randn(4, 3, 32, 32))
    assert not torch.isnan(lat).any()
    assert not torch.isnan(tok).any()

def test_clip_frozen():
    enc = Encoder(latent_dim=128, encoder_type='cnn', semantic_token_type='clip')
    for p in enc.clip_model.parameters():
        assert not p.requires_grad
```

---

### 3.2 `data/synthetic_channel.py`

```python
class AWGNChannel(nn.Module):
    """
    Adds Additive White Gaussian Noise.

    Args:
        snr_db (float | None): Fixed SNR. If None, sample from snr_range per call.
        snr_range (tuple): (min_db, max_db) for random SNR mode.
        return_snr (bool): If True, return (noisy_latent, snr_used).

    Input:  latent [B, k]
    Output: noisy_latent [B, k]  (and optionally snr_used float)
    """
    def forward(self, latent): ...

class RayleighChannel(nn.Module):
    """Rayleigh flat-fading + AWGN. Same interface as AWGNChannel."""
    def forward(self, latent): ...

class BurstErasureChannel(nn.Module):
    """
    Randomly zeros out `erasure_prob` fraction of latent elements, then adds AWGN.
    Args:
        erasure_prob (float): Probability each element is erased (default 0.2).
        snr_db (float): SNR for the additive noise component.
    """
    def forward(self, latent): ...
```

**AWGN implementation:**
```python
def forward(self, latent):
    if self.snr_db is None:
        snr_db = torch.FloatTensor(1).uniform_(*self.snr_range).item()
    else:
        snr_db = self.snr_db

    signal_power = latent.pow(2).mean(dim=-1, keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_std = (signal_power / snr_linear).sqrt()
    noise = torch.randn_like(latent) * noise_std

    if self.return_snr:
        return latent + noise, snr_db
    return latent + noise
```

**Unit tests (`tests/test_channel.py`):**
```python
def test_awgn_noise_power():
    ch = AWGNChannel(snr_db=10)
    x = torch.ones(200, 128)  # unit power
    noisy = ch(x)
    noise_power = (noisy - x).pow(2).mean().item()
    assert abs(noise_power - 0.1) < 0.02   # SNR=10dB → noise power ≈ 0.1

def test_shape_preserved():
    ch = AWGNChannel(snr_db=10)
    x = torch.randn(8, 128)
    assert ch(x).shape == (8, 128)

def test_random_snr_range():
    ch = AWGNChannel(snr_range=(5, 15), return_snr=True)
    _, snr = ch(torch.randn(4, 64))
    assert 5 <= snr <= 15
```

---

### 3.3 `models/flow_decoder.py`

```python
class FlowDecoder(nn.Module):
    """
    Conditional Continuous Normalizing Flow trained via flow matching.

    Args:
        latent_dim (int): Encoder latent k (conditioning input)
        token_dim (int): Semantic token dim (512 for CLIP ViT-B/32)
        image_size (int): Output image side length (32 or 64)
        n_channels (int): Hidden channels in vector field ResNet (default 128)
        n_blocks (int): Number of ResNet blocks (default 5)
        path_type (str): 'linear' | 'ot'
        use_guidance (bool): Enable classifier-free guidance (null-token training)
        snr_conditioning (bool): Include SNR scalar as extra conditioning input

    Methods:
        compute_loss(x0, latent, token, snr_emb=None) -> Tensor
        sample(latent, token, steps=50, guidance_scale=1.0, snr_emb=None) -> Tensor
    """
```

**Vector field network architecture:**
```
Inputs:
  x_t:    [B, 3, H, W]          (noisy image at time t)
  t:      [B]                   (flow time ∈ [0,1])
  cond:   [B, cond_dim]         (concat of: projected latent + token + optional SNR)

Processing:
  time_emb  = SinusoidalEmbedding(t, dim=128)           → [B, 128]
  cond_proj = Linear(cond_dim, 128) → SiLU → Linear(128, 128)  → [B, 128]
  combined  = time_emb + cond_proj                       → [B, 128]

  x = Conv(3 → n_channels, 3×3, padding=1)(x_t)
  for each ResNet block:
      residual = x
      x = GroupNorm(8, n_channels)(x)
      x = SiLU(x)
      x = Conv(n_channels, n_channels, 3×3, padding=1)(x)
      # Inject conditioning via additive bias (broadcast spatially)
      x = x + Linear(128, n_channels)(combined).unsqueeze(-1).unsqueeze(-1)
      x = GroupNorm(8, n_channels)(x)
      x = SiLU(x)
      x = Conv(n_channels, n_channels, 3×3, padding=1)(x)
      x = x + residual
  output = Conv(n_channels → 3, 3×3, padding=1)(x)      → [B, 3, H, W] velocity
```

**Flow matching loss (linear path):**
```python
def compute_loss(self, x0, latent, token, snr_emb=None):
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device)                   # t ~ U[0,1]
    x1 = torch.randn_like(x0)                             # noise
    t_b = t.view(B, 1, 1, 1)
    x_t = (1 - t_b) * x1 + t_b * x0                      # linear interpolation
    target_v = x0 - x1                                    # target velocity

    # Classifier-free guidance: randomly drop conditioning
    if self.use_guidance and self.training:
        mask = (torch.rand(B, device=x0.device) > 0.1)   # 10% null token
        token = token * mask.float().view(B, 1)

    cond = self._build_cond(latent, token, snr_emb)       # [B, cond_dim]
    pred_v = self.vector_field(x_t, t, cond)
    return F.mse_loss(pred_v, target_v)
```

**OT path:** before computing x_t, pair (x0, x1) using the Earth Mover distance via `pot.emd2` on mini-batches of size 64. Replace `x1` with the OT-matched noise sample. Target velocity remains `x0 - x1`.

**Sampling:**
```python
def sample(self, latent, token, steps=50, guidance_scale=1.0, snr_emb=None):
    from torchdiffeq import odeint
    B = latent.shape[0]
    x = torch.randn(B, 3, self.image_size, self.image_size, device=latent.device)
    t_span = torch.linspace(0, 1, steps, device=latent.device)
    cond = self._build_cond(latent, token, snr_emb)

    def ode_fn(t, x):
        t_batch = t.expand(B)
        v_cond = self.vector_field(x, t_batch, cond)
        if guidance_scale != 1.0:
            null_cond = self._build_cond(latent, torch.zeros_like(token), snr_emb)
            v_null = self.vector_field(x, t_batch, null_cond)
            return v_null + guidance_scale * (v_cond - v_null)
        return v_cond

    with torch.no_grad():
        traj = odeint(ode_fn, x, t_span, method='euler')  # 'euler' fastest; 'dopri5' more accurate
    return traj[-1].clamp(-1, 1)
```

**Unit tests (`tests/test_flow_decoder.py`):**
```python
def test_loss_finite():
    model = FlowDecoder(latent_dim=128, token_dim=512, image_size=32, n_blocks=3)
    loss = model.compute_loss(torch.randn(2,3,32,32), torch.randn(2,128), torch.randn(2,512))
    assert loss.item() > 0 and not torch.isnan(loss)

def test_sample_shape():
    model = FlowDecoder(latent_dim=128, token_dim=512, image_size=32, n_blocks=3)
    out = model.sample(torch.randn(2,128), torch.randn(2,512), steps=5)
    assert out.shape == (2, 3, 32, 32)
    assert out.min() >= -1.1 and out.max() <= 1.1

def test_snr_conditioning():
    model = FlowDecoder(latent_dim=128, token_dim=512, image_size=32,
                        n_blocks=3, snr_conditioning=True)
    snr_emb = torch.tensor([[10.0], [5.0]])
    out = model.sample(torch.randn(2,128), torch.randn(2,512), steps=5, snr_emb=snr_emb)
    assert out.shape == (2, 3, 32, 32)
```

---

### 3.4 `models/diffusion_decoder.py`

```python
class DiffusionDecoder(nn.Module):
    """
    Conditional diffusion model decoder (DDPM training, DDIM/DPM-Solver++ sampling).

    Args:
        latent_dim (int): Encoder latent k
        token_dim (int): Semantic token dim
        image_size (int): Output image side length
        conditioning (str): 'concat' | 'cross_attention' | 'adain' | 'none'
        base_channels (int): U-Net base channels (default 128)
        n_levels (int): U-Net encoder/decoder levels (default 3)
        timesteps (int): Diffusion training steps (default 1000)
        use_vae_latent (bool): Operate in micro-VAE latent space
        snr_conditioning (bool): Include SNR embedding

    Methods:
        compute_loss(x0, latent, token, snr_emb=None) -> Tensor
        sample(latent, token, steps=50, sampler='ddim',
               guidance_scale=1.0, snr_emb=None) -> Tensor
    """
```

**U-Net architecture:**
```
Input: [B, C_in, H, W] where C_in=3 (pixel) or 4 (VAE latent)

Encoder levels (repeat n_levels times, halving spatial dims):
    ResBlock(channels, channels*2) → channels *= 2
    ResBlock(channels, channels)
    Downsample (Conv stride=2)
    [At lowest level: + CrossAttention on token if conditioning='cross_attention']

Bottleneck:
    ResBlock × 2
    CrossAttention or AdaIN injection of (time_emb + cond)

Decoder levels (mirror encoder with skip connections):
    Upsample (bilinear + Conv)
    ResBlock(channels*2 + skip, channels) → channels //= 2
    ResBlock(channels, channels)

Output: Conv(base_channels → C_in)

Time embedding: sinusoidal(t, dim=256) → Linear(256,512) → SiLU → Linear(512,512)
    Injected into every ResBlock via AdaGroupNorm:
        scale, shift = Linear(512, 2*channels)(time_emb).chunk(2, dim=1)
        x = GroupNorm(x) * (1 + scale) + shift
```

**Diffusion loss:**
```python
def compute_loss(self, x0, latent, token, snr_emb=None):
    B = x0.shape[0]
    t = torch.randint(0, self.timesteps, (B,), device=x0.device)
    eps = torch.randn_like(x0)
    alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
    x_t = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * eps

    # 10% null conditioning for classifier-free guidance
    if self.training:
        mask = (torch.rand(B, device=x0.device) > 0.1).float().view(B, 1)
        token_in = token * mask
    else:
        token_in = token

    cond = self._build_cond(latent, token_in, snr_emb)
    eps_pred = self.unet(x_t, t, cond)
    return F.mse_loss(eps_pred, eps)
```

**DDIM sampling:**
```python
def sample(self, latent, token, steps=50, sampler='ddim',
           guidance_scale=1.0, snr_emb=None):
    B = latent.shape[0]
    x = torch.randn(B, self.C_in, self.image_size, self.image_size,
                    device=latent.device)
    cond      = self._build_cond(latent, token, snr_emb)
    null_cond = self._build_cond(latent, torch.zeros_like(token), snr_emb)

    timesteps = torch.linspace(self.timesteps-1, 0, steps, dtype=torch.long)
    for i, t_cur in enumerate(timesteps):
        t_batch = t_cur.expand(B).to(x.device)
        with torch.no_grad():
            eps_cond = self.unet(x, t_batch, cond)
            if guidance_scale != 1.0:
                eps_null = self.unet(x, t_batch, null_cond)
                eps = eps_null + guidance_scale * (eps_cond - eps_null)
            else:
                eps = eps_cond
        # DDIM step
        alpha_bar     = self.alpha_bar[t_cur]
        alpha_bar_prev = self.alpha_bar[timesteps[i+1]] if i+1 < steps else torch.tensor(1.0)
        x0_pred = (x - (1-alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
        x = alpha_bar_prev.sqrt() * x0_pred + (1-alpha_bar_prev).sqrt() * eps
    return x.clamp(-1, 1)
```

**Unit tests (`tests/test_diffusion_decoder.py`):**
```python
def test_loss_finite():
    model = DiffusionDecoder(latent_dim=128, token_dim=512, image_size=32,
                              conditioning='cross_attention', base_channels=64)
    loss = model.compute_loss(torch.randn(2,3,32,32),
                               torch.randn(2,128), torch.randn(2,512))
    assert loss.item() > 0 and not torch.isnan(loss)

def test_sample_shape():
    model = DiffusionDecoder(latent_dim=128, token_dim=512, image_size=32,
                              base_channels=64)
    out = model.sample(torch.randn(2,128), torch.randn(2,512), steps=5)
    assert out.shape == (2, 3, 32, 32)
```

---

### 3.5 `losses/`

```python
# losses/semantic_loss.py
class CLIPLoss(nn.Module):
    """1 - cosine_similarity(CLIP(orig), CLIP(recon)). CLIP encoder frozen."""
    def __init__(self, device, clip_model_name='ViT-B/32'):
        super().__init__()
        import clip
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        for p in self.model.parameters(): p.requires_grad_(False)

    def forward(self, x_orig: Tensor, x_recon: Tensor) -> Tensor:
        # x in [-1,1], resize to 224×224 for CLIP
        def encode(x):
            x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
            x = (x + 1) / 2   # → [0,1]
            x = self.normalise(x)
            return self.model.encode_image(x).float()
        with torch.no_grad():
            f_orig  = F.normalize(encode(x_orig),  dim=-1)
        f_recon = F.normalize(encode(x_recon), dim=-1)
        return 1 - (f_orig * f_recon).sum(dim=-1).mean()

# losses/perceptual_loss.py
class VGGPerceptualLoss(nn.Module):
    """L1 of VGG-16 relu2_2 features. VGG frozen."""
    def forward(self, x_orig, x_recon) -> Tensor: ...
```

**Unit tests (`tests/test_losses.py`):**
```python
def test_clip_loss_identical():
    device = torch.device('cpu')
    loss_fn = CLIPLoss(device)
    x = torch.randn(2, 3, 32, 32)
    assert loss_fn(x, x).item() < 0.01   # identical → ~0

def test_clip_loss_range():
    device = torch.device('cpu')
    loss_fn = CLIPLoss(device)
    x, y = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    loss = loss_fn(x, y).item()
    assert 0.0 <= loss <= 2.0
```

---

### 3.6 `evaluation/metrics.py`

```python
def compute_clip_similarity(orig: Tensor, recon: Tensor, clip_model, device) -> float:
    """Mean cosine similarity ∈ [0,1]. Higher is better."""
    ...

def compute_lpips(orig: Tensor, recon: Tensor, device) -> float:
    """Mean LPIPS (AlexNet). Lower is better. Initialise lpips.LPIPS once, not per call."""
    ...

def compute_fid(real_images: Tensor, fake_images: Tensor, device, batch_size=128) -> float:
    """FID using torchmetrics.image.FrechetInceptionDistance. Lower is better.
    Needs at least 2048 images for reliable FID — pass full test set as real."""
    ...

def compute_psnr(orig: Tensor, recon: Tensor) -> float:
    """PSNR in dB. Inputs in [-1,1]."""
    mse = F.mse_loss(recon, orig).item()
    return 10 * math.log10(4.0 / mse)   # dynamic range = 2 (from -1 to 1)

def compute_ssim(orig: Tensor, recon: Tensor) -> float:
    """Mean SSIM using torchmetrics."""
    ...

def compute_bpp(latent_dim: int, quant_bits: int, H: int, W: int) -> float:
    return latent_dim * quant_bits / (H * W)

def measure_sampling_time_gpu(decoder, latent: Tensor, token: Tensor,
                               steps: int, n_repeats: int = 20) -> float:
    """Mean ms per image using CUDA events. Returns ms/image."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            start.record()
            decoder.sample(latent, token, steps=steps)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / latent.shape[0])  # ms per image
    return float(np.mean(times[2:]))   # discard first 2 (warmup)
```

**Unit tests (`tests/test_metrics.py`):**
```python
def test_psnr_identical():
    x = torch.rand(4, 3, 32, 32) * 2 - 1
    assert compute_psnr(x, x) > 80   # identical → very high

def test_bpp():
    assert compute_bpp(128, 8, 32, 32) == pytest.approx(1.0)
    assert compute_bpp(64, 8, 32, 32)  == pytest.approx(0.5)

def test_clip_sim_range():
    # mock test without full CLIP model
    ...
```

---

## 4. Training Pipeline

### 4.1 `training/trainer.py`

```python
class Trainer:
    """
    Handles: fp16 training loop, validation, checkpointing every 5 epochs,
    W&B logging, LR scheduling.

    Args:
        encoder  (Encoder)
        channel  (AWGNChannel | RayleighChannel)
        decoder  (FlowDecoder | DiffusionDecoder)
        loss_cfg (dict): weights for flow/diffusion, clip, vgg losses
        cfg      (dict): full experiment config
        drive_root (str): path to Google Drive or Kaggle output root
    """

    def __init__(self, encoder, channel, decoder, loss_cfg, cfg, drive_root):
        self.device = get_device()
        self.encoder = encoder.to(self.device)
        self.channel = channel.to(self.device)
        self.decoder = decoder.to(self.device)
        self.scaler  = torch.cuda.amp.GradScaler()   # fp16 scaler

        params = list(encoder.parameters()) + list(decoder.parameters())
        # Note: channel has no learnable params; CLIP encoder is frozen
        self.optimizer = torch.optim.AdamW(params, lr=cfg['training']['lr'],
                                            weight_decay=cfg['training']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg['training']['n_epochs'])

        self.ckpt_dir = get_checkpoint_dir(cfg, drive_root)
        self.cfg = cfg
        self.loss_cfg = loss_cfg
        self.best_clip = 0.0

    def _train_step(self, batch) -> dict:
        images, labels, _ = batch
        images = images.to(self.device)

        with torch.cuda.amp.autocast():
            latent, token = self.encoder(images)
            if self.cfg['channel'].get('snr_conditioning'):
                noisy_latent, snr_used = self.channel(latent)
                snr_emb = torch.full((images.shape[0], 1), snr_used, device=self.device)
            else:
                noisy_latent = self.channel(latent)
                snr_emb = None

            main_loss = self.decoder.compute_loss(images, noisy_latent, token, snr_emb)

            clip_loss = torch.tensor(0.0, device=self.device)
            if self.loss_cfg.get('clip_weight', 0) > 0:
                with torch.no_grad():
                    recon = self.decoder.sample(noisy_latent, token, steps=10, snr_emb=snr_emb)
                clip_loss = self.clip_loss_fn(images, recon) * self.loss_cfg['clip_weight']

            total = main_loss + clip_loss

        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            self.cfg['training']['gradient_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return {'main': main_loss.item(), 'clip': clip_loss.item(), 'total': total.item()}

    def run(self, train_loader, val_loader, n_epochs):
        import wandb
        if self.cfg['logging']['use_wandb']:
            wandb.init(project=self.cfg['logging']['project'],
                       name=self.cfg['experiment_id'],
                       config=self.cfg)

        for epoch in range(1, n_epochs + 1):
            # --- Train ---
            self.encoder.train(); self.decoder.train()
            train_losses = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}"):
                train_losses.append(self._train_step(batch))

            mean_train = {k: np.mean([d[k] for d in train_losses])
                          for k in train_losses[0]}

            # --- Validate every eval_every_n_epochs ---
            val_metrics = {}
            if epoch % self.cfg['evaluation']['eval_every_n_epochs'] == 0:
                val_metrics = self._validate(val_loader)
                if val_metrics.get('clip', 0) > self.best_clip:
                    self.best_clip = val_metrics['clip']
                    save_checkpoint({'epoch': epoch, 'model_state': ...},
                                     os.path.join(self.ckpt_dir, f'epoch_{epoch}.pt'),
                                     is_best=True)

            # --- Save every 5 epochs regardless ---
            if epoch % 5 == 0:
                save_checkpoint({'epoch': epoch, ...},
                                 os.path.join(self.ckpt_dir, f'epoch_{epoch}.pt'))

            # --- Log ---
            if self.cfg['logging']['use_wandb']:
                wandb.log({'epoch': epoch, **{f'train/{k}': v for k,v in mean_train.items()},
                            **{f'val/{k}': v for k,v in val_metrics.items()},
                            'lr': self.scheduler.get_last_lr()[0]})

            self.scheduler.step()
```

**Unit test (`tests/test_trainer.py`):**
```python
def test_one_step():
    """Smoke test — one training step must not raise."""
    cfg = load_yaml('configs/flow/B1_linear.yaml')
    cfg['training']['n_epochs'] = 1
    enc = Encoder(**cfg['encoder'])
    ch  = AWGNChannel(**cfg['channel'])
    dec = FlowDecoder(**cfg['decoder'])
    trainer = Trainer(enc, ch, dec, cfg['loss'], cfg, drive_root='/tmp')
    loader = DataLoader(DummyDataset(n=8, image_size=32), batch_size=4)
    trainer._train_step(next(iter(loader)))   # must not raise
```

---

## 5. Data Pipeline (`data/datasets.py`)

```python
class SemCommDataset(Dataset):
    """
    Returns: (image_tensor [3,H,W] in [-1,1], class_label int, clip_token [512])

    CLIP tokens are pre-computed and cached to disk once to avoid
    re-running CLIP inference at every training step.
    """
    def __init__(self, root, split, dataset_name, image_size,
                 clip_cache_path=None, transform=None): ...

    @staticmethod
    def precompute_clip_tokens(root, dataset_name, image_size,
                                cache_path, device, batch_size=256):
        """
        Run once. Saves {cache_path}.pt containing all CLIP tokens.
        Call this at the top of session_01 and reuse every session.
        """
        ...
```

**Standard transforms (`data/augmentations.py`):**
```python
NORMALISE = T.Normalize(mean=[0.5]*3, std=[0.5]*3)   # maps [0,1] → [-1,1]

TRAIN = {
    'none':   T.Compose([T.Resize(size), T.ToTensor(), NORMALISE]),
    'basic':  T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size, padding=4),
                          T.ToTensor(), NORMALISE]),
    'full':   T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size, padding=4),
                          T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                          T.ToTensor(), NORMALISE]),
    'mixup':  # Applied in collate_fn, not here
}

VAL = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), NORMALISE])
```

**Colab/Kaggle data loading pattern:**
```python
# In notebook:
CIFAR_ROOT = f'{DRIVE_ROOT}/data/cifar10'   # or /kaggle/input/cifar10
CLIP_CACHE = f'{DRIVE_ROOT}/data/cifar10_clip_tokens_train.pt'

if not os.path.exists(CLIP_CACHE):
    SemCommDataset.precompute_clip_tokens(CIFAR_ROOT, 'cifar10', 32,
                                           CLIP_CACHE, DEVICE)

train_ds = SemCommDataset(CIFAR_ROOT, 'train', 'cifar10', 32,
                           clip_cache_path=CLIP_CACHE, transform=TRAIN['basic'])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                           num_workers=2, pin_memory=True, persistent_workers=True)
```

**Unit tests (`tests/test_data.py`):**
```python
def test_cifar_shapes():
    ds = SemCommDataset('./data', 'train', 'cifar10', 32)
    img, lbl, tok = ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(lbl, int)
    assert tok.shape == (512,)

def test_image_range():
    ds = SemCommDataset('./data', 'train', 'cifar10', 32)
    img, _, _ = ds[0]
    assert img.min() >= -1.1 and img.max() <= 1.1
```

---

## 6. Experiment Config Schema (YAML)

Every experiment has a YAML config. The agent must create configs for all 34 experiments.

```yaml
# configs/flow/B1_linear.yaml
experiment_id: B1_linear
name: "Flow Path Compare — Linear"
seed: 42

dataset:
  name: cifar10
  root: /content/drive/MyDrive/semcomm/data/cifar10
  image_size: 32
  augmentation: basic
  clip_cache_path: /content/drive/MyDrive/semcomm/data/cifar10_clip_train.pt

encoder:
  latent_dim: 128
  encoder_type: cnn
  semantic_token_type: clip
  quant_bits: 8

channel:
  type: awgn
  snr_db: 10
  snr_conditioning: false

decoder:
  type: flow
  n_channels: 128
  n_blocks: 5
  path_type: linear     # change to 'ot' for B1_ot.yaml
  use_guidance: false
  snr_conditioning: false

loss:
  flow_weight: 1.0
  clip_weight: 0.0      # set >0 only for C8 semantic loss experiments
  vgg_weight: 0.0

training:
  n_epochs: 100
  batch_size: 64
  lr: 1.0e-4
  weight_decay: 1.0e-5
  gradient_clip: 1.0
  scheduler: cosine
  mixed_precision: true
  num_workers: 2

evaluation:
  metrics: [clip, lpips, fid, psnr, ssim]
  eval_every_n_epochs: 10
  sampling_steps: 50
  guidance_scale: 1.0
  n_fid_samples: 5000

logging:
  use_wandb: true
  project: semcomm_freegpu
  save_every_n_epochs: 5
  checkpoint_dir: /content/drive/MyDrive/semcomm/outputs/B1_linear/checkpoints
```

**Config generation script (`scripts/generate_configs.py`):**
```python
import copy, yaml, os

def write_config(cfg: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

base = yaml.safe_load(open('configs/flow/B1_linear.yaml'))

# B1: two path types
for path_type in ['linear', 'ot']:
    cfg = copy.deepcopy(base)
    cfg['experiment_id'] = f'B1_{path_type}'
    cfg['decoder']['path_type'] = path_type
    cfg['logging']['checkpoint_dir'] = cfg['logging']['checkpoint_dir'].replace('B1_linear', f'B1_{path_type}')
    write_config(cfg, f'configs/flow/B1_{path_type}.yaml')

# B2: n_blocks sweep
for n in [3, 5, 8]:
    cfg = copy.deepcopy(base)
    cfg['experiment_id'] = f'B2_nblocks{n}'
    cfg['decoder']['n_blocks'] = n
    cfg['training']['n_epochs'] = 60
    write_config(cfg, f'configs/flow/B2_nblocks{n}.yaml')

# ... repeat for all experiments
```

---

## 7. Evaluation Pipeline

### `evaluation/evaluate.py`

```python
"""
Usage in notebook:
    from evaluation.evaluate import run_evaluation
    metrics = run_evaluation(cfg, checkpoint_path, test_loader, device)
"""

def run_evaluation(cfg, checkpoint_path, test_loader, device) -> dict:
    encoder, channel, decoder = build_from_cfg(cfg)
    load_checkpoint(checkpoint_path, decoder)  # encoder weights also in ckpt
    encoder.eval(); decoder.eval()

    all_orig, all_recon = [], []
    with torch.no_grad():
        for images, _, tokens in tqdm(test_loader, desc='Evaluating'):
            images, tokens = images.to(device), tokens.to(device)
            latent, token = encoder(images)
            noisy = channel(latent)
            recon = decoder.sample(noisy, token,
                                    steps=cfg['evaluation']['sampling_steps'],
                                    guidance_scale=cfg['evaluation']['guidance_scale'])
            all_orig.append(images.cpu())
            all_recon.append(recon.cpu())

    orig  = torch.cat(all_orig)
    recon = torch.cat(all_recon)

    metrics = {
        'clip':  compute_clip_similarity(orig, recon, clip_model, device),
        'lpips': compute_lpips(orig, recon, device),
        'fid':   compute_fid(orig, recon[:cfg['evaluation']['n_fid_samples']], device),
        'psnr':  compute_psnr(orig, recon),
        'ssim':  compute_ssim(orig, recon),
        'bpp':   compute_bpp(cfg['encoder']['latent_dim'],
                              cfg['encoder']['quant_bits'],
                              cfg['dataset']['image_size'],
                              cfg['dataset']['image_size']),
    }

    save_path = os.path.join(DRIVE_ROOT, 'outputs', cfg['experiment_id'], 'metrics.json')
    with open(save_path, 'w') as f:
        json.dump({**metrics, 'experiment_id': cfg['experiment_id']}, f, indent=2)

    return metrics
```

### `evaluation/visualise.py`

Functions the agent must implement:

```python
def plot_rate_semantic_curve(results: list[dict], baseline_results: list[dict],
                              save_path: str):
    """CLIP vs bpp. One line per model. Baselines (JPEG, DeepJSCC) overlaid as dashed."""

def plot_snr_curve(results: list[dict], save_path: str):
    """CLIP vs SNR in dB. One line per model. Mark threshold where CLIP drops below 0.80."""

def plot_speed_quality(results: list[dict], save_path: str):
    """Scatter/line: CLIP vs ms/image on T4. Show Pareto frontier."""

def plot_reconstruction_gallery(orig: Tensor, recons: dict[str, Tensor],
                                 save_path: str, n_images: int = 8):
    """Grid: rows = images, cols = [Original, Flow, Diffusion, DeepJSCC, JPEG]."""

def plot_clip_histogram(clip_scores: dict[str, list[float]], save_path: str):
    """Overlapping histogram of per-image CLIP scores across models."""
```

All plots: save as PDF (for papers) and PNG (for notebooks). Use `matplotlib` with `seaborn` style. Always include axis labels with units, legend, and title.

---

## 8. Step-by-Step Procedure for Each Experiment

The agent should follow this exact sequence for every experiment:

```
STEP 1  Select config:    configs/{group}/{experiment_id}.yaml
STEP 2  Run unit tests:   pytest tests/test_encoder.py tests/test_channel.py
                                  tests/test_flow_decoder.py -v -x
STEP 3  Precompute CLIP cache if not already on Drive:
            SemCommDataset.precompute_clip_tokens(...)
STEP 4  Set seed:         set_seed(cfg['seed'])
STEP 5  Build models:     encoder, channel, decoder = build_from_cfg(cfg)
STEP 6  Train:            trainer.run(train_loader, val_loader, n_epochs)
            - Checkpoints saved every 5 epochs to Drive automatically
            - Monitor W&B for loss divergence or NaN (stop early if needed)
STEP 7  Evaluate:         metrics = run_evaluation(cfg, 'best.pt', test_loader, device)
STEP 8  Save outputs:     metrics.json → Drive/outputs/{id}/
                          plots        → Drive/outputs/{id}/figures/
STEP 9  Record results:   append row to Drive/outputs/summary_table.csv
STEP 10 Commit config:    push configs/{group}/{id}.yaml to Git
```

**If the session disconnects during STEP 6:** restart, load the latest epoch checkpoint (found in Drive), set `start_epoch = loaded_epoch + 1`, and resume training.

---

## 9. CI / Automated Testing

Since there is no self-hosted runner, use **GitHub Actions on CPU** for unit tests only (no GPU experiments in CI).

### `.github/workflows/ci.yml`

```yaml
name: Unit Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install CPU-only dependencies
        run: |
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
          pip install numpy scipy Pillow matplotlib tqdm pyyaml lpips einops torchdiffeq pot
          # Skip wandb, diffusers, CLIP (slow to install) — mock in tests

      - name: Lint
        run: |
          pip install flake8 black
          black --check semcomm/
          flake8 semcomm/ --max-line-length 100 --ignore E203,W503

      - name: Unit tests (CPU, no CLIP)
        run: pytest tests/ -v --tb=short -x \
               --ignore=tests/test_losses.py \  # CLIP needed — skip in CI
               -k "not clip and not fid"         # skip GPU-heavy metric tests

      - name: Smoke test (tiny model, CPU)
        run: |
          python -c "
          from models.encoder import Encoder
          from data.synthetic_channel import AWGNChannel
          from models.flow_decoder import FlowDecoder
          import torch
          enc = Encoder(latent_dim=32, encoder_type='cnn', semantic_token_type='none')
          ch  = AWGNChannel(snr_db=10)
          dec = FlowDecoder(latent_dim=32, token_dim=0, image_size=8, n_blocks=2, n_channels=16)
          x   = torch.randn(2, 3, 8, 8)
          lat, tok = enc(x)
          noisy = ch(lat)
          loss = dec.compute_loss(x, noisy, tok)
          print('Smoke test passed. Loss:', loss.item())
          "
```

---

## 10. Logging Reference

Every experiment logs the following quantities to W&B. The agent must ensure all are present.

| Quantity | When | Type |
|----------|------|------|
| `train/main_loss` | Every step | scalar |
| `train/clip_loss` | Every step (0 if disabled) | scalar |
| `train/total_loss` | Every step | scalar |
| `train/grad_norm` | Every step | scalar |
| `lr` | Every step | scalar |
| `val/clip` | Every eval epoch | scalar |
| `val/lpips` | Every eval epoch | scalar |
| `val/fid` | Every eval epoch (when n_fid_samples hit) | scalar |
| `val/psnr` | Every eval epoch | scalar |
| `val/ssim` | Every eval epoch | scalar |
| `val/sampling_time_ms` | Every eval epoch | scalar |
| `examples` | Every 10 epochs | image grid (8 images: orig + recon) |
| `bpp` | Once at start | scalar |

---

## 11. Reproducibility Checklist

Before marking any experiment complete, verify:

- [ ] Seed was set with `set_seed(42)` before any random operation.
- [ ] `metrics.json` is saved to `Drive/outputs/{experiment_id}/`.
- [ ] `best.pt` checkpoint is saved and loadable (verify by loading in a fresh cell).
- [ ] W&B run is accessible with all quantities logged.
- [ ] All figures are saved as both PDF and PNG in `Drive/outputs/{experiment_id}/figures/`.
- [ ] YAML config is committed to the Git repo.
- [ ] Row appended to `Drive/outputs/summary_table.csv` with columns: `experiment_id, clip, lpips, fid, psnr, bpp, sampling_time_ms, n_epochs_trained, gpu, date`.
- [ ] Any anomalies (loss spikes, NaN, OOM) documented in `Drive/outputs/{experiment_id}/notes.txt`.

**Handling OOM on T4:** If a model causes OOM, try in order: (1) reduce batch size to 32, then 16; (2) reduce n_channels from 128 to 64; (3) reduce image_size to 32 if using 64. Document which setting was used.
