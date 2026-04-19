import torch
import torch.nn.functional as F
import math
import numpy as np


def compute_clip_similarity(
    orig: torch.Tensor, recon: torch.Tensor, clip_model, device
) -> float:
    """Mean cosine similarity in [0,1]. Higher is better."""
    import torchvision.transforms as T

    normalise = T.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    def encode(x):
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = (x + 1.0) / 2.0
        x = normalise(x)
        return clip_model.encode_image(x).float()

    with torch.no_grad():
        f_orig = F.normalize(encode(orig.to(device)), dim=-1)
        f_recon = F.normalize(encode(recon.to(device)), dim=-1)

    sim = (f_orig * f_recon).sum(dim=-1).mean().item()
    return float(sim)


def compute_lpips(
    orig: torch.Tensor, recon: torch.Tensor, device, lpips_model=None
) -> float:
    """Mean LPIPS (AlexNet). Lower is better."""
    if lpips_model is None:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()

    with torch.no_grad():
        loss = lpips_model(orig.to(device), recon.to(device))
    return float(loss.mean().item())


def compute_fid(
    real_images: torch.Tensor, fake_images: torch.Tensor, device, batch_size=128
) -> float:
    """FID using torchmetrics.image.FrechetInceptionDistance. Lower is better."""
    try:
        from torchmetrics.image import FrechetInceptionDistance
    except ImportError:
        from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)

    # torchmetrics FID expects images in [0, 255] uint8
    def to_uint8(x):
        return ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)

    real_images_uint8 = to_uint8(real_images)
    fake_images_uint8 = to_uint8(fake_images)

    for i in range(0, len(real_images_uint8), batch_size):
        fid.update(real_images_uint8[i : i + batch_size].to(device), real=True)
    for i in range(0, len(fake_images_uint8), batch_size):
        fid.update(fake_images_uint8[i : i + batch_size].to(device), real=False)

    return float(fid.compute().item())


def compute_psnr(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """PSNR in dB. Inputs in [-1,1]."""
    mse = F.mse_loss(recon, orig).item()
    if mse == 0:
        return 100.0
    return 10 * math.log10(4.0 / mse)


def compute_ssim(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """Mean SSIM using torchmetrics."""
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
    except ImportError:
        from torchmetrics.functional import (
            structural_similarity_index_measure as ssim_fn,
        )

        return float(ssim_fn(recon.cpu(), orig.cpu(), data_range=2.0).item())
    ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
    return float(ssim(recon.cpu(), orig.cpu()).item())


def compute_bpp(latent_dim: int, quant_bits: int, H: int, W: int) -> float:
    return latent_dim * quant_bits / (H * W)


def measure_sampling_time_gpu(
    decoder, latent: torch.Tensor, token: torch.Tensor, steps: int, n_repeats: int = 20
) -> float:
    """Mean ms per image using CUDA events. Returns ms/image."""
    if not torch.cuda.is_available():
        return 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            start.record()
            decoder.sample(latent, token, steps=steps)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / latent.shape[0])  # ms per image
    if len(times) > 2:
        return float(np.mean(times[2:]))  # discard first 2 (warmup)
    elif len(times) > 0:
        return float(np.mean(times))
    return 0.0
