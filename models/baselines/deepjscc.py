import torch
import torch.nn as nn
from ..encoder import quantise


class DeepJSCC(nn.Module):
    """DeepJSCC Baseline (CNN autoencoder)"""

    def __init__(self, latent_dim=128, quant_bits=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.quant_bits = quant_bits

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, latent_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x, snr_db=10):
        latent = self.encoder(x)
        if self.quant_bits > 0:
            latent = quantise(latent, self.quant_bits)

        signal_power = latent.pow(2).mean(dim=-1, keepdim=True)
        snr_linear = 10 ** (snr_db / 10)
        noise_std = (signal_power / snr_linear).sqrt()
        noise = torch.randn_like(latent) * noise_std

        noisy_latent = latent + noise
        recon = self.decoder(noisy_latent)
        return recon


class DeepJSCCDecoder(nn.Module):
    """
    Standalone decoder wrapper for the DeepJSCC purely CNN-based decoder.
    Provides standard compute_loss and sample methods for the Trainer.
    """

    def __init__(self, latent_dim=128, image_size=32):
        super().__init__()
        self.image_size = image_size
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def compute_loss(self, images, noisy_latent, tokens, snr_emb):
        recon = self.decoder(noisy_latent)
        if recon.shape[-2:] != images.shape[-2:]:
            import torch.nn.functional as F

            recon = F.interpolate(recon, size=images.shape[-2:], mode="bilinear")
        return torch.nn.functional.mse_loss(recon, images)

    def sample(self, noisy_latent, tokens=None, steps=None, snr_emb=None, **kwargs):
        recon = self.decoder(noisy_latent)
        if recon.shape[-1] != self.image_size:
            import torch.nn.functional as F
            recon = F.interpolate(recon, size=(self.image_size, self.image_size), mode="bilinear")
        return recon
