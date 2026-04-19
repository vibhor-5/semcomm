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
