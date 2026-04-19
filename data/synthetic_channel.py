import torch
import torch.nn as nn


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

    def __init__(self, snr_db=None, snr_range=(0, 20), return_snr=False):
        super().__init__()
        self.snr_db = snr_db
        self.snr_range = snr_range
        self.return_snr = return_snr

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


class RayleighChannel(nn.Module):
    """Rayleigh flat-fading + AWGN. Same interface as AWGNChannel."""

    def __init__(self, snr_db=None, snr_range=(0, 20), return_snr=False):
        super().__init__()
        self.snr_db = snr_db
        self.snr_range = snr_range
        self.return_snr = return_snr

    def forward(self, latent):
        if self.snr_db is None:
            snr_db = torch.FloatTensor(1).uniform_(*self.snr_range).item()
        else:
            snr_db = self.snr_db

        signal_power = latent.pow(2).mean(dim=-1, keepdim=True)
        snr_linear = 10 ** (snr_db / 10)

        # Fast fading (rayleigh envelope)
        # Random complex number and compute magnitude
        h = torch.randn_like(latent) * 0.707 + 0.707

        noise_std = (signal_power / snr_linear).sqrt()
        noise = torch.randn_like(latent) * noise_std

        output = latent * h + noise
        if self.return_snr:
            return output, snr_db
        return output


class BurstErasureChannel(nn.Module):
    """
    Randomly zeros out `erasure_prob` fraction of latent elements, then adds AWGN.
    Args:
        erasure_prob (float): Probability each element is erased (default 0.2).
        snr_db (float): SNR for the additive noise component.
    """

    def __init__(self, erasure_prob=0.2, snr_db=10, return_snr=False):
        super().__init__()
        self.erasure_prob = erasure_prob
        self.snr_db = snr_db
        self.return_snr = return_snr

    def forward(self, latent):
        mask = (torch.rand_like(latent) > self.erasure_prob).float()
        erased_latent = latent * mask

        signal_power = latent.pow(2).mean(dim=-1, keepdim=True)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_std = (signal_power / snr_linear).sqrt()
        noise = torch.randn_like(latent) * noise_std

        if self.return_snr:
            return erased_latent + noise, self.snr_db
        return erased_latent + noise
