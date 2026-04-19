import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        token_dim: int,
        image_size: int,
        conditioning: str = "concat",
        base_channels: int = 128,
        n_levels: int = 3,
        timesteps: int = 1000,
        use_vae_latent: bool = False,
        snr_conditioning: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.image_size = image_size
        self.conditioning = conditioning
        self.base_channels = base_channels
        self.n_levels = n_levels
        self.timesteps = timesteps
        self.use_vae_latent = use_vae_latent
        self.snr_conditioning = snr_conditioning
        self.C_in = 4 if use_vae_latent else 3

        # DDPM variables
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1.0 - beta
        self.register_buffer("alpha_bar", torch.cumprod(alpha, dim=0))

        from diffusers import UNet2DConditionModel

        cond_dim = latent_dim + token_dim + (1 if snr_conditioning else 0)

        # fallback to basic unet for simple none/concat options if needed,
        # but using diffusers ensures high quality matching the requirements
        if conditioning in ["cross_attention", "concat", "none", "adain"]:
            self.unet = UNet2DConditionModel(
                sample_size=image_size,
                in_channels=self.C_in + (cond_dim if conditioning == "concat" else 0),
                out_channels=self.C_in,
                layers_per_block=2,
                block_out_channels=(
                    (base_channels, base_channels * 2, base_channels * 2)
                    if n_levels == 3
                    else (
                        base_channels,
                        base_channels * 2,
                        base_channels * 2,
                        base_channels * 4,
                    )
                ),
                down_block_types=(
                    (
                        (
                            "CrossAttnDownBlock2D"
                            if conditioning == "cross_attention"
                            else "DownBlock2D"
                        ),
                        (
                            "CrossAttnDownBlock2D"
                            if conditioning == "cross_attention"
                            else "DownBlock2D"
                        ),
                        "DownBlock2D",
                    )
                    if n_levels == 3
                    else ("DownBlock2D",) * n_levels
                ),
                up_block_types=(
                    (
                        "UpBlock2D",
                        (
                            "CrossAttnUpBlock2D"
                            if conditioning == "cross_attention"
                            else "UpBlock2D"
                        ),
                        (
                            "CrossAttnUpBlock2D"
                            if conditioning == "cross_attention"
                            else "UpBlock2D"
                        ),
                    )
                    if n_levels == 3
                    else ("UpBlock2D",) * n_levels
                ),
                cross_attention_dim=(
                    cond_dim if conditioning == "cross_attention" else None
                ),
            )

    def _build_cond(self, latent, token, snr_emb):
        parts = []
        if latent is not None and latent.numel() > 0:
            parts.append(latent)

        if (
            self.token_dim > 0
            and token is not None
            and token.shape[-1] >= self.token_dim
        ):
            parts.append(token[..., : self.token_dim])
        elif self.token_dim > 0 and token is not None and token.numel() > 0:
            parts.append(token)

        if self.snr_conditioning and snr_emb is not None:
            parts.append(snr_emb)

        if len(parts) > 0:
            cond = torch.cat(parts, dim=-1)
        else:
            cond = torch.zeros((latent.shape[0], 1), device=latent.device)  # dummy

        if self.conditioning == "cross_attention":
            cond = cond.unsqueeze(1)  # shape [B, 1, cond_dim] for diffusers
        return cond

    def compute_loss(self, x0, latent, token, snr_emb=None):
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
        x_t = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * eps

        if self.training:
            mask = (torch.rand(B, device=x0.device) > 0.1).float().view(B, 1)
            token_in = token * mask
        else:
            token_in = token

        cond = self._build_cond(latent, token_in, snr_emb)

        if self.conditioning == "concat":
            cond_spatial = cond.view(B, -1, 1, 1).expand(
                -1, -1, x_t.shape[2], x_t.shape[3]
            )
            net_in = torch.cat([x_t, cond_spatial], dim=1)
            eps_pred = self.unet(net_in, t).sample
        elif self.conditioning == "cross_attention":
            eps_pred = self.unet(x_t, t, encoder_hidden_states=cond).sample
        else:
            eps_pred = self.unet(x_t, t).sample

        return F.mse_loss(eps_pred, eps)

    def sample(
        self, latent, token, steps=50, sampler="ddim", guidance_scale=1.0, snr_emb=None
    ):
        B = latent.shape[0]
        x = torch.randn(
            B, self.C_in, self.image_size, self.image_size, device=latent.device
        )
        cond = self._build_cond(latent, token, snr_emb)
        null_cond = self._build_cond(latent, torch.zeros_like(token), snr_emb)

        timesteps = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long)
        for i, t_cur in enumerate(timesteps):
            t_batch = t_cur.expand(B).to(x.device)

            def get_eps(in_x, t_, c_):
                if self.conditioning == "concat":
                    c_sp = c_.view(B, -1, 1, 1).expand(
                        -1, -1, in_x.shape[2], in_x.shape[3]
                    )
                    net_in = torch.cat([in_x, c_sp], dim=1)
                    return self.unet(net_in, t_).sample
                elif self.conditioning == "cross_attention":
                    return self.unet(in_x, t_, encoder_hidden_states=c_).sample
                else:
                    return self.unet(in_x, t_).sample

            with torch.no_grad():
                eps_cond = get_eps(x, t_batch, cond)
                if guidance_scale != 1.0:
                    eps_null = get_eps(x, t_batch, null_cond)
                    eps = eps_null + guidance_scale * (eps_cond - eps_null)
                else:
                    eps = eps_cond

            alpha_bar = self.alpha_bar[t_cur]
            alpha_bar_prev = (
                self.alpha_bar[timesteps[i + 1]]
                if i + 1 < steps
                else torch.tensor(1.0, device=x.device)
            )
            x0_pred = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
            x = alpha_bar_prev.sqrt() * x0_pred + (1 - alpha_bar_prev).sqrt() * eps
        return x.clamp(-1, 1)
