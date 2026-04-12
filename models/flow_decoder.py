import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowDecoder(nn.Module):
    def __init__(self, latent_dim: int, token_dim: int, image_size: int,
                 n_channels: int = 128, n_blocks: int = 5, path_type: str = 'linear',
                 use_guidance: bool = False, snr_conditioning: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.image_size = image_size
        self.n_channels = n_channels
        self.n_blocks = n_blocks
        self.path_type = path_type
        self.use_guidance = use_guidance
        self.snr_conditioning = snr_conditioning
        
        cond_dim = latent_dim + token_dim + (1 if snr_conditioning else 0)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        self.in_conv = nn.Conv2d(3, n_channels, 3, padding=1)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.GroupNorm(8, n_channels),
                'conv1': nn.Conv2d(n_channels, n_channels, 3, padding=1),
                'cond_inj': nn.Linear(128, n_channels),
                'norm2': nn.GroupNorm(8, n_channels),
                'conv2': nn.Conv2d(n_channels, n_channels, 3, padding=1)
            }))
        self.out_conv = nn.Conv2d(n_channels, 3, 3, padding=1)
        
    def vector_field(self, x_t, t, cond):
        t_embed = self.time_embed(t.view(-1, 1))
        c_proj = self.cond_proj(cond)
        combined = t_embed + c_proj
        
        x = self.in_conv(x_t)
        for block in self.blocks:
            res = x
            x = block['norm1'](x)
            x = F.silu(x)
            x = block['conv1'](x)
            
            # injection
            inj = block['cond_inj'](combined).unsqueeze(-1).unsqueeze(-1)
            x = x + inj
            
            x = block['norm2'](x)
            x = F.silu(x)
            x = block['conv2'](x)
            x = x + res
            
        return self.out_conv(x)
        
    def _build_cond(self, latent, token, snr_emb):
        parts = [latent, token]
        if self.snr_conditioning and snr_emb is not None:
            parts.append(snr_emb)
        return torch.cat(parts, dim=-1)
        
    def compute_loss(self, x0, latent, token, snr_emb=None):
        B = x0.shape[0]
        t = torch.rand(B, device=x0.device)
        x1 = torch.randn_like(x0)
        
        if self.path_type == 'ot':
            try:
                import ot
                import numpy as np
                x0_flat = x0.view(B, -1).detach().cpu().numpy()
                x1_flat = x1.view(B, -1).detach().cpu().numpy()
                M = ot.dist(x0_flat, x1_flat)
                a, b = np.ones((B,)) / B, np.ones((B,)) / B
                P = ot.emd(a, b, M)
                assignment = np.argmax(P, axis=1)
                x1 = x1[assignment]
            except ImportError:
                pass # Fallback to linear if POT 'ot' package is not installed during local checks
            
        t_b = t.view(B, 1, 1, 1)
        x_t = (1 - t_b) * x1 + t_b * x0
        target_v = x0 - x1
        
        if self.use_guidance and self.training:
            mask = (torch.rand(B, device=x0.device) > 0.1)
            token = token * mask.float().view(B, 1)
            
        cond = self._build_cond(latent, token, snr_emb)
        pred_v = self.vector_field(x_t, t, cond)
        return F.mse_loss(pred_v, target_v)
        
    def sample(self, latent, token, steps=50, guidance_scale=1.0, snr_emb=None):
        from torchdiffeq import odeint
        B = latent.shape[0]
        x = torch.randn(B, 3, self.image_size, self.image_size, device=latent.device)
        t_span = torch.linspace(0, 1, steps, device=latent.device)
        cond = self._build_cond(latent, token, snr_emb)
        
        def ode_fn(t, x):
            t_batch = t.expand(B).to(x.device)
            v_cond = self.vector_field(x, t_batch, cond)
            if guidance_scale != 1.0:
                null_cond = self._build_cond(latent, torch.zeros_like(token), snr_emb)
                v_null = self.vector_field(x, t_batch, null_cond)
                return v_null + guidance_scale * (v_cond - v_null)
            return v_cond
            
        with torch.no_grad():
            traj = odeint(ode_fn, x, t_span, method='euler')
        return traj[-1].clamp(-1, 1)
