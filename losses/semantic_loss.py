import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class CLIPLoss(nn.Module):
    """1 - cosine_similarity(CLIP(orig), CLIP(recon)). CLIP encoder frozen."""
    def __init__(self, device, clip_model_name='ViT-B/32'):
        super().__init__()
        import clip
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        for p in self.model.parameters(): 
            p.requires_grad_(False)
        self.normalise = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))

    def forward(self, x_orig: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        # x in [-1,1], resize to 224×224 for CLIP
        def encode(x):
            x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
            x = (x + 1) / 2   # -> [0,1]
            x = self.normalise(x)
            return self.model.encode_image(x).float()
            
        with torch.no_grad():
            f_orig  = F.normalize(encode(x_orig),  dim=-1)
        f_recon = F.normalize(encode(x_recon), dim=-1)
        return 1 - (f_orig * f_recon).sum(dim=-1).mean()
