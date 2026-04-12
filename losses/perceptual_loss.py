import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """L1 of VGG-16 relu2_2 features. VGG frozen."""
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice = torch.nn.Sequential()
        # VGG features up to relu2_2 (layer 9 in .features)
        for x in range(9):
            self.slice.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False
        self.device = device
        self.slice.to(device)
        self.slice.eval()

    def forward(self, x_orig, x_recon):
        # inputs typically in [-1, 1], so transform them to [0, 1] then Imagenet Normalization
        x_orig = (x_orig + 1.0) / 2.0
        x_recon = (x_recon + 1.0) / 2.0
        
        # normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        x_orig = (x_orig - mean) / std
        x_recon = (x_recon - mean) / std

        with torch.no_grad():
            feat_orig = self.slice(x_orig)
        feat_recon = self.slice(x_recon)
        return F.l1_loss(feat_orig, feat_recon)
