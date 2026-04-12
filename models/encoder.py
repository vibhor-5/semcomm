import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models

def quantise(x: Tensor, bits: int = 8) -> Tensor:
    scale = 2**bits - 1
    x_scaled = (x.clamp(-1, 1) + 1) / 2 * scale
    x_q = x_scaled.round()
    # Straight-through: gradients pass as if identity
    x_q = x_scaled + (x_q - x_scaled).detach()
    return x_q / scale * 2 - 1
    
class Encoder(nn.Module):
    """
    Maps input image to (latent, semantic_token).
    """
    def __init__(self, latent_dim: int, encoder_type: str = 'cnn', 
                 semantic_token_type: str = 'clip', num_classes: int = 10,
                 quant_bits: int = 8, device=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.semantic_token_type = semantic_token_type
        self.quant_bits = quant_bits
        
        if encoder_type == 'cnn':
            self.backbone = nn.Sequential(
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
                nn.Linear(512, latent_dim)
            )
        elif encoder_type == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            resnet.fc = nn.Linear(resnet.fc.in_features, latent_dim)
            self.backbone = resnet
        elif encoder_type == 'mobilenetv2':
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, latent_dim)
            self.backbone = mobilenet
            
        if semantic_token_type == 'clip':
            import clip
            self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=device if device else 'cpu')
            for p in self.clip_model.parameters(): 
                p.requires_grad = False
                
    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.backbone(image)
        latent = torch.tanh(latent) # put roughly in [-1, 1] before quantisation
        
        if self.quant_bits > 0:
            latent = quantise(latent, self.quant_bits)
            
        # semantic token fallback logic if not passed precomputed
        token = torch.zeros(image.size(0), 512, device=image.device)
        if self.semantic_token_type == 'clip' and getattr(self, 'clip_model', None) is not None:
            with torch.no_grad():
                import torch.nn.functional as F
                x = F.interpolate(image, size=224, mode='bilinear')
                x = (x + 1)/2
                import torchvision.transforms as T
                normalise = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                        std=(0.26862954, 0.26130258, 0.27577711))
                x = normalise(x)
                token = self.clip_model.encode_image(x).float()
                token = F.normalize(token, dim=-1)
            
        return latent, token
