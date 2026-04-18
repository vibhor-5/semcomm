import torch
import numpy as np
import torchvision.transforms as T

NORMALISE = T.Normalize(mean=[0.5]*3, std=[0.5]*3)   # maps [0,1] → [-1,1]


def get_transforms(size):
    TRAIN = {
        'none':   T.Compose([T.Resize((size, size)), T.ToTensor(), NORMALISE]),
        'basic':  T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size, padding=4),
                              T.ToTensor(), NORMALISE]),
        'full':   T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size, padding=4),
                              T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                              T.ToTensor(), NORMALISE]),
        # 'mixup' augmentation is applied via mixup_collate_fn at the DataLoader level
        'mixup':  T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size, padding=4),
                              T.ToTensor(), NORMALISE]),
    }

    VAL = T.Compose([T.Resize((size, size)), T.CenterCrop(size), T.ToTensor(), NORMALISE])

    return TRAIN, VAL


def mixup_collate_fn(alpha: float = 0.4):
    """
    Returns a collate function that applies MixUp to a batch.
    MixUp mixes pairs of images and their CLIP tokens with the same lambda.
    Labels are NOT mixed (we only care about image + token for semantic comm).

    Args:
        alpha: Beta distribution parameter for MixUp. Higher = more mixing.

    Usage:
        loader = DataLoader(dataset, collate_fn=mixup_collate_fn(alpha=0.4))
    """
    def collate(batch):
        images, labels, tokens = zip(*batch)
        images = torch.stack(images)
        tokens = torch.stack(tokens)
        labels = torch.tensor(labels)

        lam = float(np.random.beta(alpha, alpha))
        B = images.size(0)
        idx = torch.randperm(B)

        mixed_images = lam * images + (1 - lam) * images[idx]
        mixed_tokens = lam * tokens + (1 - lam) * tokens[idx]

        return mixed_images, labels, mixed_tokens

    return collate
