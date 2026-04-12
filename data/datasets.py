import os
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
from .augmentations import get_transforms

class SemCommDataset(Dataset):
    """
    Returns: (image_tensor [3,H,W] in [-1,1], class_label int, clip_token [512])

    CLIP tokens are pre-computed and cached to disk once to avoid
    re-running CLIP inference at every training step.
    """
    def __init__(self, root, split, dataset_name, image_size,
                 clip_cache_path=None, transform=None):
        self.split = split
        self.transform = transform
        
        # Load transforms if passing string 'basic', 'none', 'full' from get_transforms
        if isinstance(transform, str):
            train_trans, val_trans = get_transforms(image_size)
            if split == 'train':
                self.transform = train_trans.get(transform, train_trans['basic'])
            else:
                self.transform = val_trans

        if dataset_name == 'cifar10':
            self.base_dataset = torchvision.datasets.CIFAR10(
                root=root, train=(split == 'train'), download=True)
        elif dataset_name == 'cifar100':
            self.base_dataset = torchvision.datasets.CIFAR100(
                root=root, train=(split == 'train'), download=True)
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

        self.clip_tokens = None
        if clip_cache_path and os.path.exists(clip_cache_path):
            self.clip_tokens = torch.load(clip_cache_path, map_location='cpu')

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        if self.transform is not None:
            img = self.transform(img)
            
        token = torch.zeros(512)
        if self.clip_tokens is not None:
            token = self.clip_tokens[idx]
            
        return img, label, token

    @staticmethod
    def precompute_clip_tokens(root, dataset_name, image_size,
                                cache_path, device, batch_size=256):
        """
        Run once. Saves {cache_path}.pt containing all CLIP tokens.
        Call this at the top of session_01 and reuse every session.
        """
        import clip
        import torch.nn.functional as F
        
        print(f"Precomputing CLIP tokens into {cache_path}")
        model, preprocess = clip.load('ViT-B/32', device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
            
        is_train = 'train' in cache_path
        if dataset_name == 'cifar10':
            ds = torchvision.datasets.CIFAR10(root=root, train=is_train, download=True)
        elif dataset_name == 'cifar100':
            ds = torchvision.datasets.CIFAR100(root=root, train=is_train, download=True)
            
        tokens = []
        with torch.no_grad():
            batch = []
            for i in tqdm(range(len(ds))):
                img, _ = ds[i]
                img_prep = preprocess(img).unsqueeze(0).to(device)
                batch.append(img_prep)
                
                if len(batch) == batch_size or i == len(ds) - 1:
                    batch_cat = torch.cat(batch, dim=0)
                    feat = model.encode_image(batch_cat).float().cpu()
                    feat = F.normalize(feat, dim=-1)
                    tokens.append(feat)
                    batch = []
                    
        all_tokens = torch.cat(tokens, dim=0)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(all_tokens, cache_path)
        print("Done precomputing tokens.")
