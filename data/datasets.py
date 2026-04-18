import os
import torch
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from .augmentations import get_transforms


class SemCommDataset(Dataset):
    """
    Returns: (image_tensor [3,H,W] in [-1,1], class_label int, clip_token [512])

    CLIP tokens are pre-computed and cached to disk once to avoid
    re-running CLIP inference at every training step.

    Supported dataset_name values:
        'cifar10'       — CIFAR-10 32×32 (standard torchvision)
        'cifar100'      — CIFAR-100 32×32
        'tinyimagenet'  — TinyImageNet 64×64 (must be downloaded separately)
        'coco_subset'   — COCO val2017 2k-subset (pre-saved to disk)
    """

    def __init__(self, root, split, dataset_name, image_size,
                 clip_cache_path=None, transform=None):
        self.split = split
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.transform = transform

        # Accept string shortnames for transforms
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
        elif dataset_name == 'tinyimagenet':
            self.base_dataset = _TinyImageNetDataset(root, split, image_size)
        elif dataset_name == 'coco_subset':
            self.base_dataset = _COCOSubsetDataset(root, split, image_size)
        else:
            raise ValueError(f"Unknown dataset '{dataset_name}'. "
                             "Expected one of: cifar10, cifar100, tinyimagenet, coco_subset")

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
        Run once. Saves {cache_path} containing all CLIP tokens.
        Call this at the top of session_01 and reuse every session.
        """
        try:
            import clip
        except ImportError:
            raise ImportError(
                "The 'clip' module is required for semantic_token_type='clip'. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            )
        import torch.nn.functional as F

        print(f"Precomputing CLIP tokens → {cache_path}")
        model, preprocess = clip.load('ViT-B/32', device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        is_train = 'train' in os.path.basename(cache_path)
        if dataset_name == 'cifar10':
            ds = torchvision.datasets.CIFAR10(root=root, train=is_train, download=True)
        elif dataset_name == 'cifar100':
            ds = torchvision.datasets.CIFAR100(root=root, train=is_train, download=True)
        elif dataset_name == 'tinyimagenet':
            ds = _TinyImageNetDataset(root, 'train' if is_train else 'val', image_size)
        elif dataset_name == 'coco_subset':
            split_str = 'train' if is_train else 'val'
            ds = _COCOSubsetDataset(root, split_str, image_size)
        else:
            raise ValueError(f"Unknown dataset '{dataset_name}'")

        tokens = []
        with torch.no_grad():
            batch_imgs = []
            for i in tqdm(range(len(ds)), desc="CLIP token pre-computation"):
                img, _ = ds[i]
                # img is a PIL Image from the raw dataset loaders
                if not isinstance(img, Image.Image):
                    img = T.ToPILImage()(img)
                img_prep = preprocess(img).unsqueeze(0).to(device)
                batch_imgs.append(img_prep)

                if len(batch_imgs) == batch_size or i == len(ds) - 1:
                    batch_cat = torch.cat(batch_imgs, dim=0)
                    feat = model.encode_image(batch_cat).float().cpu()
                    feat = F.normalize(feat, dim=-1)
                    tokens.append(feat)
                    batch_imgs = []

        all_tokens = torch.cat(tokens, dim=0)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(all_tokens, cache_path)
        print(f"Saved {len(all_tokens)} tokens → {cache_path}")


# ---------------------------------------------------------------------------
# Helper dataset classes for non-torchvision datasets
# ---------------------------------------------------------------------------

class _TinyImageNetDataset(Dataset):
    """
    TinyImageNet loader.
    Expected directory layout (download from http://cs231n.stanford.edu/tiny-imagenet-200.zip):
        {root}/
            train/
                n01443537/
                    images/
                        n01443537_0.JPEG
                        ...
            val/
                images/
                    val_00000001.JPEG
                    ...
                val_annotations.txt
    """

    def __init__(self, root: str, split: str, image_size: int):
        assert split in ('train', 'val', 'test'), f"Unknown split '{split}'"
        self.image_size = image_size
        self.samples = []   # list of (path, label_idx)
        self.class_to_idx = {}

        if split == 'train':
            class_dirs = sorted(os.listdir(os.path.join(root, 'train')))
            self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}
            for cls in class_dirs:
                img_dir = os.path.join(root, 'train', cls, 'images')
                if not os.path.isdir(img_dir):
                    continue
                for fname in os.listdir(img_dir):
                    if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.samples.append((os.path.join(img_dir, fname),
                                             self.class_to_idx[cls]))
        else:  # val / test — use val_annotations.txt
            ann_file = os.path.join(root, 'val', 'val_annotations.txt')
            img_dir  = os.path.join(root, 'val', 'images')
            # Build class list from train directory for consistent indices
            train_dir = os.path.join(root, 'train')
            if os.path.isdir(train_dir):
                class_dirs = sorted(os.listdir(train_dir))
                self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}
            if os.path.exists(ann_file):
                with open(ann_file) as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        fname, cls = parts[0], parts[1]
                        label = self.class_to_idx.get(cls, 0)
                        self.samples.append((os.path.join(img_dir, fname), label))
            else:
                # Flat fallback — label = -1
                for fname in sorted(os.listdir(img_dir)):
                    if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.samples.append((os.path.join(img_dir, fname), -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return img, label


class _COCOSubsetDataset(Dataset):
    """
    COCO val2017 2k-subset loader.
    Expected layout (pre-sampled with seed=42):
        {root}/
            train/  (1600 PNG/JPEG images)
            val/    (200 images)
            test/   (200 images)

    Images should have been resized/saved at image_size × image_size beforehand
    OR they will be resized on-the-fly here.
    Label is always 0 (no class labels used for COCO OOD evaluation).
    """

    def __init__(self, root: str, split: str, image_size: int):
        assert split in ('train', 'val', 'test'), f"Unknown split '{split}'"
        self.image_size = image_size
        self.img_dir = os.path.join(root, split)
        self.samples = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]) if os.path.isdir(self.img_dir) else []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.samples[idx])
        img = Image.open(path).convert('RGB')
        return img, 0   # no class label for COCO OOD eval

