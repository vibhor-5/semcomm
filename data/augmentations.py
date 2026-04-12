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
    }
    
    VAL = T.Compose([T.Resize((size, size)), T.CenterCrop(size), T.ToTensor(), NORMALISE])
    
    return TRAIN, VAL
