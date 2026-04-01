"""
Augmentation pipelines using albumentations.
Train transforms include aggressive augmentation for robustness.
Val transforms are deterministic (resize + normalize only).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Standard ImageNet statistics for normalization
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def get_train_transforms(h: int = 256, w: int = 512) -> A.Compose:
    """
    Augmentation pipeline for training:
    - Resize to target (h, w)
    - Random horizontal flip (road images are symmetric)
    - Color jitter (simulates lighting/weather variation)
    - Random crop (adds spatial variation within same image)
    - Grid distortion (simulates lens distortion)
    - Gaussian noise (sensor noise robustness)
    - Normalize + convert to tensor
    """
    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
        A.RandomCrop(h, w, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def get_val_transforms(h: int = 256, w: int = 512) -> A.Compose:
    """Deterministic transforms for validation and testing."""
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
