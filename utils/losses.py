"""
Loss functions for drivable space segmentation.
Combines Focal Loss (handles class imbalance) + Dice Loss (overlap maximization).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples, focuses on hard ones.
    Especially useful for sparse classes (curbs, barriers).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class DiceLoss(nn.Module):
    """
    Dice Loss — maximizes overlap between predicted and ground-truth masks.
    Operates on the positive (drivable) class only.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = F.softmax(preds, dim=1)[:, 1]   # probability of class 1 (drivable)
        targets = targets.float()
        inter   = (probs * targets).sum()
        return 1 - (2 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """Weighted sum of Focal Loss and Dice Loss."""
    def __init__(self, focal_w: float = 0.5, dice_w: float = 0.5):
        super().__init__()
        self.focal  = FocalLoss()
        self.dice   = DiceLoss()
        self.fw     = focal_w
        self.dw     = dice_w

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(preds, targets) + \
               self.dw * self.dice(preds, targets)
