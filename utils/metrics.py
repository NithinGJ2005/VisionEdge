"""
Evaluation metrics: mIoU and pixel accuracy.
"""

import torch


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> float:
    """
    Compute mean Intersection-over-Union over all classes.

    Args:
        preds:   (B, C, H, W) logits
        targets: (B, H, W) ground-truth class indices
    Returns:
        scalar mIoU value
    """
    preds    = preds.argmax(dim=1)   # (B, H, W)
    iou_list = []
    for cls in range(num_classes):
        pred_c   = (preds == cls)
        target_c = (targets == cls)
        inter    = (pred_c & target_c).sum().float()
        union    = (pred_c | target_c).sum().float()
        if union == 0:
            iou_list.append(torch.tensor(1.0))
        else:
            iou_list.append(inter / union)
    return torch.stack(iou_list).mean().item()


def compute_pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute overall pixel accuracy.

    Args:
        preds:   (B, C, H, W) logits
        targets: (B, H, W) ground-truth
    Returns:
        scalar accuracy in [0, 1]
    """
    preds   = preds.argmax(dim=1)
    correct = (preds == targets).sum().float()
    total   = targets.numel()
    return (correct / total).item()


def compute_per_class_iou(preds: torch.Tensor, targets: torch.Tensor,
                           num_classes: int = 2) -> list:
    """Return per-class IoU list."""
    preds    = preds.argmax(dim=1)
    ious     = []
    for cls in range(num_classes):
        inter = ((preds == cls) & (targets == cls)).sum().float()
        union = ((preds == cls) | (targets == cls)).sum().float()
        ious.append((inter / union).item() if union > 0 else 1.0)
    return ious
