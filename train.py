"""
Training script for MobileNetV2 + U-Net drivable space segmentation.
Trains from scratch on synthetic (or nuScenes) data.

Usage:
    # With synthetic data:
    python train.py --dataroot ./data/synthetic --epochs 5 --batch_size 4

    # With nuScenes processed data:
    python train.py --dataroot ./data/processed --epochs 50 --batch_size 16
"""

import argparse
import os
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

from models.unet import UNet
from utils.losses import CombinedLoss
from utils.metrics import compute_miou, compute_pixel_accuracy
from utils.augmentations import get_train_transforms, get_val_transforms


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    """
    Segmentation dataset that reads from a JSON metadata file.

    Expected JSON format:
        [{"image": "/abs/path/to/img.jpg", "mask": "/abs/path/to/mask.png"}, ...]

    Mask: 255 = drivable (class 1), 0 = non-drivable (class 0)
    """
    def __init__(self, meta_path: str, transform=None):
        with open(meta_path) as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img  = np.array(Image.open(self.data[idx]['image']).convert('RGB'))
        mask = np.array(Image.open(self.data[idx]['mask']).convert('L'))
        mask = (mask > 127).astype(np.uint8)   # binarize → {0, 1}

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img, mask = out['image'], out['mask']

        return img, torch.tensor(mask, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  VisionEdge — Drivable Space Segmentation Training")
    print(f"{'='*60}")
    print(f"  Device    : {device}")
    print(f"  Dataroot  : {args.dataroot}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR        : {args.lr}")
    print(f"{'='*60}\n")

    # Model
    model     = UNet(num_classes=2).to(device)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_p:,} ({total_p/1e6:.1f}M)\n")

    criterion = CombinedLoss(focal_w=0.5, dice_w=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Data
    num_workers = 0 if os.name == 'nt' else 4   # Windows: 0 workers

    train_ds = SegDataset(
        os.path.join(args.dataroot, 'train_meta.json'),
        get_train_transforms()
    )
    val_ds = SegDataset(
        os.path.join(args.dataroot, 'val_meta.json'),
        get_val_transforms()
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}")
    print(f"  Train batches : {len(train_dl)}")
    print(f"  Val   batches : {len(val_dl)}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    best_miou  = 0.0
    history    = []

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        t0         = time.time()

        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_dl)

        # ── Validate ───────────────────────────────────────────
        model.eval()
        miou_scores, acc_scores = [], []
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                miou_scores.append(compute_miou(preds, masks))
                acc_scores.append(compute_pixel_accuracy(preds, masks))

        val_miou = sum(miou_scores) / len(miou_scores)
        val_acc  = sum(acc_scores)  / len(acc_scores)
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val mIoU: {val_miou:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        history.append({
            'epoch': epoch, 'loss': avg_loss,
            'val_miou': val_miou, 'val_acc': val_acc
        })

        # Save best checkpoint
        if val_miou > best_miou:
            best_miou = val_miou
            ckpt_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou':   val_miou,
                'val_acc':    val_acc,
            }, ckpt_path)
            print(f"  ✅  Best model saved → {ckpt_path}  (mIoU={best_miou:.4f})")

    # Save training history
    hist_path = os.path.join(args.output_dir, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!  Best Val mIoU: {best_miou:.4f}")
    print(f"  Checkpoint: {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train MobileNetV2+UNet segmentation model')
    p.add_argument('--dataroot',   required=True, help='Path to processed/synthetic data dir')
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch_size', type=int,   default=16)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--output_dir', default='./checkpoints')
    train(p.parse_args())
