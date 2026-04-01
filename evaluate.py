"""
Evaluation script — computes mIoU, pixel accuracy, and per-class IoU on test set.

Usage:
    python evaluate.py \
        --dataroot   ./data/synthetic \
        --checkpoint ./checkpoints/best_model.pth
"""

import argparse
import os
import json
import time
import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from utils.metrics import compute_miou, compute_pixel_accuracy, compute_per_class_iou
from utils.augmentations import get_val_transforms
from train import SegDataset


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  VisionEdge — Segmentation Evaluation")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"{'='*60}\n")

    # Load model
    model = UNet(num_classes=2).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
              f"val mIoU={ckpt.get('val_miou', '?'):.4f}")
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Dataset
    num_workers = 0 if os.name == 'nt' else 4
    meta_path   = os.path.join(args.dataroot, 'test_meta.json')
    ds = SegDataset(meta_path, get_val_transforms())
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=num_workers)

    print(f"  Test samples: {len(ds)}\n")

    mious, accs, road_ious, nonroad_ious = [], [], [], []

    # FPS benchmark on GPU/CPU
    fps_times = []

    with torch.no_grad():
        for imgs, masks in dl:
            imgs, masks = imgs.to(device), masks.to(device)
            t0 = time.perf_counter()
            preds = model(imgs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            fps_times.append(time.perf_counter() - t0)

            mious.append(compute_miou(preds, masks))
            accs.append(compute_pixel_accuracy(preds, masks))
            per_cls = compute_per_class_iou(preds, masks)
            nonroad_ious.append(per_cls[0])
            road_ious.append(per_cls[1])

    # Average metrics
    avg_miou        = sum(mious) / len(mious)
    avg_acc         = sum(accs)  / len(accs)
    avg_road_iou    = sum(road_ious) / len(road_ious)
    avg_nonroad_iou = sum(nonroad_ious) / len(nonroad_ious)

    # FPS: batch_size / time_per_batch (exclude first warm-up batch)
    if len(fps_times) > 1:
        avg_time = sum(fps_times[1:]) / len(fps_times[1:])
        fps      = 8 / avg_time   # batch_size=8
    else:
        fps = 8 / fps_times[0]

    print(f"{'='*60}")
    print(f"  Test mIoU          : {avg_miou:.4f}")
    print(f"  Test Pixel Accuracy: {avg_acc:.4f}")
    print(f"  Road IoU (class 1) : {avg_road_iou:.4f}")
    print(f"  Non-Road IoU (cl0) : {avg_nonroad_iou:.4f}")
    print(f"  Estimated FPS      : {fps:.1f}")
    print(f"{'='*60}\n")

    # Save results
    results = {
        'test_miou':       round(avg_miou,        4),
        'pixel_accuracy':  round(avg_acc,          4),
        'road_iou':        round(avg_road_iou,     4),
        'nonroad_iou':     round(avg_nonroad_iou,  4),
        'fps':             round(fps,              1),
        'device':          str(device),
    }
    out_path = os.path.join(os.path.dirname(args.checkpoint), 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evaluate segmentation model')
    p.add_argument('--dataroot',   required=True)
    p.add_argument('--checkpoint', required=True)
    evaluate(p.parse_args())
