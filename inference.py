"""
Single-image inference + FPS benchmark.

Usage:
    python inference.py \
        --image_path   path/to/image.jpg \
        --checkpoint   checkpoints/best_model.pth \
        --output_path  outputs/sample_masks/output_mask.png
"""

import argparse
import time
import os
import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.unet import UNet


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a green drivable-space mask on the original image."""
    overlay = image.copy()
    green   = np.zeros_like(image)
    green[:, :, 1] = 180   # green channel
    overlay[mask == 1] = (
        (1 - alpha) * image[mask == 1] + alpha * green[mask == 1]
    ).astype(np.uint8)
    return overlay


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  VisionEdge Inference | device={device}")

    # Load model
    model = UNet(num_classes=2).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Pre-processing transform
    transform = A.Compose([
        A.Resize(256, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Load image
    orig_img = np.array(Image.open(args.image_path).convert('RGB'))
    inp      = transform(image=orig_img)['image'].unsqueeze(0).to(device)

    # Warm up
    with torch.no_grad():
        _ = model(inp)

    # FPS benchmark (100 repeat passes)
    n_reps = 100
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_reps):
            out = model(inp)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fps = n_reps / (time.perf_counter() - t0)

    # Generate mask
    mask = out.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Resize mask back to original size if needed
    mask_img = Image.fromarray(mask * 255)
    mask_img = mask_img.resize((orig_img.shape[1], orig_img.shape[0]),
                                Image.NEAREST)
    mask_arr = np.array(mask_img) // 255

    # Save binary mask
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    mask_img.save(args.output_path)

    # Save overlay
    overlay      = overlay_mask(orig_img, mask_arr)
    overlay_path = args.output_path.replace('.png', '_overlay.png')
    Image.fromarray(overlay).save(overlay_path)

    print(f"  ✅  Binary mask  → {args.output_path}")
    print(f"  ✅  Overlay      → {overlay_path}")
    print(f"  📊  FPS (x{n_reps} run): {fps:.1f}  ({'✓ ≥30' if fps >= 30 else '✗ <30'})")
    return fps


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run inference on a single image')
    p.add_argument('--image_path',  required=True)
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--output_path', default='outputs/sample_masks/output_mask.png')
    infer(p.parse_args())
