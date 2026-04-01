"""
Synthetic Dataset Generator for Drivable Space Segmentation.

Since the full nuScenes dataset requires a license and significant download time,
this script generates a small synthetic dataset that closely mimics the nuScenes
data format (images + binary masks).  This is used for:
  - Local development and model smoke-testing
  - Demonstrating the full training pipeline end-to-end

For the actual competition submission, replace with real nuScenes data using
scripts/prepare_data.py.

Synthetic data format mirrors nuScenes:
  - Images: 512×256 RGB JPEGs (realistic road/sky textures)
  - Masks:  512×256 PNGs (255=drivable, 0=non-drivable)
  - Meta:   train_meta.json / val_meta.json / test_meta.json
"""

import os
import json
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_road_image(w: int = 512, h: int = 256, seed: int = 0) -> np.ndarray:
    """Generate a synthetic driving scene image."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    img = Image.new('RGB', (w, h))
    draw = ImageDraw.Draw(img)

    # Sky gradient (top half)
    sky_top    = (rng.randint(80, 140), rng.randint(120, 180), rng.randint(180, 230))
    sky_bottom = (rng.randint(160, 210), rng.randint(180, 220), rng.randint(210, 245))
    horizon_y  = int(h * rng.uniform(0.35, 0.55))
    for y in range(horizon_y):
        t = y / horizon_y
        r = int(sky_top[0] + t * (sky_bottom[0] - sky_top[0]))
        g = int(sky_top[1] + t * (sky_bottom[1] - sky_top[1]))
        b = int(sky_top[2] + t * (sky_bottom[2] - sky_top[2]))
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Road (bottom half, trapezoid)
    road_grey = rng.randint(80, 120)
    road_col  = (road_grey, road_grey, road_grey + rng.randint(-10, 10))
    vp_x = w // 2 + rng.randint(-60, 60)   # vanishing point
    road_pts = [
        (vp_x - 40, horizon_y),
        (vp_x + 40, horizon_y),
        (w,         h),
        (0,         h)
    ]
    draw.polygon(road_pts, fill=road_col)

    # Grass on sides
    grass_col = (rng.randint(40, 90), rng.randint(90, 140), rng.randint(30, 70))
    # left grass
    draw.polygon([(0, horizon_y), (vp_x - 40, horizon_y), (0, h)], fill=grass_col)
    # right grass
    draw.polygon([(vp_x + 40, horizon_y), (w, horizon_y), (w, h)], fill=grass_col)

    # Road markings
    marking_col = (220, 220, 200)
    for i in range(5):
        y1 = int(horizon_y + (h - horizon_y) * (i * 0.18))
        y2 = int(horizon_y + (h - horizon_y) * (i * 0.18 + 0.08))
        cx = vp_x
        draw.line([(cx, y1), (cx, y2)], fill=marking_col, width=max(1, (i + 1) * 2))

    # Add noise texture
    arr = np.array(img).astype(np.float32)
    noise = np_rng.randn(*arr.shape) * 8
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img   = Image.fromarray(arr)

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return np.array(img)


def make_road_mask(w: int = 512, h: int = 256, seed: int = 0) -> np.ndarray:
    """Generate a binary segmentation mask matching the road in the image above."""
    rng = random.Random(seed)
    horizon_y = int(h * rng.uniform(0.35, 0.55))
    vp_x      = w // 2 + rng.randint(-60, 60)

    mask = np.zeros((h, w), dtype=np.uint8)
    img  = Image.fromarray(mask)
    draw = ImageDraw.Draw(img)
    road_pts = [
        (vp_x - 40, horizon_y),
        (vp_x + 40, horizon_y),
        (w,          h),
        (0,          h)
    ]
    draw.polygon(road_pts, fill=255)
    return np.array(img)


def generate_dataset(out_dir: str, total: int = 200, seed: int = 42):
    """
    Generate a synthetic dataset and write train/val/test metadata JSON files.

    Args:
        out_dir: destination directory
        total:   total number of image-mask pairs to generate
        seed:    random seed for reproducibility
    """
    random.seed(seed)

    img_dir  = os.path.join(out_dir, 'images')
    mask_dir = os.path.join(out_dir, 'masks')
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    records = []
    for i in range(total):
        img_arr  = make_road_image(seed=i)
        mask_arr = make_road_mask(seed=i)

        img_path  = os.path.join(img_dir,  f'{i:05d}.jpg')
        mask_path = os.path.join(mask_dir, f'{i:05d}.png')

        Image.fromarray(img_arr).save(img_path, quality=95)
        Image.fromarray(mask_arr).save(mask_path)
        records.append({'image': img_path, 'mask': mask_path})

    # 70 / 15 / 15 split
    random.shuffle(records)
    n_train = int(0.70 * total)
    n_val   = int(0.15 * total)
    train_r = records[:n_train]
    val_r   = records[n_train: n_train + n_val]
    test_r  = records[n_train + n_val:]

    for split, data in [('train', train_r), ('val', val_r), ('test', test_r)]:
        meta_path = os.path.join(out_dir, f'{split}_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  {split:5s}: {len(data):4d} samples  →  {meta_path}")

    print(f"\n✅ Synthetic dataset ({total} samples) saved to: {out_dir}")
    return out_dir


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate synthetic drivable-space dataset')
    p.add_argument('--out_dir', default='./data/synthetic', help='Output directory')
    p.add_argument('--total',   type=int, default=200,      help='Number of samples')
    p.add_argument('--seed',    type=int, default=42,       help='Random seed')
    args = p.parse_args()
    generate_dataset(args.out_dir, args.total, args.seed)
