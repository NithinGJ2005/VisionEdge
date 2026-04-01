"""
ONNX export script for edge deployment (Qualcomm Snapdragon, etc.).

Usage:
    python export_onnx.py \
        --checkpoint checkpoints/best_model.pth \
        --output     model.onnx
"""

import argparse
import torch
from models.unet import UNet


def export(args):
    device = torch.device('cpu')   # export on CPU for max compatibility
    model  = UNet(num_classes=2).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    dummy = torch.randn(1, 3, 256, 512)

    torch.onnx.export(
        model, dummy, args.output,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch'},
            'output': {0: 'batch'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    print(f"✅ ONNX model exported → {args.output}")

    # Quick size report
    import os
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"   Model file size: {size_mb:.1f} MB")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Export model to ONNX format')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output',     default='model.onnx')
    export(p.parse_args())
