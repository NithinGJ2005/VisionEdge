"""
MobileNetV2 Encoder — built from scratch (no pre-trained weights).
MAHE-Harman Track 2: Real-Time Drivable Space Segmentation
Team: VisionEdge
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU6 block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class InvertedResidual(nn.Module):
    """MobileNetV2 bottleneck block with depth-wise separable convolutions."""
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_ch, hidden, kernel=1))
        layers += [
            ConvBNReLU(hidden, hidden, stride=stride, groups=hidden),  # depthwise
            nn.Conv2d(hidden, out_ch, 1, bias=False),                  # pointwise
            nn.BatchNorm2d(out_ch)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 encoder producing multi-scale feature maps for U-Net skip connections.
    Initialized entirely from scratch — no ImageNet weights used.

    Output channels per skip level (index → channels):
        skips[0]  → 32   (after first conv,  stride 2)
        skips[1]  → 16   (layer 0, t=1 c=16  n=1 s=1)
        skips[2]  → 24   (layer 1, t=6 c=24  n=2 s=2)
        skips[3]  → 32   (layer 2, t=6 c=32  n=3 s=2)
        skips[4]  → 64   (layer 3, t=6 c=64  n=4 s=2)
        skips[5]  → 96   (layer 4, t=6 c=96  n=3 s=1)
        skips[6]  → 160  (layer 5, t=6 c=160 n=3 s=2)
        skips[7]  → 320  (layer 6, t=6 c=320 n=1 s=1) ← bottleneck
    """
    def __init__(self):
        super().__init__()
        # (expand_ratio, out_channels, num_blocks, stride)
        config = [
            (1, 16,  1, 1),
            (6, 24,  2, 2),
            (6, 32,  3, 2),
            (6, 64,  4, 2),
            (6, 96,  3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        self.first = ConvBNReLU(3, 32, stride=2)
        self.layers = nn.ModuleList()
        in_ch = 32
        self.skip_channels = [32]
        for t, c, n, s in config:
            blocks = []
            for i in range(n):
                blocks.append(InvertedResidual(in_ch, c, s if i == 0 else 1, t))
                in_ch = c
            self.layers.append(nn.Sequential(*blocks))
            self.skip_channels.append(in_ch)

    def forward(self, x):
        skips = []
        x = self.first(x)
        skips.append(x)   # 32 ch, H/2
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
        # x == skips[-1] == 320 ch (bottleneck)
        return x, skips


if __name__ == '__main__':
    model = MobileNetV2Encoder()
    dummy = torch.randn(1, 3, 256, 512)
    out, skips = model(dummy)
    print(f"Bottleneck shape : {out.shape}")
    for i, s in enumerate(skips):
        print(f"  skip[{i}] : {s.shape}")
