"""
U-Net decoder with MobileNetV2 encoder.
Architecture built from scratch — no pre-trained weights.
MAHE-Harman Track 2: Real-Time Drivable Space Segmentation | Team VisionEdge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mobilenetv2 import MobileNetV2Encoder


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    U-Net decoder block:
      1. Transpose-conv to upsample by 2×
      2. Bilinear align with skip connection
      3. Concatenate + DoubleConv
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align spatial dimensions (handles odd resolutions)
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    MobileNetV2 + U-Net segmentation model.

    Encoder skip channels (from MobileNetV2Encoder):
      skips[0] = 32   (stride-2 input stem)
      skips[1] = 16
      skips[2] = 24
      skips[3] = 32
      skips[4] = 64
      skips[5] = 96
      skips[6] = 160
      skips[7] = 320  (bottleneck, also returned as enc_out)

    Decoder path (bottleneck → output):
      dec4: 320 → 128  (uses skip[6]=160)
      dec3: 128 → 64   (uses skip[5]=96)
      dec2:  64 → 32   (uses skip[3]=32)
      dec1:  32 → 16   (uses skip[2]=24)
      dec0:  16 → 16   (uses skip[0]=32)
      final 2× upsample + 1×1 Conv → num_classes
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.encoder = MobileNetV2Encoder()

        # Decoder blocks
        self.dec4 = DecoderBlock(320, 160, 128)
        self.dec3 = DecoderBlock(128,  96,  64)
        self.dec2 = DecoderBlock( 64,  32,  32)
        self.dec1 = DecoderBlock( 32,  24,  16)
        self.dec0 = DecoderBlock( 16,  32,  16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, skips = self.encoder(x)
        # skips indices: [0]=32  [1]=16  [2]=24  [3]=32  [4]=64  [5]=96  [6]=160  [7]=320
        x = self.dec4(enc_out, skips[6])   # 320 upsampled + 160 skip → 128
        x = self.dec3(x,       skips[5])   # 128 upsampled +  96 skip →  64
        x = self.dec2(x,       skips[3])   #  64 upsampled +  32 skip →  32
        x = self.dec1(x,       skips[2])   #  32 upsampled +  24 skip →  16
        x = self.dec0(x,       skips[0])   #  16 upsampled +  32 skip →  16
        # Final 2× upsample to restore input resolution
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.final(x)


if __name__ == '__main__':
    model = UNet(num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params:,} (~{total_params/1e6:.1f}M)")
    dummy = torch.randn(1, 3, 256, 512)
    out   = model(dummy)
    print(f"Output shape     : {out.shape}")   # (1, 2, 256, 512)
