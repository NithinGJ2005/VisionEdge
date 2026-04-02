# Real-Time Drivable Space Segmentation
### MAHE-Harman AI in Mobility Challenge — Track 2 | Team VisionEdge

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

##  Project Overview

This project implements a **real-time drivable space segmentation** pipeline for Level 4 autonomous vehicles using a **MobileNetV2 encoder + U-Net decoder** architecture, trained **entirely from scratch** on the nuScenes dataset.

**Track:** Track 2 — Real-Time Drivable Space Segmentation  
**Team:** AI-liens | JNNCE - Dept of AIML 
**Key Constraint:** ⚠️ No pre-trained weights — all training from scratch on nuScenes  

> The model identifies "free space" — areas where an L4 vehicle can safely navigate — without relying on lane markings, even in construction zones, puddle-covered roads, or road-to-grass transitions.

---

##  Model Architecture

```
Input RGB (512×256)
       ↓
MobileNetV2 Encoder (depth-wise separable convs, ~3.4M params)
  └─ Multi-scale skip features: [32, 16, 24, 32, 64, 96, 160, 320]
       ↓
U-Net Decoder (skip connections preserve boundary details)
  └─ 5 decoder blocks: 320→128→64→32→16→16
       ↓
1×1 Conv → Softmax → Argmax
       ↓
Pixel-wise Mask (Road=1 | Non-Road=0) [512×256]
```

| Component  | Detail                                            |
|------------|---------------------------------------------------|
| Encoder    | MobileNetV2 (depth-wise separable convolutions)   |
| Decoder    | U-Net with transpose-conv + skip connections      |
| Loss       | Focal Loss (γ=2) + Dice Loss (50:50 weight)       |
| Parameters | ~3.4M — lightweight for Qualcomm Snapdragon       |
| Input      | 512×256 RGB images                                |
| Output     | 2-class pixel mask (drivable / non-drivable)      |

See `assets/architecture_diagram.png` for the full pipeline diagram.

---

##  Dataset Setup

### Option A: nuScenes (for full training)

1. Download nuScenes from https://www.nuscenes.org/nuscenes  
2. Install devkit: `pip install nuscenes-devkit`  
3. Process data:

```bash
python scripts/prepare_data.py \
    --dataroot /path/to/nuscenes \
    --version  v1.0-mini \
    --outdir   ./data/processed
```

### Option B: Synthetic Data (quick demo, no download needed)

```bash
python scripts/generate_synthetic_data.py --out_dir ./data/synthetic --total 200
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/visionedge-seg.git
cd visionedge-seg

# Create and activate conda environment
conda env create -f environment.yml
conda activate visionedge-seg

# OR use pip
pip install -r requirements.txt
```

---

##  How to Run

### 1. Generate Synthetic Data (or use nuScenes — see above)

```bash
python scripts/generate_synthetic_data.py --out_dir ./data/synthetic --total 200
```

### 2. Train the Model

```bash
# Quick demo (5 epochs, small batch, synthetic data)
python train.py --dataroot ./data/synthetic --epochs 5 --batch_size 4

# Full training (nuScenes, 50 epochs)
python train.py --dataroot ./data/processed --epochs 50 --batch_size 16 --lr 0.001
```

### 3. Evaluate (mIoU + FPS)

```bash
python evaluate.py \
    --dataroot   ./data/synthetic \
    --checkpoint ./checkpoints/best_model.pth
```

### 4. Inference (single image)

```bash
python inference.py \
    --image_path  data/synthetic/images/00001.jpg \
    --checkpoint  checkpoints/best_model.pth \
    --output_path outputs/sample_masks/output_mask.png
```

### 5. Export to ONNX (for edge deployment)

```bash
python export_onnx.py \
    --checkpoint checkpoints/best_model.pth \
    --output     model.onnx
```

---

##  Results

### Synthetic Dataset Demo Training (10 epochs, 200 samples)

| Metric              | Score  |
|---------------------|--------|
| **Test mIoU**       | **0.9056** |
| Pixel Accuracy      | 0.9564 |
| Road IoU (class 1)  | 0.8735 |
| Non-Road IoU (cl 0) | 0.9377 |
| FPS (CPU)           | 13.3   |
| FPS (GPU — target)  | ≥ 30   |

### Training Progression

| Epoch | Loss   | Val mIoU | Val Acc |
|-------|--------|----------|---------|
| 1     | 0.1729 | 0.8544   | 0.9315  |
| 3     | 0.1069 | 0.8627   | 0.9357  |
| 5     | 0.0768 | 0.8883   | 0.9469  |
| 7     | 0.0655 | 0.8959   | 0.9515  |
| 9     | 0.0621 | 0.9069   | 0.9567  |
| **10**| **0.0595** | **0.9066** | **0.9568** |

> **Note:** Results above are on synthetic demonstration data. Full nuScenes training (50 epochs) is expected to achieve mIoU 0.70–0.81 as targeted in the competition.

| Model                      | mIoU (synthetic demo) | FPS (CPU / GPU target) |
|----------------------------|-----------------------|------------------------|
| MobileNetV2 + U-Net (ours) | **0.9056**            | 13.3 / ≥ 30            |
| ResNet-18 + FPN (baseline) | TBD (nuScenes)        | TBD                    |

---

## 📁 Repository Structure

```
visionedge-seg/
├── README.md
├── requirements.txt
├── environment.yml
├── train.py                     ← Main training script
├── evaluate.py                  ← mIoU + FPS evaluation
├── inference.py                 ← Single image inference
├── export_onnx.py               ← ONNX export for edge deployment
│
├── models/
│   ├── mobilenetv2.py           ← MobileNetV2 encoder (from scratch)
│   └── unet.py                  ← U-Net decoder with skip connections
│
├── utils/
│   ├── losses.py                ← Focal Loss + Dice Loss
│   ├── metrics.py               ← mIoU + pixel accuracy
│   └── augmentations.py         ← Train/val data augmentation
│
├── scripts/
│   ├── prepare_data.py          ← nuScenes data extraction
│   └── generate_synthetic_data.py ← Synthetic data for demo
│
├── configs/
│   └── config.yaml              ← All hyperparameters
│
├── data/
│   └── README.md                ← Dataset setup instructions
│
├── checkpoints/                 ← Saved model weights
├── outputs/
│   └── sample_masks/            ← Example segmentation outputs
└── assets/
    └── architecture_diagram.png ← Architecture block diagram
```

---

##  Example Outputs

See `outputs/sample_masks/` for example segmentation outputs after training.

---

## 👥 Team

| Name   | Role        | Skills                               |
|--------|-------------|--------------------------------------|
| Nithin G J | Team Lead   | ML Pipeline, Python, Pytorch, Model Training  |
| Jayanth B | CV Engineer |  Deep Learning, Segmentation ,ONNX|
| Vinay N V | Data Eng.   | nuScenes Devkit, Preprocessing ,Model Optimization |

---

## Technical Details

- **No pre-trained weights** — model initialized from scratch (competition rule)
- **Optimizer:** AdamW with CosineAnnealingLR scheduler
- **Loss:** Focal Loss (handles class imbalance) + Dice Loss (overlap maximization)
- **Augmentation:** Horizontal flip, color jitter, random crop, grid distortion, Gaussian noise
- **Export:** ONNX opset 11, dynamic batch axis, Qualcomm Snapdragon compatible
- **Target:** ≥30 FPS, <33ms latency on edge hardware

---

##  License

This project is submitted as part of the MAHE-Harman AI in Mobility Challenge Round 1.
