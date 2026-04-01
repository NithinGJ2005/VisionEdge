# VisionEdge Project Changelog

This document tracks all modifications, updates, and bug fixes applied to the VisionEdge project.

## [Unreleased]
### Added
- Created `CHANGELOG.md` to track project modifications and updates in minute detail.

### Fixed
- **PyTorch 2.6 Dynamo ONNX Export Bug:** 
  - Identified a critical issue where `torch.onnx.export` in PyTorch 2.6 dropped model weights, resulting in a ~0.4MB corrupted ONNX file instead of the full ~11MB model.
  - Initial attempt to use `fallback=True` to trigger the legacy C++ TorchScript exporter failed as it was not supported in the downgraded environment/specific PyTorch builds correctly.
  - Final resolution involved downgrading the Colab environment to PyTorch 2.4.1 (`torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1`).
  - Modified `export_onnx.py` to use `opset_version=14` and removed the `fallback=True` flag, ensuring successful export of the complete 11.0MB model.
- **Colab Unzip Path Separator Issue:**
  - Resolved an issue where Windows-based zip archives caused Linux Colab to extract files improperly due to backslashes (`\`) in paths.
  - Implemented a Python-based extraction script in the Colab notebook to convert backslashes to forward slashes (`/`), successfully restoring the `models/` directory structure.

### Modified
- `export_onnx.py`: 
  - Set `opset_version=14` for broad compatibility.
  - Used `weights_only=False` for `torch.load` to handle the `.pth` checkpoint correctly.
  - Verified successful model loading and tracing for the final 11.0MB ONNX export.
