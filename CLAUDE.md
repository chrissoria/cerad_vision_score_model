# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based multi-label image classification pipeline for scoring CERAD (Consortium to Establish a Registry for Alzheimer's Disease) constructional praxis drawings. The CERAD test includes four figures: circle, diamond, overlapping rectangles, and cube.

Currently implements **circle scoring** using a DINOv2 ViT-B/14 backbone with three independent classification heads:

- **Presence**: circle, no_circle
- **Closure**: closed, not_closed, na
- **Circularity**: circular, not_circular, na

The model handles real-world image conditions (varied lighting, backgrounds, hands in frame, etc.).

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Training
```bash
python dinov2_multihead_classifier.py --mode train \
    --image_dir . \
    --labels_csv ./labels.csv \
    --save_dir ./checkpoints \
    --batch_size 8
```

### Inference
```bash
python dinov2_multihead_classifier.py --mode inference \
    --image_dir ./test \
    --model_path ./checkpoints/final_model.pt \
    --output_csv predictions.csv
```

## Architecture

The entire implementation is contained in `dinov2_multihead_classifier.py` (~1100 lines). Key components:

1. **DIMENSION_CONFIG** (line ~59): Category definitions for all three classification dimensions
2. **MultiLabelDrawingDataset** (line ~120): PyTorch Dataset with image augmentation
3. **DINOv2MultiHeadClassifier** (line ~231): Main model class - shared DINOv2 backbone + 3 classification heads
4. **Training pipeline** (line ~330): Two-stage LP-FT (Linear Probe then LoRA Fine-Tune) strategy
5. **Inference pipeline** (line ~540): Single image and batch prediction with optional TTA (Test-Time Augmentation)
6. **CLI** (line ~1024): argparse-based entry point with "train" and "inference" modes

### Training Strategy

Uses LP-FT (Linear Probe + Fine-Tune) approach:
- **Stage 1 - Linear Probing** (~30 epochs): Freeze backbone, train only classification heads
- **Stage 2 - LoRA Fine-Tuning** (~20 epochs): Add LoRA adapters to backbone, train LoRA + heads

### Device Detection

Auto-detects and uses the best available device: CUDA > MPS (Apple Silicon) > CPU. See `get_device()` function.

## Labels CSV Format

```csv
image,presence,closure,circularity
/full/path/to/drawing001.jpg,circle,closed,circular
/full/path/to/drawing002.jpg,circle,not_closed,not_circular
/full/path/to/drawing003.jpg,no_circle,,
```

Leave closure and circularity empty when presence is `no_circle`.

## Future Work

- Extend to diamond, overlapping rectangles, and cube scoring
- YOLO pre-processing for automatic drawing detection/cropping
