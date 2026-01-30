# PraxisNet

A multi-label image classification model family for scoring CERAD (Consortium to Establish a Registry for Alzheimer's Disease) constructional praxis drawings. Built on a DINOv2 ViT-B/14 backbone with task-specific classification heads.

**Current Release:** CircleScore_v2.0

## What This Does

The CERAD constructional praxis test asks participants to copy four geometric figures:
- **Circle**
- **Diamond**
- **Overlapping Rectangles**
- **Cube**

This model classifies photographs of these drawings, handling real-world conditions (varied lighting, backgrounds, hands in frame, multiple objects, etc.).

### CircleScore_v2.0

Classifies circle drawings along three dimensions:

| Dimension | Categories | Question |
|-----------|------------|----------|
| **Presence** | `circle`, `no_circle`, `no_drawing` | Is there a circle? |
| **Closure** | `closed`, `not_closed`, `na` | Is it closed? |
| **Circularity** | `circular`, `not_circular`, `na` | Is it round? |

The 3-class presence distinguishes between:
- `circle` - A circle drawing is present
- `no_circle` - A drawing is present but it's not a circle
- `no_drawing` - No drawing present (blank page)

**Architecture:** Single model (~330 MB) with 3 classification heads sharing a DINOv2 ViT-B/14 backbone.

### CERAD Circle Scoring Criteria

From the original CERAD protocol, the circle is worth 2 points out of 11 total:

| Criterion | Points |
|-----------|--------|
| Closed within 1/8" | 1 |
| Circular shape | 1 |
| **Total for circle** | **2** |

Other figures: Diamond (3 pts), Overlapping Rectangles (2 pts), Cube (4 pts).

**CERAD scoring is binary for each criterion:**
- **Closure:** Either closed within 1/8" (1 point) or not (0 points)
- **Circularity:** Either "circular shape" (1 point) or not (0 points)

The standard doesn't provide explicit guidance on how ovular is "too ovular" - it just says "circular shape." In practice:
- Clearly round → 1 point
- Noticeably elongated/ovular → 0 points

**Mapping classifier output to CERAD scores:**

| Shape | circularity label | CERAD points |
|-------|-------------------|--------------|
| Round circle | `circular` | 1 |
| Oval/elongated | `not_circular` | 0 |
| Irregular/wobbly | `not_circular` | 0 |

| Closure | closure label | CERAD points |
|---------|---------------|--------------|
| Fully closed | `closed` | 1 |
| Gap ≤ 1/8" | `closed` | 1 |
| Gap > 1/8" | `not_closed` | 0 |

Confidence scores can flag borderline cases for manual review.

### CircleScore_v2.0 Performance

**Dataset:**
- Training set: 289 images (deduplicated)
- Test set: 57 images (held out, stratified by presence)

**Test Set Metrics:**

| Dimension | Precision | Recall | Specificity | F1 | n |
|-----------|-----------|--------|-------------|-----|---|
| Presence | 100% | 100% | 100% | 100% | 57 |
| Closure | 94% | 100% | 50% | 97% | 37 |
| Circularity | 93% | 96% | 82% | 94% | 37 |

*Positive classes: presence=circle, closure=closed, circularity=circular. Closure and circularity exclude "na" cases (no circle present).*

**v1.1 → v2.0 Improvements:**

Key changes in v2.0:
- **3-class presence**: Now distinguishes `no_drawing` (blank pages) from `no_circle` (non-circle drawings)
- **Expanded training data**: 289 deduplicated images with additional non-circle examples
- **Uses closure/circularity labels for non-circle drawings**: Training includes shape characteristics of drawings that aren't circles

| Metric | v1.1 | v2.0 | Change |
|--------|------|------|--------|
| Presence Precision | 97% | 100% | +3% |
| Presence Specificity | 95% | 100% | +5% |
| Circularity Specificity | 75% | 82% | +7% |
| Circularity F1 | 91% | 94% | +3% |

**Analysis:**
- **Presence**: Perfect detection with 3-class distinction (correctly identified 4 blank pages as `no_drawing` and 16 non-circle drawings as `no_circle`)
- **Closure**: High recall (catches all closed circles); specificity limited by small sample of `not_closed` cases (n=4)
- **Circularity**: Continued improvement in specificity (+7% from v1.1), now correctly identifies 82% of non-circular shapes
- The model handles both standardized test images and real-world photographs containing reference circles

**Next Steps:**
1. Continue adding minority class examples (`not_closed`, `not_circular`) to further improve specificity
2. Consider using probability thresholds to flag low-confidence predictions for manual review
3. Extend to other CERAD figures (diamond, rectangles, cube)

## Files Overview

| File | Purpose |
|------|---------|
| `dinov2_multihead_classifier.py` | Training and inference script |
| `example_labels.csv` | Template for training labels |
| `requirements.txt` | Python dependencies |

## Platform Support

Works on all platforms with automatic GPU detection:

| Platform | GPU Used | Performance |
|----------|----------|-------------|
| Windows/Linux + NVIDIA | CUDA | Fast |
| Mac M1/M2/M3/M4 | MPS (Metal) | Fast |
| Mac Intel / No GPU | CPU | Slower |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

**Labels CSV format:**

```csv
image,presence,closure,circularity
/path/to/drawing001.jpg,circle,closed,circular
/path/to/drawing002.jpg,circle,closed,not_circular
/path/to/drawing003.jpg,circle,not_closed,circular
/path/to/drawing004.jpg,no_circle,closed,not_circular
/path/to/drawing005.jpg,no_drawing,,
```

> For `no_circle` (non-circle drawings), you can optionally include closure/circularity labels.
> Leave `closure` and `circularity` empty for `no_drawing` (blank pages).

### 3. Train

```bash
python dinov2_multihead_classifier.py \
    --mode train \
    --image_dir . \
    --labels_csv ./labels.csv \
    --save_dir ./checkpoints \
    --batch_size 8
```

### 4. Inference

```bash
python dinov2_multihead_classifier.py \
    --mode inference \
    --image_dir ./test_images \
    --model_path ./checkpoints/final_model.pt \
    --output_csv predictions.csv
```

## Configuration

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_dir` | (required) | Directory containing images |
| `--labels_csv` | (required) | Path to labels CSV |
| `--save_dir` | `./checkpoints` | Where to save model |
| `--batch_size` | `16` | Reduce to 8 or 4 if memory issues |
| `--val_split` | `0.2` | Validation fraction |
| `--lp_epochs` | `30` | Linear probe epochs |
| `--ft_epochs` | `20` | LoRA fine-tuning epochs |
| `--seed` | `42` | Random seed |

### Inference Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to trained model |
| `--image_dir` | Directory of images to classify |
| `--image_path` | Single image to classify |
| `--output_csv` | Save results to CSV |
| `--output_json` | Save results to JSON |
| `--no_tta` | Disable test-time augmentation |

## Architecture

```
┌─────────────────────────────────────────┐
│      CircleScore_v2.0 (~330 MB)         │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   DINOv2 ViT-B/14 Backbone      │   │
│  │   (shared feature extractor)    │   │
│  └──────────────┬──────────────────┘   │
│                 │                       │
│      ┌──────────┼──────────┐           │
│      ▼          ▼          ▼           │
│  ┌───────┐ ┌────────┐ ┌───────────┐   │
│  │Head 1 │ │Head 2  │ │Head 3     │   │
│  └───────┘ └────────┘ └───────────┘   │
│  presence   closure   circularity      │
└─────────────────────────────────────────┘
```

## Training Strategy (LP-FT)

1. **Stage 1 - Linear Probing** (30 epochs)
   - Freeze DINOv2 backbone
   - Train only classification heads
   - ~0.01% parameters trainable

2. **Stage 2 - LoRA Fine-Tuning** (20 epochs)
   - Add LoRA adapters to backbone
   - Train LoRA + heads
   - ~0.3% parameters trainable

## Troubleshooting

**"MPS not available" on Mac:**
```bash
pip install --upgrade torch torchvision
```

**Out of memory:**
```bash
--batch_size 4  # Reduce batch size
```

## Future Work

### Additional PraxisNet Models
Extend to score the remaining CERAD figures:
- **DiamondScore** - angles, symmetry, closure
- **RectangleScore** - overlap accuracy, line quality
- **CubeScore** - 3D representation, perspective

### YOLO Pre-processing
Use an object detection model (e.g., YOLOv11) as a pre-processor to:
- Automatically detect and crop the drawing region from cluttered images
- Handle photographs containing multiple drawings
- Filter images where no drawing is detected
- Provide more consistent input to the classifier

This would add complexity (two-stage pipeline) but could improve accuracy on highly variable real-world images where drawings are small or partially obscured.

## References

- [CERAD Constructional Praxis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2863549/)
- [DINOv2: Learning Robust Visual Features](https://arxiv.org/abs/2304.07193)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [LP-FT Strategy](https://arxiv.org/abs/2202.10054)
