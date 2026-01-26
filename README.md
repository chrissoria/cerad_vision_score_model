# PraxisNet

A multi-label image classification model family for scoring CERAD (Consortium to Establish a Registry for Alzheimer's Disease) constructional praxis drawings. Built on a DINOv2 ViT-B/14 backbone with task-specific classification heads.

**Current Release:** CircleScore_v1.0

## What This Does

The CERAD constructional praxis test asks participants to copy four geometric figures:
- **Circle**
- **Diamond**
- **Overlapping Rectangles**
- **Cube**

This model classifies photographs of these drawings, handling real-world conditions (varied lighting, backgrounds, hands in frame, multiple objects, etc.).

### CircleScore_v1.0

Classifies circle drawings along three dimensions:

| Dimension | Categories | Question |
|-----------|------------|----------|
| **Presence** | `circle`, `no_circle` | Is there a circle? |
| **Closure** | `closed`, `not_closed`, `na` | Is it closed? |
| **Circularity** | `circular`, `not_circular`, `na` | Is it round? |

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

### CircleScore_v1.0 Performance

**Dataset:**
- Training set: 220 images
- Test set: 57 images (held out, stratified by presence)

**Test Set Metrics:**

| Dimension | Precision | Recall | Specificity | F1 | n |
|-----------|-----------|--------|-------------|-----|---|
| Presence | 100% | 100% | 100% | 100% | 57 |
| Closure | 97% | 100% | 75% | 99% | 37 |
| Circularity | 74% | 92% | 33% | 82% | 37 |

*Positive classes: presence=circle, closure=closed, circularity=circular. Closure and circularity exclude "na" cases (no circle present).*

**Analysis:**
- **Presence**: Perfect detection of whether a circle is present
- **Closure**: High recall (catches all closed circles), lower specificity due to small sample of not_closed cases (n=4)
- **Circularity**: High recall (92%) but low specificity (33%)—the model tends to be lenient, calling shapes "circular" when they may not be. This reflects the inherent subjectivity of this criterion; even trained human raters may disagree on borderline cases
- The model handles both standardized test images and real-world photographs containing reference circles (thick black printed circles shown to participants as examples). It learns to distinguish the drawn circle (thin pen strokes) from the reference

**Next Steps:**
1. Review misclassified circularity cases to identify labeling ambiguities vs true model errors
2. Add more training examples of clearly oval/irregular shapes labeled `not_circular`
3. Consider using probability thresholds to flag low-confidence predictions for manual review

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
/path/to/drawing004.jpg,no_circle,,
```

> Leave `closure` and `circularity` empty when `presence` is `no_circle`

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
│      CircleScore_v1.0 (~330 MB)         │
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
