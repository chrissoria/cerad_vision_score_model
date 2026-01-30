# PraxisNet

A multi-label image classification model family for scoring CERAD (Consortium to Establish a Registry for Alzheimer's Disease) constructional praxis drawings. Built on a DINOv2 ViT-B/14 backbone with task-specific classification heads.

## What This Does

The CERAD constructional praxis test asks participants to copy four geometric figures:
- **Circle** - [CircleScore v2.0](./circle/README.md) ✓
- **Diamond** - (future)
- **Overlapping Rectangles** - (future)
- **Cube** - (future)

This project classifies photographs of these drawings, handling real-world conditions (varied lighting, backgrounds, hands in frame, multiple objects, etc.).

Each figure has its own model with shape-specific classification heads. See individual shape directories for detailed metrics and usage.

## Project Structure

```
cerad_scoring_model/
├── circle/                          # Circle scoring model
│   ├── README.md                    # Circle-specific docs & metrics
│   ├── train_data.csv
│   ├── test_data.csv
│   └── checkpoints/                 # Model weights (not in git)
├── diamond/                         # (future)
├── rectangles/                      # (future)
├── cube/                            # (future)
├── dinov2_multihead_classifier.py   # Shared training/inference code
├── requirements.txt                 # Python dependencies
└── README.md
```

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
    --labels_csv ./circle/train_data.csv \
    --save_dir ./circle/checkpoints \
    --batch_size 8
```

### 4. Inference

```bash
python dinov2_multihead_classifier.py \
    --mode inference \
    --image_dir ./test_images \
    --model_path ./circle/checkpoints/final_model_v2.pt \
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
│      PraxisNet Model (~330 MB)          │
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
│  (shape-specific classification heads) │
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
