# CircleScore

Multi-label classifier for CERAD circle drawings. Part of the PraxisNet model family.

**Current Release:** v2.0

## What This Does

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

## CERAD Circle Scoring Criteria

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

## v2.0 Performance

**Dataset:**
- Training set: 289 images (deduplicated)
- Test set: 57 images (held out, stratified by presence)

**Test Set Metrics:**

| Dimension | Accuracy | Precision | Recall | Specificity | F1 | n |
|-----------|----------|-----------|--------|-------------|-----|---|
| Presence | 100% | 100% | 100% | 100% | 100% | 57 |
| Closure | 95% | 94% | 100% | 50% | 97% | 37 |
| Circularity | 92% | 93% | 96% | 82% | 94% | 37 |

*Positive classes: presence=circle, closure=closed, circularity=circular. Closure and circularity exclude "na" cases (no circle present).*

**Analysis:**
- **Presence**: Perfect detection with 3-class distinction (correctly identified 4 blank pages as `no_drawing` and 16 non-circle drawings as `no_circle`)
- **Closure**: High recall (catches all closed circles); specificity limited by small sample of `not_closed` cases (n=4)
- **Circularity**: Correctly identifies 82% of non-circular shapes
- The model handles both standardized test images and real-world photographs containing reference circles

## Version History

### v2.0 (Current)

Key changes:
- **3-class presence**: Now distinguishes `no_drawing` (blank pages) from `no_circle` (non-circle drawings)
- **Expanded training data**: 289 deduplicated images with additional non-circle examples
- **Uses closure/circularity labels for non-circle drawings**: Training includes shape characteristics of drawings that aren't circles

| Metric | v1.1 | v2.0 | Change |
|--------|------|------|--------|
| Presence Precision | 97% | 100% | +3% |
| Presence Specificity | 95% | 100% | +5% |
| Circularity Specificity | 75% | 82% | +7% |
| Circularity F1 | 91% | 94% | +3% |

### v1.1

- 2-class presence (circle vs no_circle)
- Initial training set

## Directory Structure

```
circle/
├── train_data.csv           # Training labels
├── test_data.csv            # Test labels
├── checkpoints/             # Model weights (not in git)
│   └── final_model_v2.pt
└── labels_with_dropdowns.xlsx
```

## Usage

### Training

```bash
python dinov2_multihead_classifier.py \
    --mode train \
    --image_dir . \
    --labels_csv ./circle/train_data.csv \
    --save_dir ./circle/checkpoints \
    --batch_size 8
```

### Inference

```bash
python dinov2_multihead_classifier.py \
    --mode inference \
    --image_dir ./test_images \
    --model_path ./circle/checkpoints/final_model_v2.pt \
    --output_csv predictions.csv
```

## Next Steps

1. Continue adding minority class examples (`not_closed`, `not_circular`) to further improve specificity
2. Consider using probability thresholds to flag low-confidence predictions for manual review
