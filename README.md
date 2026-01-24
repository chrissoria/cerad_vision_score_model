# DINOv2 Multi-Head Classifier for Circle Drawing Analysis

A multi-label image classification pipeline for evaluating drawings along **3 separate dimensions** simultaneously, using a shared DINOv2 backbone with multiple classification heads.

## 🎯 What This Does

Classifies images of circle drawings along three dimensions:

| Dimension | Categories | Question |
|-----------|------------|----------|
| **Presence** | `circle_clear`, `circle_resembles`, `no_circle` | Is there a circle? |
| **Closure** | `closed`, `almost_closed`, `na` | Is it closed? |
| **Circularity** | `circular`, `almost_circular`, `na` | Is it round? |

**Architecture:** 1 model (~350 MB) with 3 classification heads sharing a DINOv2 backbone.

## 📦 Files Overview

| File | Purpose |
|------|---------|
| `dinov2_multihead_classifier.py` | **Training script** — train your own model |
| `circle_classifier.py` | **Inference module** — for catllm integration |
| `check_training_setup.py` | **Setup checker** — verify your machine can train |
| `example_labels.csv` | **Template** — for your training labels |
| `requirements.txt` | **Dependencies** |

## 🖥️ Platform Support

Works on all platforms with automatic GPU detection:

| Platform | GPU Used | Performance |
|----------|----------|-------------|
| Windows/Linux + NVIDIA | CUDA | Fast |
| Mac M1/M2/M3/M4 | MPS (Metal) | Fast |
| Mac Intel / No GPU | CPU | Slower |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Check Your Setup (Optional)

```bash
python check_training_setup.py
```

### 3. Prepare Your Data

**Images:** Put all images in one folder (`./images/`)

**Labels CSV:** Create `labels.csv` with this format:

```csv
image,presence,closure,circularity
drawing001.jpg,circle_clear,closed,circular
drawing002.jpg,circle_clear,almost_closed,almost_circular
drawing003.jpg,circle_resembles,almost_closed,circular
drawing004.jpg,no_circle,,
```

> Leave `closure` and `circularity` empty when `presence` is `no_circle`

### 4. Train

```bash
python dinov2_multihead_classifier.py \
    --mode train \
    --image_dir ./images \
    --labels_csv ./labels.csv \
    --save_dir ./checkpoints \
    --batch_size 8
```

### 5. Upload to HuggingFace (Optional)

```python
from circle_classifier import upload_model_to_hub

upload_model_to_hub(
    model_path="./checkpoints/final_model.pt",
    repo_id="your-username/circle-classifier",
    token="hf_..."  # or run `huggingface-cli login` first
)
```

## 💻 Using the Trained Model

### Option A: Local Inference (Default)

Downloads model (~350 MB, cached) and runs on your machine:

```python
from catllm.circle_classifier import classify_circles

# Auto-downloads from HuggingFace on first run
results = classify_circles(images="./test_images")

# Or use your local model
results = classify_circles(
    images="./test_images",
    model="./checkpoints/final_model.pt"
)
```

### Option B: Cloud Inference (HuggingFace API)

No download needed—runs on HuggingFace servers:

```python
results = classify_circles(
    images="./test_images",
    use_api=True,
    hf_token="hf_..."  # or set HF_TOKEN env var
)
```

### Output Formats

**DataFrame (default):**

```python
results = classify_circles(images="./test_images")
results.to_csv("predictions.csv", index=False)
```

```
image          | presence_clear | presence_resembles | presence_none | presence_pred | closure_closed | ...
drawing001.jpg | 0.85           | 0.10               | 0.05          | circle_clear  | 0.90           | ...
```

**JSON:**

```python
results = classify_circles(
    images="./test_images",
    output_format="json",
    output_path="predictions.json"
)
```

```json
[
  {
    "image": "drawing001.jpg",
    "presence": {"clear": 0.85, "resembles": 0.10, "none": 0.05, "prediction": "circle_clear"},
    "closure": {"closed": 0.90, "almost": 0.08, "na": 0.02, "prediction": "closed"},
    "circularity": {"circular": 0.75, "almost": 0.21, "na": 0.04, "prediction": "circular"}
  }
]
```

## ⚙️ Configuration

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

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `images` | (required) | Directory, file, or list of paths |
| `model` | `"chrissoria/circle-classifier"` | HuggingFace ID or local path |
| `output_format` | `"dataframe"` | `"dataframe"` or `"json"` |
| `output_path` | `None` | Auto-save results |
| `use_tta` | `True` | Test-time augmentation |
| `device` | `None` (auto) | `"cuda"`, `"mps"`, `"cpu"` |
| `use_api` | `False` | Use HuggingFace cloud API |
| `hf_token` | `None` | Token for cloud API |
| `silent` | `False` | Suppress messages |

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           SINGLE .pt FILE (~350 MB)     │
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

## 📈 Training Strategy (LP-FT)

1. **Stage 1 - Linear Probing** (30 epochs)
   - Freeze DINOv2 backbone
   - Train only classification heads
   - ~0.01% parameters trainable

2. **Stage 2 - LoRA Fine-Tuning** (20 epochs)
   - Add LoRA adapters to backbone
   - Train LoRA + heads
   - ~0.3% parameters trainable

## ⚡ Local vs Cloud Mode

| Feature | Local Mode | Cloud Mode (`use_api=True`) |
|---------|------------|----------------------------|
| First run | Downloads ~350MB | No download |
| Subsequent runs | Uses cache | Sends to API |
| GPU required | Recommended | No |
| Works offline | Yes | No |
| HF token needed | No | Yes |

## 🔧 Working with Results

```python
# Flag uncertain predictions
uncertain = results[results["presence_clear"] < 0.7]
print(f"{len(uncertain)} images need review")

# Custom threshold
results["strict_pred"] = results.apply(
    lambda r: "circle_clear" if r["presence_clear"] > 0.8 else "uncertain",
    axis=1
)
```

## 🐛 Troubleshooting

**"MPS not available" on Mac:**
```bash
pip install --upgrade torch torchvision
```

**Out of memory:**
```bash
--batch_size 4  # Reduce batch size
```

**Slow training (using CPU):**
```bash
python check_training_setup.py  # Verify GPU detection
```

## 📚 References

- [DINOv2: Learning Robust Visual Features](https://arxiv.org/abs/2304.07193)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [LP-FT Strategy](https://arxiv.org/abs/2202.10054)
