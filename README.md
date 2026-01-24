# DINOv2 Image Classifier with LP-FT Strategy

A production-ready image classification pipeline optimized for **small datasets (~100 labeled examples)**, implementing the state-of-the-art LP-FT (Linear Probe then Fine-Tune) strategy with DINOv2 and LoRA.

## 🎯 What This Does

This script implements the research-backed optimal approach for fine-tuning vision models with limited data:

1. **Stage 1 - Linear Probing**: Freezes the DINOv2 backbone and trains only a classification head
2. **Stage 2 - LoRA Fine-Tuning**: Applies parameter-efficient LoRA adapters to the backbone

This two-stage approach consistently outperforms full fine-tuning by 5-10% when data is scarce.

## 📁 Data Organization

Organize your images into the following folder structure:

```
data/
├── train/
│   ├── category_1/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── category_2/
│   │   ├── image001.jpg
│   │   └── ...
│   └── category_N/
│       └── ...
└── val/
    ├── category_1/
    │   └── ...
    ├── category_2/
    │   └── ...
    └── category_N/
        └── ...
```

**Tips for splitting your ~100 images:**
- Use approximately 80% for training, 20% for validation
- Ensure each category has at least 2-3 validation images
- Keep the class distribution balanced if possible

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

If you're using a GPU (recommended), ensure you have CUDA installed and install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Train Your Model

```bash
python dinov2_classifier.py \
    --mode train \
    --data_dir ./data \
    --num_classes 5 \
    --save_dir ./checkpoints
```

### 3. Run Inference

**Single image:**
```bash
python dinov2_classifier.py \
    --mode inference \
    --model_path ./checkpoints/final_model.pt \
    --image_path ./test_image.jpg
```

**Batch inference (entire directory):**
```bash
python dinov2_classifier.py \
    --mode inference \
    --model_path ./checkpoints/final_model.pt \
    --image_dir ./test_images/
```

## ⚙️ Configuration Options

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | (required) | Path to data directory with train/val subdirs |
| `--num_classes` | (required) | Number of classification categories |
| `--save_dir` | `./checkpoints` | Directory to save model checkpoints |
| `--batch_size` | `16` | Training batch size |
| `--lp_epochs` | `30` | Number of epochs for linear probing stage |
| `--ft_epochs` | `20` | Number of epochs for LoRA fine-tuning stage |
| `--seed` | `42` | Random seed for reproducibility |

### Inference Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to trained model checkpoint |
| `--image_path` | Path to single image for prediction |
| `--image_dir` | Path to directory for batch predictions |
| `--no_tta` | Disable test-time augmentation (faster but slightly less accurate) |

## 📊 Expected Output

### Training Output

```
============================================================
STAGE 1: LINEAR PROBING (Frozen Backbone)
============================================================
Trainable parameters: 3,845 / 86,567,429 (0.00%)

Epoch 1/30
Training: 100%|████████| 6/6 [00:05<00:00, loss: 1.2345, acc: 45.00%]
Validation: 100%|████████| 2/2 [00:01<00:00]
Train Loss: 1.2345, Train Acc: 45.00%
Val Loss: 1.1234, Val Acc: 52.00%
✓ Saved best model (val_acc: 52.00%)
...

============================================================
STAGE 2: LoRA FINE-TUNING
============================================================
Trainable parameters: 297,733 / 86,567,429 (0.34%)
...

Best validation accuracy: 87.50%
Final model saved to: ./checkpoints/final_model.pt
```

### Inference Output

```
Prediction: category_2
Confidence: 94.32%

All probabilities:
  category_2: 94.32%
  category_1: 3.21%
  category_3: 1.89%
  category_4: 0.45%
  category_5: 0.13%
```

## 🔧 Advanced Customization

### Adjusting Hyperparameters

For even smaller datasets (<50 examples), you may want to modify these settings in the script:

```python
# In run_lpft_training():
lp_epochs=50,           # More epochs for linear probe
ft_epochs=10,           # Fewer epochs for fine-tuning (less overfitting risk)
lora_r=4,               # Lower rank (fewer parameters)
dropout=0.6,            # Higher dropout
```

### Programmatic Usage

```python
from dinov2_classifier import (
    run_lpft_training,
    load_trained_model,
    predict_single,
)
import torch

# Train
model, history = run_lpft_training(
    data_dir="./data",
    num_classes=5,
    save_dir="./checkpoints",
)

# Load and predict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names = load_trained_model("./checkpoints/final_model.pt", device)

predicted_class, confidence, all_probs = predict_single(
    model=model,
    image_path="./test.jpg",
    device=device,
    class_names=class_names,
    use_tta=True,
)

print(f"Predicted: {predicted_class} ({confidence:.2%})")
```

## 📈 Tips for Best Results

1. **Balance your classes**: Try to have roughly equal numbers of images per category
2. **Quality over quantity**: A few high-quality, representative images beat many similar ones
3. **Validation set matters**: Include diverse examples in validation to catch overfitting
4. **Check confusion matrix**: After training, analyze which categories are confused
5. **Test-time augmentation**: Keep TTA enabled (default) for 1-3% accuracy boost

## 🐛 Troubleshooting

**CUDA out of memory:**
- Reduce `--batch_size` to 8 or 4
- The script automatically handles gradient accumulation

**Poor validation accuracy:**
- Ensure validation images are representative
- Check for label errors in your data
- Try increasing `--lp_epochs`

**Training loss not decreasing:**
- Verify your data folder structure is correct
- Check that images are readable (not corrupted)

## 📚 References

This implementation is based on:
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [LP-FT: Linear Probing then Fine-Tuning](https://arxiv.org/abs/2202.10054)
