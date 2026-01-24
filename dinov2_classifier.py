"""
DINOv2 Image Classifier with LP-FT (Linear Probe then Fine-Tune) Strategy

This script implements the optimal approach for classifying images with ~100 labeled examples:
1. Stage 1: Linear probing on frozen DINOv2 features
2. Stage 2: LoRA fine-tuning with aggressive regularization

Features:
- DINOv2 ViT-B/14 foundation model
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- RandAugment and CutMix data augmentation
- Label smoothing, dropout, weight decay regularization
- Early stopping to prevent overfitting
- Test-time augmentation for inference
- Model saving and loading utilities

Usage:
    1. Organize your images into folders by category:
       data/
         train/
           category_1/
             image1.jpg
             image2.jpg
           category_2/
             image1.jpg
         val/
           category_1/
           category_2/
    
    2. Run training:
       python dinov2_classifier.py --data_dir ./data --num_classes 5

    3. Run inference:
       python dinov2_classifier.py --mode inference --image_path ./test.jpg --model_path ./best_model

Author: Generated for research project
License: MIT
"""

import os
import argparse
import json
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

# Hugging Face imports
from transformers import (
    AutoImageProcessor,
    Dinov2Model,
    Dinov2Config,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# Set seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

class CutMix:
    """CutMix augmentation for improved regularization."""
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch_images: torch.Tensor, batch_labels: torch.Tensor):
        if random.random() > self.prob:
            return batch_images, batch_labels, batch_labels, 1.0
        
        batch_size = batch_images.size(0)
        indices = torch.randperm(batch_size)
        shuffled_images = batch_images[indices]
        shuffled_labels = batch_labels[indices]
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get random box
        _, _, H, W = batch_images.shape
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        batch_images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        
        return batch_images, batch_labels, shuffled_labels, lam


def get_train_transforms(image_size: int = 224):
    """Get training transforms with RandAugment."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224):
    """Get validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # Resize to 256 for 224 crop
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transforms(image_size: int = 224):
    """Get test-time augmentation transforms for inference."""
    return [
        # Original
        transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # Five crops (center + 4 corners)
        transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.FiveCrop(image_size),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                    transforms.ToTensor()(crop)
                ) for crop in crops
            ])),
        ]),
    ]


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class DINOv2Classifier(nn.Module):
    """
    DINOv2-based classifier with optional LoRA adaptation.
    
    Architecture:
    - DINOv2 ViT-B/14 backbone (frozen or LoRA-adapted)
    - Classification head with dropout
    """
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "facebook/dinov2-base",
        dropout: float = 0.5,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load DINOv2 backbone
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size  # 768 for base
        
        # Freeze backbone if specified (for linear probing stage)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, num_classes),
        )
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classifier."""
        # Get DINOv2 features (use CLS token)
        outputs = self.backbone(pixel_values)
        features = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Classification
        logits = self.classifier(features)
        return logits
    
    def get_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features without classification (for analysis)."""
        with torch.no_grad():
            outputs = self.backbone(pixel_values)
            features = outputs.last_hidden_state[:, 0]
        return features


def apply_lora(model: DINOv2Classifier, r: int = 8, alpha: int = 16, dropout: float = 0.1):
    """
    Apply LoRA adapters to the DINOv2 backbone.
    
    Args:
        model: The DINOv2Classifier model
        r: LoRA rank (lower = fewer parameters, higher = more capacity)
        alpha: LoRA scaling factor (typically 2x rank)
        dropout: LoRA dropout for regularization
    
    Returns:
        PEFT model with LoRA adapters
    """
    # First unfreeze backbone for LoRA training
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["query", "value"],  # Apply to attention Q and V projections
        lora_dropout=dropout,
        bias="none",
        modules_to_save=["classifier"],  # Always train the classifier
    )
    
    # Apply LoRA to backbone only
    model.backbone = get_peft_model(model.backbone, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing for better calibration."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Smooth labels
        n_classes = pred.size(-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class CutMixCrossEntropy(nn.Module):
    """Cross-entropy loss that handles CutMix mixed labels."""
    
    def __init__(self, smoothing: float = 0.1):
        self.base_criterion = LabelSmoothingCrossEntropy(smoothing)
    
    def __call__(
        self,
        pred: torch.Tensor,
        target1: torch.Tensor,
        target2: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        return lam * self.base_criterion(pred, target1) + (1 - lam) * self.base_criterion(pred, target2)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cutmix: Optional[CutMix] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Apply CutMix if enabled
        if cutmix is not None:
            images, labels1, labels2, lam = cutmix(images, labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        if cutmix is not None and lam < 1.0:
            loss = lam * criterion(outputs, labels1) + (1 - lam) * criterion(outputs, labels2)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{running_loss / (batch_idx + 1):.4f}",
            "acc": f"{100. * correct / total:.2f}%"
        })
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    save_dir: str,
    stage_name: str,
    use_cutmix: bool = True,
    label_smoothing: float = 0.1,
    patience: int = 10,
) -> Dict[str, List[float]]:
    """
    Full training loop with early stopping.
    
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    
    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Learning rate scheduler (cosine annealing)
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=learning_rate * 0.01
    )
    
    # CutMix augmentation
    cutmix = CutMix(alpha=1.0, prob=0.5) if use_cutmix else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode="min")
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    best_val_acc = 0.0
    
    print(f"\n{'='*60}")
    print(f"Starting {stage_name}")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, cutmix, scheduler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, f"best_model_{stage_name}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, save_path)
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    return history


# =============================================================================
# INFERENCE
# =============================================================================

def predict_single(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    class_names: List[str],
    use_tta: bool = True,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict class for a single image with optional test-time augmentation.
    
    Args:
        model: Trained model
        image_path: Path to image file
        device: Torch device
        class_names: List of class names
        use_tta: Whether to use test-time augmentation
    
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    
    if use_tta:
        # Test-time augmentation
        tta_transforms = get_tta_transforms()
        all_probs = []
        
        with torch.no_grad():
            for transform in tta_transforms[:2]:  # Original + horizontal flip
                img_tensor = transform(image).unsqueeze(0).to(device)
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)
            
            # Five crop transform returns multiple crops
            five_crop_transform = tta_transforms[2]
            crops = five_crop_transform(image).to(device)
            for crop in crops:
                outputs = model(crop.unsqueeze(0))
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)
        
        # Average predictions
        avg_probs = torch.stack(all_probs).mean(dim=0).squeeze()
    else:
        # Single prediction
        transform = get_val_transforms()
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            avg_probs = F.softmax(outputs, dim=1).squeeze()
    
    # Get prediction
    confidence, predicted_idx = avg_probs.max(0)
    predicted_class = class_names[predicted_idx.item()]
    
    # All probabilities
    all_probabilities = {
        class_names[i]: avg_probs[i].item()
        for i in range(len(class_names))
    }
    
    return predicted_class, confidence.item(), all_probabilities


def predict_batch(
    model: nn.Module,
    image_dir: str,
    device: torch.device,
    class_names: List[str],
    use_tta: bool = True,
) -> List[Dict[str, Any]]:
    """Predict classes for all images in a directory."""
    results = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    image_paths = [
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in image_extensions
    ]
    
    for image_path in tqdm(image_paths, desc="Predicting"):
        predicted_class, confidence, all_probs = predict_single(
            model, str(image_path), device, class_names, use_tta
        )
        
        results.append({
            "image": image_path.name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": all_probs,
        })
    
    return results


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def run_lpft_training(
    data_dir: str,
    num_classes: int,
    save_dir: str = "./checkpoints",
    batch_size: int = 16,
    # Stage 1 (Linear Probe) hyperparameters
    lp_epochs: int = 30,
    lp_learning_rate: float = 1e-3,
    # Stage 2 (LoRA Fine-tune) hyperparameters
    ft_epochs: int = 20,
    ft_learning_rate: float = 5e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    # Regularization
    dropout: float = 0.5,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    patience: int = 10,
    # Other
    seed: int = 42,
):
    """
    Run the full LP-FT (Linear Probe then Fine-Tune) training pipeline.
    
    Stage 1: Linear Probing
        - Freeze DINOv2 backbone
        - Train only the classification head
        - Use basic augmentation
    
    Stage 2: LoRA Fine-Tuning
        - Apply LoRA adapters to backbone
        - Train LoRA + classification head
        - Use CutMix augmentation
    """
    set_seed(seed)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    train_dataset = ImageFolder(train_dir, transform=get_train_transforms())
    val_dataset = ImageFolder(val_dir, transform=get_val_transforms())
    
    # Save class names
    class_names = train_dataset.classes
    with open(os.path.join(save_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # =========================================================================
    # STAGE 1: Linear Probing
    # =========================================================================
    print("\n" + "="*60)
    print("STAGE 1: LINEAR PROBING (Frozen Backbone)")
    print("="*60)
    
    model = DINOv2Classifier(
        num_classes=num_classes,
        dropout=dropout,
        freeze_backbone=True,  # Freeze for linear probing
    )
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    lp_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=lp_epochs,
        learning_rate=lp_learning_rate,
        weight_decay=weight_decay,
        device=device,
        save_dir=save_dir,
        stage_name="linear_probe",
        use_cutmix=False,  # No CutMix for linear probing
        label_smoothing=label_smoothing,
        patience=patience,
    )
    
    # =========================================================================
    # STAGE 2: LoRA Fine-Tuning
    # =========================================================================
    print("\n" + "="*60)
    print("STAGE 2: LoRA FINE-TUNING")
    print("="*60)
    
    # Load best linear probe checkpoint
    checkpoint = torch.load(os.path.join(save_dir, "best_model_linear_probe.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best linear probe model (val_acc: {checkpoint['val_acc']:.2f}%)")
    
    # Apply LoRA
    model = apply_lora(model, r=lora_r, alpha=lora_alpha, dropout=0.1)
    
    ft_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=ft_epochs,
        learning_rate=ft_learning_rate,
        weight_decay=weight_decay,
        device=device,
        save_dir=save_dir,
        stage_name="lora_finetune",
        use_cutmix=True,  # Use CutMix for fine-tuning
        label_smoothing=label_smoothing,
        patience=patience,
    )
    
    # Save final model
    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }, final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    # Save training history
    history = {
        "linear_probe": lp_history,
        "lora_finetune": ft_history,
    }
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    return model, history


def load_trained_model(
    model_path: str,
    device: torch.device,
) -> Tuple[nn.Module, List[str]]:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct model
    model = DINOv2Classifier(
        num_classes=checkpoint["num_classes"],
        dropout=0.5,
        freeze_backbone=True,
    )
    
    # Apply LoRA if it was used
    if "lora_r" in checkpoint:
        model = apply_lora(
            model,
            r=checkpoint["lora_r"],
            alpha=checkpoint["lora_alpha"],
        )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint["class_names"]


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 Image Classifier with LP-FT Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python dinov2_classifier.py --data_dir ./data --num_classes 5 --save_dir ./checkpoints
  
  # Run inference on a single image
  python dinov2_classifier.py --mode inference --image_path ./test.jpg --model_path ./checkpoints/final_model.pt
  
  # Run inference on a directory
  python dinov2_classifier.py --mode inference --image_dir ./test_images --model_path ./checkpoints/final_model.pt
        """
    )
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"],
                        help="Mode: train or inference")
    
    # Training arguments
    parser.add_argument("--data_dir", type=str, help="Path to data directory (with train/val subdirs)")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save models")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lp_epochs", type=int, default=30, help="Linear probe epochs")
    parser.add_argument("--ft_epochs", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Inference arguments
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--image_path", type=str, help="Path to single image for inference")
    parser.add_argument("--image_dir", type=str, help="Path to directory of images for batch inference")
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "train":
        if not args.data_dir or not args.num_classes:
            parser.error("Training mode requires --data_dir and --num_classes")
        
        run_lpft_training(
            data_dir=args.data_dir,
            num_classes=args.num_classes,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            lp_epochs=args.lp_epochs,
            ft_epochs=args.ft_epochs,
            seed=args.seed,
        )
    
    elif args.mode == "inference":
        if not args.model_path:
            parser.error("Inference mode requires --model_path")
        if not args.image_path and not args.image_dir:
            parser.error("Inference mode requires --image_path or --image_dir")
        
        # Load model
        model, class_names = load_trained_model(args.model_path, device)
        
        if args.image_path:
            # Single image prediction
            predicted, confidence, probs = predict_single(
                model, args.image_path, device, class_names, use_tta=not args.no_tta
            )
            print(f"\nPrediction: {predicted}")
            print(f"Confidence: {confidence:.2%}")
            print("\nAll probabilities:")
            for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
                print(f"  {cls}: {prob:.2%}")
        
        elif args.image_dir:
            # Batch prediction
            results = predict_batch(
                model, args.image_dir, device, class_names, use_tta=not args.no_tta
            )
            
            # Save results
            output_path = os.path.join(args.image_dir, "predictions.json")
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_path}")
            
            # Print summary
            print(f"\nProcessed {len(results)} images")
            for r in results[:5]:
                print(f"  {r['image']}: {r['predicted_class']} ({r['confidence']:.2%})")
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more")


if __name__ == "__main__":
    main()
