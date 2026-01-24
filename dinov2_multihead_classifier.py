"""
DINOv2 Multi-Head Image Classifier for Circle Drawing Analysis

This script implements a multi-label classification approach for evaluating
drawings along multiple dimensions simultaneously:
    1. Presence: Does the image contain a circle?
    2. Closure: Is the circle closed?
    3. Circularity: Is the circle circular?

Architecture:
    - Shared DINOv2 ViT-B/14 backbone (frozen or LoRA-adapted)
    - 3 separate classification heads (one per dimension)
    - LP-FT training strategy with aggressive regularization

Usage:
    1. Create a labels CSV file with columns:
       image,presence,closure,circularity
       drawing001.jpg,circle_clear,closed,circular
       drawing002.jpg,circle_resembles,almost_closed,almost_circular
       drawing003.jpg,no_circle,,
    
    2. Run training:
       python dinov2_multihead_classifier.py --mode train --data_dir ./images --labels_csv ./labels.csv

    3. Run inference:
       python dinov2_multihead_classifier.py --mode inference --image_dir ./test --model_path ./checkpoints/final_model.pt

Author: Generated for research project
License: MIT
"""

import os
import argparse
import json
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Hugging Face imports
from transformers import Dinov2Model
from peft import LoraConfig, get_peft_model


# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================

# Define the 3 classification dimensions with full descriptions
DIMENSION_CONFIG = {
    "presence": {
        "categories": ["circle_clear", "circle_resembles", "no_circle"],
        "descriptions": {
            "circle_clear": "The image contains a drawing that clearly represents a circle",
            "circle_resembles": "The image contains a drawing that resembles a circle",
            "no_circle": "The image does NOT contain any drawing that resembles a circle",
        },
        "na_allowed": False,  # Must always have a value
    },
    "closure": {
        "categories": ["closed", "almost_closed", "na"],
        "descriptions": {
            "closed": "The circle is closed",
            "almost_closed": "The circle is almost closed",
            "na": "Not applicable (no circle present)",
        },
        "na_allowed": True,  # Can be N/A if no circle
    },
    "circularity": {
        "categories": ["circular", "almost_circular", "na"],
        "descriptions": {
            "circular": "The circle is circular",
            "almost_circular": "The circle is almost circular",
            "na": "Not applicable (no circle present)",
        },
        "na_allowed": True,  # Can be N/A if no circle
    },
}


def get_category_descriptions() -> Dict[str, Dict[str, str]]:
    """Return full descriptions for all categories."""
    return {dim: config["descriptions"] for dim, config in DIMENSION_CONFIG.items()}


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# DATA HANDLING
# =============================================================================

class MultiLabelDrawingDataset(Dataset):
    """Dataset for multi-label circle drawing classification."""
    
    def __init__(
        self,
        image_dir: str,
        labels_df: pd.DataFrame,
        transform: transforms.Compose,
        dimension_config: Dict = DIMENSION_CONFIG,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.dimension_config = dimension_config
        
        # Filter to only images that exist
        self.labels_df = labels_df.copy()
        existing_mask = self.labels_df["image"].apply(
            lambda x: (self.image_dir / x).exists()
        )
        if not existing_mask.all():
            missing = self.labels_df[~existing_mask]["image"].tolist()
            print(f"Warning: {len(missing)} images not found, skipping: {missing[:5]}...")
        self.labels_df = self.labels_df[existing_mask].reset_index(drop=True)
        
        # Build category-to-index mappings for each dimension
        self.cat_to_idx = {}
        self.idx_to_cat = {}
        for dim, config in dimension_config.items():
            self.cat_to_idx[dim] = {cat: i for i, cat in enumerate(config["categories"])}
            self.idx_to_cat[dim] = {i: cat for i, cat in enumerate(config["categories"])}
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        # Load and transform image
        image_path = self.image_dir / row["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Get labels for each dimension
        labels = {}
        for dim in self.dimension_config.keys():
            value = row.get(dim, "")
            # Handle empty/NaN values as "na" for dimensions that allow it
            if pd.isna(value) or value == "" or value is None:
                if self.dimension_config[dim]["na_allowed"]:
                    value = "na"
                else:
                    raise ValueError(f"Dimension '{dim}' requires a value for image {row['image']}")
            labels[dim] = self.cat_to_idx[dim][value]
        
        # Stack labels into tensor
        label_tensor = torch.tensor([labels[dim] for dim in self.dimension_config.keys()])
        
        return image, label_tensor
    
    def get_num_classes(self) -> Dict[str, int]:
        """Return number of classes for each dimension."""
        return {dim: len(config["categories"]) for dim, config in self.dimension_config.items()}


def get_train_transforms(image_size: int = 224):
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224):
    """Validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transforms(image_size: int = 224):
    """Test-time augmentation transforms."""
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
    ]


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class DINOv2MultiHeadClassifier(nn.Module):
    """
    DINOv2 backbone with multiple classification heads.
    
    Architecture:
        - Shared DINOv2 ViT-B/14 backbone
        - Separate classification head for each dimension
    """
    
    def __init__(
        self,
        dimension_config: Dict = DIMENSION_CONFIG,
        model_name: str = "facebook/dinov2-base",
        dropout: float = 0.5,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        self.dimension_config = dimension_config
        self.dimensions = list(dimension_config.keys())
        
        # Load DINOv2 backbone
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size  # 768 for base
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Create separate classification head for each dimension
        self.classifiers = nn.ModuleDict()
        for dim, config in dimension_config.items():
            num_classes = len(config["categories"])
            self.classifiers[dim] = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, num_classes),
            )
            # Initialize
            nn.init.xavier_uniform_(self.classifiers[dim][1].weight)
            nn.init.zeros_(self.classifiers[dim][1].bias)
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning logits for each dimension."""
        # Get shared features
        outputs = self.backbone(pixel_values)
        features = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Get predictions from each head
        logits = {}
        for dim in self.dimensions:
            logits[dim] = self.classifiers[dim](features)
        
        return logits
    
    def get_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        with torch.no_grad():
            outputs = self.backbone(pixel_values)
            features = outputs.last_hidden_state[:, 0]
        return features


def apply_lora(model: DINOv2MultiHeadClassifier, r: int = 8, alpha: int = 16, dropout: float = 0.1):
    """Apply LoRA adapters to the backbone."""
    # Unfreeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["query", "value"],
        lora_dropout=dropout,
        bias="none",
        modules_to_save=None,  # Classifiers are outside backbone
    )
    
    # Apply LoRA to backbone
    model.backbone = get_peft_model(model.backbone, lora_config)
    
    # Ensure classifiers are trainable
    for classifier in model.classifiers.values():
        for param in classifier.parameters():
            param.requires_grad = True
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


# =============================================================================
# TRAINING
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=-1)
        n_classes = pred.size(-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dimensions: List[str],
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = {dim: 0 for dim in dimensions}
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)  # Shape: (batch, num_dimensions)
        
        optimizer.zero_grad()
        logits = model(images)  # Dict of logits per dimension
        
        # Calculate loss for each dimension and sum
        loss = 0
        for i, dim in enumerate(dimensions):
            dim_labels = labels[:, i]
            loss += criterion(logits[dim], dim_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total += labels.size(0)
        
        # Calculate accuracy per dimension
        for i, dim in enumerate(dimensions):
            _, predicted = logits[dim].max(1)
            correct[dim] += predicted.eq(labels[:, i]).sum().item()
        
        avg_acc = sum(correct.values()) / (len(dimensions) * total) * 100
        pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}", "avg_acc": f"{avg_acc:.1f}%"})
    
    accuracies = {dim: 100 * correct[dim] / total for dim in dimensions}
    return total_loss / len(dataloader), accuracies


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dimensions: List[str],
) -> Tuple[float, Dict[str, float]]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = {dim: 0 for dim in dimensions}
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            
            loss = 0
            for i, dim in enumerate(dimensions):
                dim_labels = labels[:, i]
                loss += criterion(logits[dim], dim_labels)
            
            total_loss += loss.item()
            total += labels.size(0)
            
            for i, dim in enumerate(dimensions):
                _, predicted = logits[dim].max(1)
                correct[dim] += predicted.eq(labels[:, i]).sum().item()
    
    accuracies = {dim: 100 * correct[dim] / total for dim in dimensions}
    return total_loss / len(dataloader), accuracies


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
    label_smoothing: float = 0.1,
    patience: int = 10,
) -> Dict[str, Any]:
    """Full training loop."""
    model = model.to(device)
    dimensions = model.dimensions
    
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Scheduler steps once per epoch, so T_max = num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )
    
    early_stopping = EarlyStopping(patience=patience)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    
    print(f"\n{'='*60}")
    print(f"Starting {stage_name}")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, dimensions
        )
        scheduler.step()
        
        val_loss, val_acc = validate(model, val_loader, criterion, device, dimensions)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        avg_val_acc = sum(val_acc.values()) / len(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc:  {' | '.join(f'{d}: {a:.1f}%' for d, a in train_acc.items())}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {' | '.join(f'{d}: {a:.1f}%' for d, a in val_acc.items())}")
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_path = os.path.join(save_dir, f"best_model_{stage_name}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "avg_val_acc": avg_val_acc,
            }, save_path)
            print(f"✓ Saved best model (avg_val_acc: {avg_val_acc:.2f}%)")
        
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"\nBest average validation accuracy: {best_val_acc:.2f}%")
    return history


# =============================================================================
# INFERENCE
# =============================================================================

def predict_single(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    use_tta: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Predict all dimensions for a single image.
    
    Returns:
        Dict with structure:
        {
            "presence": {"circle_clear": 0.85, "circle_resembles": 0.10, "no_circle": 0.05},
            "closure": {"closed": 0.90, "almost_closed": 0.08, "na": 0.02},
            "circularity": {"circular": 0.75, "almost_circular": 0.20, "na": 0.05},
        }
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    
    if use_tta:
        tta_transforms = get_tta_transforms()
        all_logits = {dim: [] for dim in model.dimensions}
        
        with torch.no_grad():
            for transform in tta_transforms:
                img_tensor = transform(image).unsqueeze(0).to(device)
                logits = model(img_tensor)
                for dim in model.dimensions:
                    all_logits[dim].append(logits[dim])
        
        # Average logits
        avg_logits = {
            dim: torch.stack(all_logits[dim]).mean(dim=0)
            for dim in model.dimensions
        }
    else:
        transform = get_val_transforms()
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            avg_logits = model(img_tensor)
    
    # Convert to probabilities
    results = {}
    for dim in model.dimensions:
        probs = F.softmax(avg_logits[dim], dim=1).squeeze()
        categories = model.dimension_config[dim]["categories"]
        results[dim] = {
            cat: probs[i].item()
            for i, cat in enumerate(categories)
        }
    
    return results


def classify(
    input_data: Union[str, List[str], Path],
    model_path: str,
    use_tta: bool = True,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Classify images and return results as DataFrame.
    
    Args:
        input_data: Directory path, list of image paths, or single image path
        model_path: Path to trained model checkpoint
        use_tta: Whether to use test-time augmentation
        device: Torch device (auto-detected if None)
    
    Returns:
        DataFrame with columns:
            - image: filename
            - For each dimension: probability columns (short names) + prediction column
            
    Example output:
        image            | presence_clear | presence_resembles | presence_none | presence_pred | closure_closed | closure_almost | closure_na | closure_pred | ...
        drawing001.jpg   | 0.85           | 0.10               | 0.05          | circle_clear  | 0.90           | 0.08           | 0.02       | closed       | ...
    """
    if device is None:
        device = get_device()
    
    # Load model
    model, metadata = load_trained_model(model_path, device)
    
    # Collect image paths
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
    
    if isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
        if input_path.is_dir():
            image_paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in image_extensions])
        elif input_path.is_file():
            image_paths = [input_path]
        else:
            raise ValueError(f"Path does not exist: {input_data}")
    elif isinstance(input_data, list):
        image_paths = [Path(p) for p in input_data]
    else:
        raise ValueError(f"input_data must be a path or list of paths")
    
    # Short column name mapping
    SHORT_NAMES = {
        "presence": {
            "circle_clear": "clear",
            "circle_resembles": "resembles", 
            "no_circle": "none",
        },
        "closure": {
            "closed": "closed",
            "almost_closed": "almost",
            "na": "na",
        },
        "circularity": {
            "circular": "circular",
            "almost_circular": "almost",
            "na": "na",
        },
    }
    
    # Classify each image
    results = []
    for image_path in tqdm(image_paths, desc="Classifying"):
        preds = predict_single(model, str(image_path), device, use_tta)
        
        row = {"image": image_path.name}
        
        for dim, probs in preds.items():
            # Add probability columns with short names
            for cat, prob in probs.items():
                short_name = SHORT_NAMES[dim].get(cat, cat)
                col_name = f"{dim}_{short_name}"
                row[col_name] = round(prob, 4)
            
            # Add prediction column (highest probability)
            best_cat = max(probs, key=probs.get)
            row[f"{dim}_pred"] = best_cat
        
        results.append(row)
    
    # Create DataFrame with ordered columns
    df = pd.DataFrame(results)
    
    # Reorder columns: image, then each dimension's probs + pred
    column_order = ["image"]
    for dim in ["presence", "closure", "circularity"]:
        for cat in SHORT_NAMES[dim].values():
            column_order.append(f"{dim}_{cat}")
        column_order.append(f"{dim}_pred")
    
    # Only include columns that exist
    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]
    
    return df


def classify_to_json(
    input_data: Union[str, List[str], Path],
    model_path: str,
    output_path: Optional[str] = None,
    use_tta: bool = True,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Classify images and return results as nested JSON structure.
    
    Args:
        input_data: Directory path, list of image paths, or single image path
        model_path: Path to trained model checkpoint
        output_path: If provided, save JSON to this file
        use_tta: Whether to use test-time augmentation
        device: Torch device (auto-detected if None)
    
    Returns:
        List of dicts with structure:
        [
            {
                "image": "drawing001.jpg",
                "presence": {"clear": 0.85, "resembles": 0.10, "none": 0.05, "prediction": "circle_clear"},
                "closure": {"closed": 0.90, "almost": 0.08, "na": 0.02, "prediction": "closed"},
                "circularity": {"circular": 0.75, "almost": 0.21, "na": 0.04, "prediction": "circular"}
            },
            ...
        ]
    """
    if device is None:
        device = get_device()
    
    # Load model
    model, metadata = load_trained_model(model_path, device)
    
    # Collect image paths
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
    
    if isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
        if input_path.is_dir():
            image_paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in image_extensions])
        elif input_path.is_file():
            image_paths = [input_path]
        else:
            raise ValueError(f"Path does not exist: {input_data}")
    elif isinstance(input_data, list):
        image_paths = [Path(p) for p in input_data]
    else:
        raise ValueError(f"input_data must be a path or list of paths")
    
    # Short key mapping for JSON
    SHORT_KEYS = {
        "presence": {
            "circle_clear": "clear",
            "circle_resembles": "resembles",
            "no_circle": "none",
        },
        "closure": {
            "closed": "closed",
            "almost_closed": "almost",
            "na": "na",
        },
        "circularity": {
            "circular": "circular",
            "almost_circular": "almost",
            "na": "na",
        },
    }
    
    # Classify each image
    results = []
    for image_path in tqdm(image_paths, desc="Classifying"):
        preds = predict_single(model, str(image_path), device, use_tta)
        
        record = {"image": image_path.name}
        
        for dim, probs in preds.items():
            dim_result = {}
            
            # Add probabilities with short keys
            for cat, prob in probs.items():
                short_key = SHORT_KEYS[dim].get(cat, cat)
                dim_result[short_key] = round(prob, 4)
            
            # Add prediction
            best_cat = max(probs, key=probs.get)
            dim_result["prediction"] = best_cat
            
            record[dim] = dim_result
        
        results.append(record)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    return results


def results_to_json(df: pd.DataFrame, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convert a classify() DataFrame result to nested JSON structure.
    
    Args:
        df: DataFrame from classify()
        output_path: If provided, save JSON to this file
    
    Returns:
        Nested JSON structure
    """
    results = []
    
    for _, row in df.iterrows():
        record = {"image": row["image"]}
        
        for dim in ["presence", "closure", "circularity"]:
            dim_result = {}
            
            # Extract probabilities
            for col in df.columns:
                if col.startswith(f"{dim}_") and not col.endswith("_pred"):
                    short_name = col.replace(f"{dim}_", "")
                    dim_result[short_name] = row[col]
            
            # Extract prediction
            pred_col = f"{dim}_pred"
            if pred_col in df.columns:
                dim_result["prediction"] = row[pred_col]
            
            record[dim] = dim_result
        
        results.append(record)
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    return results


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def run_lpft_training(
    image_dir: str,
    labels_csv: str,
    save_dir: str = "./checkpoints",
    val_split: float = 0.2,
    batch_size: int = 16,
    lp_epochs: int = 30,
    lp_learning_rate: float = 1e-3,
    ft_epochs: int = 20,
    ft_learning_rate: float = 5e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    dropout: float = 0.5,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    patience: int = 10,
    seed: int = 42,
):
    """
    Run the full LP-FT training pipeline.
    
    Args:
        image_dir: Directory containing images
        labels_csv: Path to CSV with columns: image, presence, closure, circularity
        val_split: Fraction of data to use for validation
        ... (other args same as before)
    """
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load labels
    labels_df = pd.read_csv(labels_csv)
    print(f"Loaded {len(labels_df)} labeled images")
    print(f"Columns: {list(labels_df.columns)}")
    
    # Validate required columns
    required_cols = ["image", "presence", "closure", "circularity"]
    missing = [c for c in required_cols if c not in labels_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Train/val split
    labels_df = labels_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(labels_df) * (1 - val_split))
    train_df = labels_df.iloc[:split_idx]
    val_df = labels_df.iloc[split_idx:]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = MultiLabelDrawingDataset(image_dir, train_df, get_train_transforms())
    val_dataset = MultiLabelDrawingDataset(image_dir, val_df, get_val_transforms())
    
    # Print class distribution
    print("\nClass distribution (training):")
    for dim in ["presence", "closure", "circularity"]:
        counts = train_df[dim].fillna("na").value_counts()
        print(f"  {dim}: {dict(counts)}")
    
    # Data loaders
    # DataLoader settings (adjust for platform)
    num_workers = 0 if device.type == "mps" else 4  # multiprocessing can hang on Mac
    pin_memory = device.type == "cuda"  # pin_memory only helps for CUDA
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    # =========================================================================
    # STAGE 1: Linear Probing
    # =========================================================================
    print("\n" + "="*60)
    print("STAGE 1: LINEAR PROBING (Frozen Backbone)")
    print("="*60)
    
    model = DINOv2MultiHeadClassifier(dropout=dropout, freeze_backbone=True)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    
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
        label_smoothing=label_smoothing,
        patience=patience,
    )
    
    # =========================================================================
    # STAGE 2: LoRA Fine-Tuning
    # =========================================================================
    print("\n" + "="*60)
    print("STAGE 2: LoRA FINE-TUNING")
    print("="*60)
    
    # Load best linear probe
    checkpoint = torch.load(os.path.join(save_dir, "best_model_linear_probe.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best linear probe model (avg_val_acc: {checkpoint['avg_val_acc']:.2f}%)")
    
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
        label_smoothing=label_smoothing,
        patience=patience,
    )
    
    # Save final model with metadata
    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "dimension_config": DIMENSION_CONFIG,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }, final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    # Save training history
    history = {"linear_probe": lp_history, "lora_finetune": ft_history}
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)
    
    return model, history


def load_trained_model(
    model_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    """Load a trained model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    dimension_config = checkpoint.get("dimension_config", DIMENSION_CONFIG)
    
    model = DINOv2MultiHeadClassifier(
        dimension_config=dimension_config,
        dropout=0.5,
        freeze_backbone=True,
    )
    
    if "lora_r" in checkpoint:
        model = apply_lora(
            model,
            r=checkpoint["lora_r"],
            alpha=checkpoint["lora_alpha"],
        )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 Multi-Head Classifier for Circle Drawing Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train
  python dinov2_multihead_classifier.py --mode train --image_dir ./images --labels_csv ./labels.csv
  
  # Inference
  python dinov2_multihead_classifier.py --mode inference --image_dir ./test --model_path ./checkpoints/final_model.pt
  
  # Single image
  python dinov2_multihead_classifier.py --mode inference --image_path ./test.jpg --model_path ./checkpoints/final_model.pt
        """
    )
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"])
    
    # Training args
    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--labels_csv", type=str, help="Path to labels CSV")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lp_epochs", type=int, default=30)
    parser.add_argument("--ft_epochs", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    
    # Inference args
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--no_tta", action="store_true")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if not args.image_dir or not args.labels_csv:
            parser.error("Training requires --image_dir and --labels_csv")
        
        run_lpft_training(
            image_dir=args.image_dir,
            labels_csv=args.labels_csv,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            lp_epochs=args.lp_epochs,
            ft_epochs=args.ft_epochs,
            val_split=args.val_split,
            seed=args.seed,
        )
    
    elif args.mode == "inference":
        if not args.model_path:
            parser.error("Inference requires --model_path")
        if not args.image_dir and not args.image_path:
            parser.error("Inference requires --image_dir or --image_path")
        
        input_data = args.image_dir or args.image_path
        
        # Get DataFrame results
        results_df = classify(
            input_data=input_data,
            model_path=args.model_path,
            use_tta=not args.no_tta,
        )
        
        # Save CSV if requested
        if args.output_csv:
            results_df.to_csv(args.output_csv, index=False)
            print(f"CSV saved to: {args.output_csv}")
        
        # Save JSON if requested
        if args.output_json:
            results_json = results_to_json(results_df, args.output_json)
        
        # Default: save CSV if no output specified
        if not args.output_csv and not args.output_json:
            default_csv = "predictions.csv"
            results_df.to_csv(default_csv, index=False)
            print(f"CSV saved to: {default_csv}")
        
        # Print sample
        print("\nSample predictions:")
        print(results_df.head().to_string())


if __name__ == "__main__":
    main()
