"""
Training, evaluation, and multi-seed benchmarking pipeline for 1D CNNs on PTB-XL.

Key features:
1. Multi-label classification with BCEWithLogitsLoss.
2. Macro and per-class AUROC calculation using scikit-learn.
3. Early stopping based on validation Macro AUROC.
4. Model checkpointing.
5. Extraction of lead-wise attention distributions.
6. Deterministic seed management for reproducibility.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import SUPERCLASSES, STANDARD_LEAD_NAMES


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)

        loss = criterion(logits, y_batch)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        running_loss += loss.item() * len(x_batch)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluates model on a DataLoader.

    Returns:
        avg_loss: Average BCE loss.
        macro_auroc: Macro-averaged AUROC across 5 superclasses.
        per_class_auroc: Dict mapping each class name to its AUROC.
        y_trues: Binary ground truth matrix (N, 5).
        y_preds: Predicted probabilities matrix (N, 5).
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        running_loss += loss.item() * len(x_batch)

        probs = torch.sigmoid(logits)
        all_targets.append(y_batch.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_trues = np.vstack(all_targets)
    y_preds = np.vstack(all_probs)
    avg_loss = running_loss / len(loader.dataset)

    # Compute per-class AUROC
    per_class_scores = roc_auc_score(y_trues, y_preds, average=None)
    macro_auroc = float(np.mean(per_class_scores))
    per_class_auroc = {
        cls: float(score) for cls, score in zip(SUPERCLASSES, per_class_scores)
    }

    return avg_loss, macro_auroc, per_class_auroc, y_trues, y_preds


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = "checkpoints/best_model.pt",
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Complete training loop with learning rate decay and early stopping
    driven by Validation Macro AUROC.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5)

    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_auroc": [],
    }

    best_val_auroc = -1.0
    epochs_no_improve = 0

    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Model", disable=not verbose)
    for epoch in epoch_pbar:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auroc, val_per_class, _, _ = evaluate_model(
            model, val_loader, criterion, device
        )

        scheduler.step(val_auroc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_auroc"].append(val_auroc)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_no_improve = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        if verbose:
            epoch_pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_auroc=f"{val_auroc:.4f}",
                best_auroc=f"{best_val_auroc:.4f}",
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch} (Patience={patience})")
            break

    # Restore best checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        if verbose:
            print(f"Loaded best checkpoint with Val Macro AUROC: {best_val_auroc:.4f}")

    return history


@torch.no_grad()
def extract_lead_attention_weights(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts attention weights per lead across a dataset loader.

    Returns:
        attn_weights: array of shape (N, 12), sum across leads = 1.0 per sample
        targets: binary label array of shape (N, 5)
    """
    model.eval()
    all_attns = []
    all_targets = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        # Calling model with return_attention=True
        if hasattr(model, "forward") and "return_attention" in model.forward.__code__.co_varnames:
            _, attn = model(x_batch, return_attention=True)
            all_attns.append(attn.cpu().numpy())
            all_targets.append(y_batch.numpy())
        else:
            raise ValueError("Model does not provide return_attention mechanism.")

    return np.vstack(all_attns), np.vstack(all_targets)
