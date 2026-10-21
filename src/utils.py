"""
Utilities for fair benchmarking, results aggregation, and attention visualization.

Includes:
1. Multi-seed experiment orchestrator (3 seeds reporting mean ± std).
2. Formatted Markdown / Pandas results table.
3. Attention weight per-lead bar plot generator.
"""

from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import STANDARD_LEAD_NAMES, SUPERCLASSES
from src.models import count_parameters
from src.train import (
    evaluate_model,
    extract_lead_attention_weights,
    set_seed,
    train_model,
)


def run_multiseed_experiment(
    model_fn: Callable[[], nn.Module],
    model_name: str,
    train_loader_fn: Callable[[], DataLoader],
    val_loader: DataLoader,
    test_loader: DataLoader,
    seeds: List[int] = [42, 123, 999],
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = None,
) -> Dict:
    """
    Runs training across multiple random seeds, saves test evaluations,
    and returns aggregated statistics (mean ± std).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    macro_aurocs = []
    per_class_runs = {cls: [] for cls in SUPERCLASSES}
    criterion = nn.BCEWithLogitsLoss()

    # Track parameter count from first instantiation
    sample_model = model_fn()
    param_count = count_parameters(sample_model)
    del sample_model

    print(f"\n{'='*65}")
    print(f"Starting Multi-Seed Benchmark for: {model_name} (Params: {param_count:,})")
    print(f"Seeds: {seeds}")
    print(f"{'='*65}")

    best_seed_model = None
    best_seed_auroc = -1.0

    for i, seed in enumerate(seeds, 1):
        print(f"\n--- [Seed {i}/{len(seeds)}: {seed}] ---")
        set_seed(seed)

        model = model_fn().to(device)
        train_loader = train_loader_fn()

        checkpoint = f"checkpoints/{model_name.lower().replace(' ', '_')}_seed_{seed}.pt"
        train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            checkpoint_path=checkpoint,
            verbose=True,
        )

        # Evaluate on the unseen test fold (fold 10)
        _, test_macro, test_per_class, _, _ = evaluate_model(
            model, test_loader, criterion, device
        )

        macro_aurocs.append(test_macro)
        for cls in SUPERCLASSES:
            per_class_runs[cls].append(test_per_class[cls])

        print(f"Seed {seed} Test Macro AUROC: {test_macro:.4f}")
        for cls in SUPERCLASSES:
            print(f"  - {cls:5s}: {test_per_class[cls]:.4f}")

        if test_macro > best_seed_auroc:
            best_seed_auroc = test_macro
            best_seed_model = model

    results = {
        "model_name": model_name,
        "parameters": param_count,
        "seeds": seeds,
        "macro_aurocs": macro_aurocs,
        "macro_mean": float(np.mean(macro_aurocs)),
        "macro_std": float(np.std(macro_aurocs)),
        "per_class_means": {cls: float(np.mean(per_class_runs[cls])) for cls in SUPERCLASSES},
        "per_class_stds": {cls: float(np.std(per_class_runs[cls])) for cls in SUPERCLASSES},
        "best_model": best_seed_model,
    }

    return results


def build_comparison_table(results_list: List[Dict]) -> pd.DataFrame:
    """
    Creates a clean summary DataFrame displaying:
    Model, Parameters, Macro AUROC (mean ± std), and per-class AUROC (mean ± std).
    """
    rows = []
    for res in results_list:
        row = {
            "Model": res["model_name"],
            "Params": f"{res['parameters']:,}",
            "Macro AUROC": f"{res['macro_mean']:.4f} ± {res['macro_std']:.4f}",
        }
        for cls in SUPERCLASSES:
            row[cls] = f"{res['per_class_means'][cls]:.4f} ± {res['per_class_stds'][cls]:.4f}"
        rows.append(row)

    df_results = pd.DataFrame(rows)
    return df_results


def plot_lead_attention_weights(
    attn_weights: np.ndarray,
    targets: Optional[np.ndarray] = None,
    lead_names: List[str] = STANDARD_LEAD_NAMES,
    save_path: Optional[str] = "lead_attention_weights.png",
) -> plt.Figure:
    """
    Plots the average attention weight per lead across the test set.
    If targets are provided, also plots the distribution for specific conditions (e.g. MI vs NORM).
    """
    mean_weights = np.mean(attn_weights, axis=0)  # shape (12,)
    std_weights = np.std(attn_weights, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_indices = np.arange(len(lead_names))

    # Color code limb leads vs precordial chest leads
    colors = ["#2b5c8f"] * 6 + ["#e26d5c"] * 6

    bars = ax.bar(
        x_indices,
        mean_weights,
        yerr=std_weights,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )

    # Reference uniform baseline (1 / 12 = 0.0833)
    uniform_weight = 1.0 / len(lead_names)
    ax.axhline(
        uniform_weight,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"Uniform Weight ({uniform_weight:.3f})",
    )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(lead_names, fontsize=11, fontweight="bold")
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_xlabel("ECG Leads (Blue: Limb Leads, Orange: Precordial Leads)", fontsize=11)
    ax.set_title(
        "Learned Lead Attention Weights on PTB-XL Test Fold (Mean ± Std)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True)

    # Add numeric labels above bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Attention plot saved to: {save_path}")

    return fig
