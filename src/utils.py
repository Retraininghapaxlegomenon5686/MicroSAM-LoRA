"""Shared utility helpers: reproducibility and file-based visualization.

Scripts run headless (no Jupyter inline display), so plots are saved to PNG
files under RESULTS_DIR rather than shown inline.
"""
import os
import random

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_training_curves(history, out_path, title=""):
    """Save loss/dice/iou curves to a PNG."""
    keys = [k for k in ("train_loss", "val_loss", "val_dice", "val_iou") if k in history]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        ax.plot(history[key])
        ax.set_title(key)
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved training curves to {out_path}")


def save_prediction_gallery(images, ground_truths, predictions, out_path,
                             titles=("Image", "Ground truth", "Prediction")):
    """Save a 3-row gallery (image / ground truth / prediction) to a PNG."""
    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 9))
    if n == 1:
        axes = axes.reshape(3, 1)
    for i in range(n):
        axes[0, i].imshow(images[i], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(ground_truths[i], cmap="gray"); axes[1, i].axis("off")
        axes[2, i].imshow(predictions[i], cmap="gray"); axes[2, i].axis("off")
    for row, title in enumerate(titles):
        axes[row, 0].set_title(title, loc="left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved prediction gallery to {out_path}")
