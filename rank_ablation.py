"""
LoRA rank ablation — trains + evaluates MedSAM+LoRA at multiple ranks and
collects everything into one comparison table.

Reuses train_lora() and evaluate_medsam() directly (no logic duplicated),
so any bug fix made there automatically applies here too.

Designed for a multi-hour Kaggle run that might not finish in one sitting:
- A rank already fully evaluated (results/rank_r{r}_test_metrics.csv exists)
  is skipped entirely on rerun.
- A rank that was mid-training when the session ended resumes from its own
  latest checkpoint (same mechanism as train_medsam_lora.py --resume).
- Just re-run the same command after restarting the session/kernel.

Usage:
    python rank_ablation.py --ranks 4 8 16 32 --epochs 15
    python rank_ablation.py --ranks 4 8 16 32 --force    # ignore existing results, redo everything
"""
import argparse
import os
from types import SimpleNamespace

import pandas as pd
import torch
from transformers import SamModel, SamProcessor

from src import config
from src.data.dataset import load_manifest
from src.models.lora import count_trainable_params, inject_lora
from src.utils import set_seed
from evaluate_medsam import evaluate_medsam
from train_medsam_lora import train_lora


def run_one_rank(r, args, device):
    print(f"\n{'=' * 60}\nLoRA rank r={r}\n{'=' * 60}")

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"medsam_lora_r{r}_best.pth")
    metrics_path = os.path.join(config.RESULTS_DIR, f"rank_r{r}_test_metrics.csv")

    if os.path.exists(metrics_path) and not args.force:
        print(f"Rank {r} already evaluated ({metrics_path} exists) — skipping. Use --force to redo.")
        return pd.read_csv(metrics_path, index_col=0).iloc[:, 0]

    # --- Train (or resume a partially-trained run) ---
    # Always call train_lora with resume=True (unless --force) and let its own
    # epoch-aware resume logic decide whether there's anything left to do.
    # Checking `os.path.exists(ckpt_path)` here would be wrong: best_val_dice
    # starts at -1, so a checkpoint exists after just the FIRST completed
    # epoch — treating that as "fully trained" would silently truncate a
    # 15-epoch run to 1 epoch after any interruption. If training already
    # finished, train_lora's `range(start_epoch, epochs)` loop simply does
    # nothing and returns immediately — cheap and safe to call either way.
    if args.force or not os.path.exists(metrics_path):
        train_args = SimpleNamespace(
            r=r, alpha=args.alpha, epochs=args.epochs,
            batch_size=config.LORA_BATCH_SIZE, grad_accum_steps=config.LORA_GRAD_ACCUM_STEPS,
            lr=args.lr, patience=args.patience, resume=(not args.force),
        )
        train_lora(train_args, device)

    # --- Evaluate the finished checkpoint on the held-out test set ---
    processor = SamProcessor.from_pretrained(config.MEDSAM_MODEL_ID)
    model = SamModel.from_pretrained(config.MEDSAM_MODEL_ID).to(device)
    model = inject_lora(model, r=r, alpha=args.alpha)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    trainable, total = count_trainable_params(model)

    manifest = load_manifest(config.TEST_PROCESSED_DIR)
    fg_manifest = manifest[manifest["has_foreground"]].reset_index(drop=True)
    df = evaluate_medsam(model, fg_manifest, config.TEST_PROCESSED_DIR, processor, device, f"r={r} test eval")

    result = df[["accuracy", "dice", "iou", "precision", "recall"]].mean()
    result["trainable_params"] = trainable
    result["total_params"] = total
    result["trainable_pct"] = round(100 * trainable / total, 2)

    result.to_csv(metrics_path)
    print(f"\nr={r} test results:\n{result}")

    # Free GPU memory before the next rank's model gets loaded — same lesson
    # as the base_model/lora_model OOM from Phase 2 training (see README).
    del model
    torch.cuda.empty_cache()
    return result


def main(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print(f"Running ranks: {args.ranks}  (alpha={args.alpha} fixed across all ranks — "
          f"scale=alpha/r shrinks as rank grows, a deliberate, stated choice, not an oversight)")

    results = {r: run_one_rank(r, args, device) for r in args.ranks}

    summary = pd.DataFrame(results).T
    summary.index.name = "rank"
    summary = summary[["trainable_params", "trainable_pct", "dice", "iou", "precision", "recall", "accuracy"]]

    summary_path = os.path.join(config.RESULTS_DIR, "rank_ablation.csv")
    summary.to_csv(summary_path)

    print(f"\n{'=' * 60}\nRank ablation complete\n{'=' * 60}")
    print(summary.round(4))
    print(f"\nSaved full sweep to {summary_path}")
    print("\nMarkdown table, ready to paste into the README's LoRA Rank Ablation section:\n")
    try:
        print(summary.round(4).to_markdown())
    except ImportError:
        print("(install `tabulate` for a rendered markdown table: pip install tabulate)")
        print(summary.round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA rank ablation across multiple ranks.")
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--alpha", type=int, default=config.LORA_ALPHA)
    parser.add_argument("--epochs", type=int, default=config.LORA_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LORA_LR)
    parser.add_argument("--patience", type=int, default=config.LORA_PATIENCE)
    parser.add_argument("--force", action="store_true",
                         help="Retrain/re-evaluate every rank even if results already exist.")
    main(parser.parse_args())
