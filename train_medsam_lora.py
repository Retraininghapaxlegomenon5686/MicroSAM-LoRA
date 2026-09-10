"""
Train MedSAM with LoRA (Phase 2), or a decoder-only cached baseline ablation.

An image-embedding cache is only valid for parameters that never change
during training. Once LoRA is injected into the vision encoder, its output
changes every step, so caching before training and updating LoRA weights
after are mutually exclusive claims (see README). The two modes below are
kept strictly separate for that reason:

Usage:
    python train_medsam_lora.py --mode lora --r 8 --epochs 15
    python train_medsam_lora.py --mode decoder_only --epochs 8
    python train_medsam_lora.py --mode lora --resume    # continue from the latest checkpoint
"""
import argparse
import functools
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from src import config
from src.data.dataset import MedSAMBoxDataset, load_manifest, medsam_collate
from src.losses import DiceBCELoss, dice_iou
from src.models.lora import count_trainable_params, prepare_medsam_lora
from src.utils import save_training_curves, set_seed


def build_cache(dataset, model, processor, device, desc="caching"):
    """Run every image through the frozen vision encoder once and cache the
    resulting embedding. Valid ONLY when the encoder is not being trained."""
    cache = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=desc):
            img_rgb, box, mask = dataset[i]
            inputs = processor(img_rgb, input_boxes=[[box]], return_tensors="pt").to(device)
            emb = model.get_image_embeddings(inputs["pixel_values"])
            mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            cache.append({
                "embedding": emb.squeeze(0).cpu(),
                "input_boxes": inputs["input_boxes"][0].cpu(),
                "mask_256": torch.from_numpy(mask_256).float(),
            })
    return cache


def train_decoder_only(args, device):
    processor = SamProcessor.from_pretrained(config.MEDSAM_MODEL_ID)
    manifest = load_manifest(config.PROCESSED_DIR)
    train_ds = MedSAMBoxDataset(manifest, config.PROCESSED_DIR, "train", augment=False)
    val_ds = MedSAMBoxDataset(manifest, config.PROCESSED_DIR, "val", augment=False)

    model = SamModel.from_pretrained(config.MEDSAM_MODEL_ID).to(device)

    os.makedirs(config.EMBEDDING_CACHE_DIR, exist_ok=True)
    train_cache_path = os.path.join(config.EMBEDDING_CACHE_DIR, "train_cache.pt")
    val_cache_path = os.path.join(config.EMBEDDING_CACHE_DIR, "val_cache.pt")
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        print("Found cached embeddings on disk — loading instead of recomputing.")
        train_cache = torch.load(train_cache_path)
        val_cache = torch.load(val_cache_path)
    else:
        train_cache = build_cache(train_ds, model, processor, device, "caching train")
        val_cache = build_cache(val_ds, model, processor, device, "caching val")
        torch.save(train_cache, train_cache_path)
        torch.save(val_cache, val_cache_path)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.mask_decoder.parameters(), lr=args.lr)
    criterion = DiceBCELoss()
    history = {"train_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(args.epochs):
        model.train()
        idxs = list(range(len(train_cache)))
        random.shuffle(idxs)
        train_losses = []
        for start in range(0, len(idxs), args.batch_size):
            batch_idxs = idxs[start:start + args.batch_size]
            embs = torch.stack([train_cache[i]["embedding"] for i in batch_idxs]).to(device)
            boxes = torch.stack([train_cache[i]["input_boxes"] for i in batch_idxs]).to(device)
            targets = torch.stack([train_cache[i]["mask_256"] for i in batch_idxs]).to(device)

            optimizer.zero_grad()
            out = model(image_embeddings=embs, input_boxes=boxes, multimask_output=False)
            loss = criterion(out.pred_masks[:, 0, 0], targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_dices, val_ious = [], []
        with torch.no_grad():
            for start in range(0, len(val_cache), args.batch_size):
                chunk = val_cache[start:start + args.batch_size]
                embs = torch.stack([c["embedding"] for c in chunk]).to(device)
                boxes = torch.stack([c["input_boxes"] for c in chunk]).to(device)
                targets = torch.stack([c["mask_256"] for c in chunk]).to(device)
                out = model(image_embeddings=embs, input_boxes=boxes, multimask_output=False)
                d, i = dice_iou(out.pred_masks[:, 0, 0], targets)
                val_dices.append(d); val_ious.append(i)

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_dice"].append(float(np.mean(val_dices)))
        history["val_iou"].append(float(np.mean(val_ious)))
        print(f"Epoch {epoch+1}/{args.epochs}: loss={history['train_loss'][-1]:.4f} "
              f"val_dice={history['val_dice'][-1]:.4f} val_iou={history['val_iou'][-1]:.4f}")

    save_training_curves(history, os.path.join(config.RESULTS_DIR, "decoder_only_curves.png"),
                          "Decoder-only (cached, valid — encoder frozen)")
    print(f"\nBest val Dice (256-res, approximate): {max(history['val_dice']):.4f}")


def train_lora(args, device):
    processor = SamProcessor.from_pretrained(config.MEDSAM_MODEL_ID)
    manifest = load_manifest(config.PROCESSED_DIR)
    train_ds = MedSAMBoxDataset(manifest, config.PROCESSED_DIR, "train", augment=True)
    val_ds = MedSAMBoxDataset(manifest, config.PROCESSED_DIR, "val", augment=False)
    collate = functools.partial(medsam_collate, processor=processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    model = SamModel.from_pretrained(config.MEDSAM_MODEL_ID).to(device)
    model = prepare_medsam_lora(model, r=args.r, alpha=args.alpha)
    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    criterion = DiceBCELoss()

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_val_dice = -1
    epochs_no_improve = 0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"medsam_lora_r{args.r}_best.pth")
    latest_ckpt_path = ckpt_path.replace(".pth", "_latest.pth")
    start_epoch = 0

    if args.resume and os.path.exists(latest_ckpt_path):
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt["history"]
        best_val_dice = ckpt["best_val_dice"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch} (best val dice so far: {best_val_dice:.4f}).")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_losses = []
        optimizer.zero_grad()
        n_batches = len(train_loader)
        for step, (inputs, targets) in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 dtype=torch.float16, enabled=(device == "cuda")):
                out = model(pixel_values=inputs["pixel_values"], input_boxes=inputs["input_boxes"],
                            multimask_output=False)
                loss = criterion(out.pred_masks[:, 0, 0], targets) / args.grad_accum_steps
            scaler.scale(loss).backward()
            train_losses.append(loss.item() * args.grad_accum_steps)

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == n_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        model.eval()
        val_losses, val_dices, val_ious = [], [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                targets = targets.to(device)
                out = model(pixel_values=inputs["pixel_values"], input_boxes=inputs["input_boxes"],
                            multimask_output=False)
                logits = out.pred_masks[:, 0, 0]
                val_losses.append(criterion(logits, targets).item())
                d, i = dice_iou(logits, targets)
                val_dices.append(d); val_ious.append(i)

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))
        history["val_dice"].append(float(np.mean(val_dices)))
        history["val_iou"].append(float(np.mean(val_ious)))

        improved = history["val_dice"][-1] > best_val_dice
        if improved:
            best_val_dice = history["val_dice"][-1]
            epochs_no_improve = 0
            torch.save({k: v for k, v in model.state_dict().items()
                        if "lora_" in k or "mask_decoder" in k}, ckpt_path)
        else:
            epochs_no_improve += 1

        # Full resumable checkpoint saved EVERY epoch — lets a time-boxed
        # training run (e.g. a limited Colab session) stop and continue later.
        torch.save({
            "epoch": epoch,
            "model_state": {k: v for k, v in model.state_dict().items()
                             if "lora_" in k or "mask_decoder" in k},
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_val_dice": best_val_dice,
        }, latest_ckpt_path)

        print(f"Epoch {epoch+1}/{args.epochs}: dice={history['val_dice'][-1]:.4f} "
              f"iou={history['val_iou'][-1]:.4f} "
              f"{'(best)' if improved else f'(no improve x{epochs_no_improve})'}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    save_training_curves(history, os.path.join(config.RESULTS_DIR, "lora_training_curves.png"),
                          f"MedSAM + LoRA (r={args.r})")
    print(f"\nBest val Dice (256-res, approximate): {max(history['val_dice']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedSAM with LoRA, or a decoder-only cached baseline.")
    parser.add_argument("--mode", choices=["lora", "decoder_only"], default="lora")
    parser.add_argument("--r", type=int, default=config.LORA_RANK)
    parser.add_argument("--alpha", type=int, default=config.LORA_ALPHA)
    parser.add_argument("--epochs", type=int, default=config.LORA_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.LORA_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=config.LORA_GRAD_ACCUM_STEPS)
    parser.add_argument("--lr", type=float, default=config.LORA_LR)
    parser.add_argument("--patience", type=int, default=config.LORA_PATIENCE)
    parser.add_argument("--resume", action="store_true", help="Resume LoRA training from the latest checkpoint.")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if args.mode == "decoder_only":
        train_decoder_only(args, device)
    else:
        train_lora(args, device)
