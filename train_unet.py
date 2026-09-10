"""
Train the VGG16-UNet baseline (Phase 1).

Usage:
    python train_unet.py --epochs 30 --batch-size 4 --lr 1e-4
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import ProstateDataset, load_manifest
from src.losses import DiceLoss, dice_coefficient, iou_score
from src.models.unet import VGG16UNet
from src.utils import save_prediction_gallery, save_training_curves, set_seed


def train(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    manifest = load_manifest(config.PROCESSED_DIR)
    train_ds = ProstateDataset(manifest, config.PROCESSED_DIR, "train", augment=True)
    val_ds = ProstateDataset(manifest, config.PROCESSED_DIR, "val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)} slices, Val: {len(val_ds)} slices")

    model = VGG16UNet(pretrained=True, freeze_encoder=True).to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, config.UNET_CKPT_NAME)
    sample_batch = None

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_dices, val_ious = [], [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_losses.append(loss.item())
                val_dices.append(dice_coefficient(logits, masks))
                val_ious.append(iou_score(logits, masks))
                if sample_batch is None:
                    sample_batch = (imgs.cpu(), masks.cpu(), torch.sigmoid(logits).cpu())

        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        val_dice, val_iou = float(np.mean(val_dices)), float(np.mean(val_ious))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_dice={val_dice:.4f} val_iou={val_iou:.4f} "
              f"{'(best)' if improved else f'(no improve x{epochs_no_improve})'}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    save_training_curves(history, os.path.join(config.RESULTS_DIR, "unet_training_curves.png"), "UNet training")
    if sample_batch is not None:
        imgs_s, masks_s, preds_s = sample_batch
        n_show = min(4, imgs_s.size(0))
        save_prediction_gallery(
            [imgs_s[i, 0] for i in range(n_show)],
            [masks_s[i, 0] for i in range(n_show)],
            [(preds_s[i, 0] > 0.5).float() for i in range(n_show)],
            os.path.join(config.RESULTS_DIR, "unet_predictions.png"),
        )
    print(f"\nBest val Dice: {max(history['val_dice']):.4f}. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the VGG16-UNet baseline.")
    parser.add_argument("--epochs", type=int, default=config.UNET_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.UNET_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.UNET_LR)
    parser.add_argument("--patience", type=int, default=config.UNET_PATIENCE)
    train(parser.parse_args())
