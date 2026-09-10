"""
Evaluate the trained UNet on the official (untouched) test set.

Usage:
    python evaluate_unet.py                     # all test slices (real deployment condition)
    python evaluate_unet.py --foreground-only     # for fair comparison against MedSAM
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config
from src.data.dataset import load_manifest
from src.losses import keep_largest_component, slice_metrics
from src.models.unet import VGG16UNet


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = load_manifest(config.TEST_PROCESSED_DIR)
    if args.foreground_only:
        manifest = manifest[manifest["has_foreground"]].reset_index(drop=True)

    model = VGG16UNet(pretrained=False).to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, config.UNET_CKPT_NAME)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    raw_records, clean_records = [], []
    with torch.no_grad():
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="evaluating"):
            img = cv2.imread(os.path.join(config.TEST_PROCESSED_DIR, "images", row["filename"]), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(os.path.join(config.TEST_PROCESSED_DIR, "masks", row["filename"]), cv2.IMREAD_GRAYSCALE)
            gt_bin = (gt > 127).astype(np.uint8)

            img_t = (torch.from_numpy(img.astype(np.float32) / 255.0)
                     .unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device))
            logits = model(img_t)
            pred_raw = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
            pred_clean = keep_largest_component(pred_raw)

            raw_records.append(slice_metrics(pred_raw, gt_bin))
            clean_records.append(slice_metrics(pred_clean, gt_bin))

    raw_df = pd.DataFrame(raw_records)
    clean_df = pd.DataFrame(clean_records)
    summary = pd.DataFrame({
        "Raw prediction": raw_df.mean(),
        "Largest-component only": clean_df.mean(),
    }).round(4)

    print(f"\nEvaluated on {len(manifest)} test slices, {manifest['patient'].nunique()} patients "
          f"({'foreground-only' if args.foreground_only else 'all slices'}):\n")
    print(summary)

    out_path = os.path.join(config.RESULTS_DIR, "unet_test_metrics.csv")
    summary.to_csv(out_path)
    print(f"\nSaved metrics to {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the UNet on the test set.")
    parser.add_argument("--foreground-only", action="store_true",
                         help="Restrict to foreground slices (for a fair comparison against MedSAM).")
    evaluate(parser.parse_args())
