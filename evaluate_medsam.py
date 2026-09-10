"""
Evaluate MedSAM (zero-shot or LoRA-fine-tuned) on the official test set.

Uses an oracle box prompt derived from the ground-truth mask for every
prediction. This makes the comparison against the UNet (which gets no such
hint) informative but NOT fair — see README before reporting these numbers
as a strict improvement over the UNet.

Usage:
    python evaluate_medsam.py --mode zero_shot
    python evaluate_medsam.py --mode lora --r 8
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from src import config
from src.data.dataset import load_manifest, mask_to_box
from src.losses import keep_largest_component, slice_metrics
from src.models.lora import inject_lora


@torch.no_grad()
def evaluate_medsam(model, manifest_df, processed_root, processor, device, desc):
    records = []
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc=desc):
        img_gray = cv2.imread(os.path.join(processed_root, "images", row["filename"]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(processed_root, "masks", row["filename"]), cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        gt_bin = (gt > 127).astype(np.uint8)
        box = mask_to_box(gt)

        inputs = processor(img_rgb, input_boxes=[[box]], return_tensors="pt").to(device)
        out = model(**inputs, multimask_output=False)
        # IMPORTANT: pass raw logits, not sigmoid()'d output. post_process_masks
        # binarizes with `mask > mask_threshold` (default 0.0) — a logit-space
        # threshold. Pre-applying sigmoid makes every value strictly positive,
        # silently "predicting" the entire image as foreground. Cost this
        # project a debugging session; see README.
        pred = processor.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
            binarize=True,
        )[0][0, 0].numpy().astype(np.uint8)
        pred_clean = keep_largest_component(pred)

        records.append(slice_metrics(pred_clean, gt_bin))
    return pd.DataFrame(records)


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = load_manifest(config.TEST_PROCESSED_DIR)
    fg_manifest = manifest[manifest["has_foreground"]].reset_index(drop=True)

    processor = SamProcessor.from_pretrained(config.MEDSAM_MODEL_ID)
    model = SamModel.from_pretrained(config.MEDSAM_MODEL_ID).to(device)

    if args.mode == "lora":
        model = inject_lora(model, r=args.r, alpha=args.alpha)
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"medsam_lora_r{args.r}_best.pth")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)

    model.eval()
    df = evaluate_medsam(model, fg_manifest, config.TEST_PROCESSED_DIR, processor, device, args.mode)

    print(f"\n{args.mode} — {len(fg_manifest)} foreground test slices, "
          f"{fg_manifest['patient'].nunique()} patients:\n")
    summary = df[["accuracy", "dice", "iou", "precision", "recall"]].mean().round(4)
    print(summary)

    out_path = os.path.join(config.RESULTS_DIR, f"medsam_{args.mode}_test_metrics.csv")
    summary.to_csv(out_path)
    print(f"\nSaved metrics to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MedSAM on the test set.")
    parser.add_argument("--mode", choices=["zero_shot", "lora"], default="lora")
    parser.add_argument("--r", type=int, default=config.LORA_RANK)
    parser.add_argument("--alpha", type=int, default=config.LORA_ALPHA)
    evaluate(parser.parse_args())
