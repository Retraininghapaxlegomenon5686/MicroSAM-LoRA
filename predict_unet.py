"""
Run the trained UNet on a single image and save the predicted mask.

Usage:
    python predict_unet.py --image path/to/slice.png --output mask.png
"""
import argparse
import os

import cv2
import numpy as np
import torch

from src import config
from src.losses import keep_largest_component
from src.models.unet import VGG16UNet


def predict(image_path, output_path, checkpoint=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = checkpoint or os.path.join(config.CHECKPOINT_DIR, config.UNET_CKPT_NAME)

    model = VGG16UNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img_resized = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img_t = (torch.from_numpy(img_resized.astype(np.float32) / 255.0)
             .unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device))

    with torch.no_grad():
        logits = model(img_t)
        pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
    pred_clean = keep_largest_component(pred) * 255

    cv2.imwrite(output_path, pred_clean)
    print(f"Saved predicted mask to {output_path}")
    return pred_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trained UNet on a single image.")
    parser.add_argument("--image", required=True, help="Path to a grayscale ultrasound slice (PNG/JPG).")
    parser.add_argument("--output", default="prediction.png")
    parser.add_argument("--checkpoint", default=None, help="Override the default checkpoint path.")
    args = parser.parse_args()
    predict(args.image, args.output, args.checkpoint)
