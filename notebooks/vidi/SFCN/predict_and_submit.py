#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_and_submit.py — Generate submission CSV for crowd counting models.

This script loads a trained SFCN model checkpoint, iterates over the list of
`image_id`s in a sample submission CSV, reads the corresponding test
images, preprocesses them (letterbox, normalise), feeds them through the
model to obtain density maps, integrates the density maps to get a count,
and writes the results to a new submission CSV. If an image is missing,
the predicted count is set to 0.

Usage example:

```bash
python predict_and_submit.py --test_dir data/test/images --sample_csv sample_submission.csv --ckpt sfcn_best.pth --out_csv submission.csv --img_size 768 --batch 4 --amp
```
"""

import os
import csv
import argparse
from glob import glob

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms

# Import the SFCN model definition from the training script. We assume
# train_sfcn_fixed.py is in the same directory as this script.
try:
    from train_sfcn import SFCN_VGG
    from train_sfcn import IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    raise ImportError(
        "Unable to import SFCN_VGG and mean/std from train_sfcn_fixed.py."
    )


def letterbox_image(img: np.ndarray, target: int) -> np.ndarray:
    """Resize and pad an image to a square canvas without distortion."""
    h, w = img.shape[:2]
    scale = min(target / h, target / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (target - nh) // 2
    left = (target - nw) // 2
    canvas = np.zeros((target, target, 3), dtype=img_rs.dtype)
    canvas[top : top + nh, left : left + nw] = img_rs
    return canvas


def find_test_image(test_dir: str, image_id: str) -> str:
    """Find a test image file matching the given image_id.

    It searches for files with common image extensions (jpg, jpeg, png,
    bmp, webp) whose basename matches the image_id. Returns the first match,
    or an empty string if none is found.
    """
    # First, check if image_id already includes extension
    direct_path = os.path.join(test_dir, image_id)
    if os.path.exists(direct_path):
        return direct_path
    
    # If not, try adding common extensions
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    # Try exact match with extensions
    for ext in exts:
        candidate = os.path.join(test_dir, f"{image_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    # Fallback: search by prefix
    for ext in exts:
        matches = glob(os.path.join(test_dir, f"{image_id}*{ext}"))
        if matches:
            return matches[0]
    return ""


def preprocess_image(path: str, img_size: int) -> torch.Tensor:
    """Load and preprocess a test image for the model.

    The image is read with OpenCV, letterboxed to a square of size `img_size`,
    converted to a tensor, and normalised with ImageNet mean/std. The
    returned tensor has shape (3, H, W) and dtype float32.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pad = letterbox_image(img, target=img_size)
    # Convert to tensor and normalise
    to_tensor = transforms.ToTensor()
    t = to_tensor(img_pad)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    t = normalize(t)
    return t


@torch.no_grad()
def predict_counts(
    model: torch.nn.Module,
    image_tensors: torch.Tensor,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """Run inference on a batch of preprocessed images and return counts.

    Args:
        model: The loaded SFCN model.
        image_tensors: A tensor of shape (batch, 3, H, W).
        device: CUDA or CPU device where the model is located.
        use_amp: If True, run inference under autocast to save memory.

    Returns:
        Tensor of predicted counts (shape (batch,)).
    """
    image_tensors = image_tensors.to(device)
    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            density = model(image_tensors)
    else:
        density = model(image_tensors)
    # Sum over spatial dimensions to get counts
    counts = density.sum(dim=(1, 2, 3))
    return counts.cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument(
        "--sample_csv", type=str, required=True, help="Path to the sample submission CSV"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--out_csv", type=str, required=True, help="Path where the output submission CSV will be saved"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=768,
        help="Size to which test images will be letterboxed (should match the training base_size)",
    )
    parser.add_argument(
        "--batch", type=int, default=4, help="Batch size for inference (>=1)"
    )
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision for inference")
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    # Instantiate model without pretrained weights (we'll load from checkpoint)
    model = SFCN_VGG(pretrained=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Read sample submission CSV to get the order of image_ids
    with open(args.sample_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != ["image_id", "predicted_count"]:
            raise ValueError(
                f"Sample CSV header expected ['image_id','predicted_count'], got {header}"
            )
        entries = [row for row in reader]

    image_ids = [row[0] for row in entries]
    # Prepare the output list
    results = []
    # Process in batches
    batch_size = max(1, args.batch)
    current_batch_ids: list[str] = []
    current_batch_tensors: list[torch.Tensor] = []
    for idx, img_id in enumerate(image_ids):
        img_path = find_test_image(args.test_dir, img_id)
        if not img_path:
            # If no image found, append zero count immediately
            print(f"[WARN] Test image for id {img_id} not found. Using 0.")
            results.append((img_id, 0))
            continue
        # Preprocess the image
        try:
            tensor = preprocess_image(img_path, args.img_size)
        except Exception as e:
            print(f"[WARN] Error processing image {img_path}: {e}. Using 0.")
            results.append((img_id, 0))
            continue
        current_batch_ids.append(img_id)
        current_batch_tensors.append(tensor)
        # When batch is full or last item, run inference
        if len(current_batch_tensors) == batch_size or idx == len(image_ids) - 1:
            # Stack tensors into a batch
            batch_tensor = torch.stack(current_batch_tensors, dim=0)
            counts = predict_counts(model, batch_tensor, device, use_amp=args.amp)
            for bid, count in zip(current_batch_ids, counts):
                # Round to nearest int and clamp to non-negative
                pred_count = max(0, int(round(float(count))))
                results.append((bid, pred_count))
            # Reset batch lists
            current_batch_ids = []
            current_batch_tensors = []

    # Write out CSV in the same order as sample
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "predicted_count"])
        for image_id, pred_count in results:
            writer.writerow([image_id, pred_count])
    print(f"Submission saved to {args.out_csv}")


if __name__ == "__main__":
    main()