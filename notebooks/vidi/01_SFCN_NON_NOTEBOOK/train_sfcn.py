#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_sfcn_fixed.py — Training script for SFCN crowd counting with several
improvements:

* Supports optional patch training with the ability to avoid sampling
  completely empty patches (patches with no annotated people). When
  `avoid_empty_patches=True`, random crops are retried a few times until
  at least one point lies inside the patch, falling back to a final
  random crop after a fixed number of attempts.
* Adds an auxiliary count loss on the predicted density map to encourage
  the network to learn the global count faster. The strength of this loss
  is controlled by `count_loss_alpha` (default 0.0 disables it).
* Randomises the train/validation split to avoid ordering biases.
* Normalises inputs with ImageNet statistics and warns if the density
  integral deviates substantially from the number of points.

Usage example:

```bash
python train_sfcn_fixed.py \
  --img_dir data/train/images \
  --lbl_dir data/train/labels \
  --base_size 768 \
  --patch_size 384 \
  --patches_per_image 4 \
  --avoid_empty_patches \
  --count_loss_alpha 0.1 \
  --batch 4 \
  --epochs 120 \
  --amp
```

Set `--patch_size 0` if you want to train on full letterboxed images
initially; you can resume with patch training later by loading the
checkpoint and specifying `--patch_size > 0` on a subsequent run.
"""

import os
import re
import json
import math
import random
import argparse
import warnings
from glob import glob
from typing import List, Tuple

import numpy as np
import cv2
from scipy.spatial import KDTree

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ImageNet mean and std for normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 1337) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imread_rgb(path: str) -> np.ndarray:
    """Read an RGB image using OpenCV and convert BGR→RGB."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def letterbox(img: np.ndarray, target: int = 512) -> Tuple[np.ndarray, float, int, int]:
    """Resize and pad an image to a square canvas without distortion.

    Returns the padded image, the scale factor used, and the left/top
    padding applied. The output size is (target, target).
    """
    h, w = img.shape[:2]
    scale = min(target / h, target / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (target - nh) // 2
    left = (target - nw) // 2
    canvas = np.zeros((target, target, 3), dtype=img_rs.dtype)
    canvas[top:top + nh, left:left + nw] = img_rs
    return canvas, scale, left, top


def parse_points_from_json(path: str) -> Tuple[np.ndarray, int]:
    """Parse annotated points from a JSON file.

    Supports several common crowd counting annotation formats. Returns an
    array of shape (N, 2) containing [x, y] coordinates and, if present
    in the JSON, the declared number of people. If no `human_num` or
    `num_human` field is present the count is returned as None.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pts: List[List[float]]
    num = None
    if isinstance(obj, dict) and "points" in obj:
        pts = obj["points"]
        # unify possible keys for ground-truth count
        num = obj.get("human_num", obj.get("num_human", None))
        # If points are dicts, extract x/y fields
        if len(pts) > 0 and isinstance(pts[0], dict):
            pts = [[p["x"], p["y"]] for p in pts if "x" in p and "y" in p]
    elif isinstance(obj, dict) and "annotations" in obj:
        pts = [[a["x"], a["y"]] for a in obj["annotations"] if "x" in a and "y" in a]
        num = obj.get("human_num", obj.get("num_human", None))
    elif isinstance(obj, list):
        # direct list of [x,y]
        pts = obj
    else:
        raise ValueError(f"Unknown JSON schema: {path}")
    pts_arr = np.array(pts, dtype=np.float32) if len(pts) > 0 else np.zeros((0, 2), np.float32)
    return pts_arr, num


def derive_json_path(lbl_dir: str, img_path: str) -> str:
    """Derive the corresponding JSON label path for a given image path.

    Tries to match the image basename to a JSON file in the label directory.
    Falls back to matching digits if an exact name isn't found.
    """
    name = os.path.splitext(os.path.basename(img_path))[0]
    cand = os.path.join(lbl_dir, name + ".json")
    if os.path.exists(cand):
        return cand
    # Try matching trailing digits
    m = re.findall(r"\d+", name)
    if m:
        alt = os.path.join(lbl_dir, f"{m[-1]}.json")
        if os.path.exists(alt):
            return alt
    # Fallback: any file starting with the same name
    lst = glob(os.path.join(lbl_dir, f"{name}*.json"))
    if lst:
        return lst[0]
    raise FileNotFoundError(f"JSON label not found for {img_path}")


def make_density_map(
    points_xy: np.ndarray,
    grid_size: int,
    down: int = 8,
    sigma_mode: str = "adaptive",
    knn: int = 3,
    beta: float = 0.3,
    const_sigma: float = 2.0,
) -> np.ndarray:
    """Generate a density map on a grid given annotated points.

    The density map is of shape (grid_size//down, grid_size//down). Each
    point is represented by a Gaussian whose sigma is either constant or
    computed from the k-nearest neighbours. The integral of the density map
    approximates the number of points.
    """
    target = grid_size
    dh, dw = target // down, target // down
    den = np.zeros((dh, dw), dtype=np.float32)
    if len(points_xy) == 0:
        return den
    # Scale points to the density map resolution
    pts = points_xy.copy()
    pts[:, 0] = pts[:, 0] * (dw / target)
    pts[:, 1] = pts[:, 1] * (dh / target)
    tree = KDTree(pts) if len(pts) > 1 else None
    for (x, y) in pts:
        # Determine sigma
        if sigma_mode == "adaptive" and tree is not None and len(pts) > 3:
            dists, _ = tree.query([x, y], k=min(knn + 1, len(pts)))
            sigma = max(1.0, float(np.mean(dists[1:])) * beta)
        else:
            sigma = const_sigma
        cx, cy = float(x), float(y)
        rad = int(max(1, math.ceil(3 * sigma)))
        x0, x1 = max(0, int(math.floor(cx - rad))), min(dw, int(math.ceil(cx + rad + 1)))
        y0, y1 = max(0, int(math.floor(cy - rad))), min(dh, int(math.ceil(cy + rad + 1)))
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1) - cx
        ys = np.arange(y0, y1) - cy
        xx, yy = np.meshgrid(xs, ys)
        g = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
        s = g.sum()
        if s > 0:
            den[y0:y1, x0:x1] += (g / s).astype(np.float32)
    return den


class CrowdDataset(Dataset):
    """Custom dataset for crowd counting.

    Supports optional random patch cropping on training data. When
    `avoid_empty_patches` is set, the dataset will attempt to choose a
    patch containing at least one point, falling back to a random crop if
    it fails after a number of retries.
    """

    def __init__(
        self,
        img_dir: str,
        lbl_dir: str,
        base_size: int = 768,
        down: int = 8,
        aug: bool = True,
        mode: str = "train",
        patch_size: int = 0,
        patches_per_image: int = 1,
        sigma_mode: str = "adaptive",
        avoid_empty_patches: bool = False,
    ) -> None:
        super().__init__()
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.*")))
        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")
        self.lbl_dir = lbl_dir
        self.base_size = base_size
        self.down = down
        self.aug = aug
        self.mode = mode
        self.patch_size = patch_size
        self.patches_per_image = max(1, int(patches_per_image))
        self.sigma_mode = sigma_mode
        self.avoid_empty_patches = avoid_empty_patches

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.color_jit = transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)

        # Compute effective length for patch training
        if self.mode == "train" and self.patch_size > 0 and self.patches_per_image > 1:
            self.effective_len = len(self.img_paths) * self.patches_per_image
        else:
            self.effective_len = len(self.img_paths)

    def __len__(self) -> int:
        return self.effective_len

    def _load_img_pts(self, idx_base: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load an image and its transformed annotation points."""
        pimg = self.img_paths[idx_base]
        img = imread_rgb(pimg)
        h, w = img.shape[:2]
        plbl = derive_json_path(self.lbl_dir, pimg)
        pts, _ = parse_points_from_json(plbl)
        # Augment: horizontal flip
        if self.mode == "train" and self.aug and random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            if len(pts) > 0:
                pts = pts.copy()
                pts[:, 0] = (w - 1) - pts[:, 0]
        # Augment: colour jitter
        if self.mode == "train" and self.aug and random.random() < 0.5:
            pil = transforms.ToPILImage()(img)
            pil = self.color_jit(pil)
            img = np.array(pil)
        # Letterbox to base size
        canvas, scale, left, top = letterbox(img, target=self.base_size)
        if len(pts) > 0:
            pts_tr = pts.copy()
            pts_tr[:, 0] = pts_tr[:, 0] * scale + left
            pts_tr[:, 1] = pts_tr[:, 1] * scale + top
            # Clamp to canvas bounds
            m = (
                (pts_tr[:, 0] >= 0)
                & (pts_tr[:, 0] < self.base_size)
                & (pts_tr[:, 1] >= 0)
                & (pts_tr[:, 1] < self.base_size)
            )
            pts_tr = pts_tr[m]
        else:
            pts_tr = np.zeros((0, 2), np.float32)
        return canvas, pts_tr

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Map global index to an image index (for patch training)
        if self.mode == "train" and self.patch_size > 0 and self.patches_per_image > 1:
            idx_base = index // self.patches_per_image
        else:
            idx_base = index
        idx_base %= len(self.img_paths)

        img_lb, pts_tr = self._load_img_pts(idx_base)

        # Optional patch cropping
        if self.mode == "train" and self.patch_size > 0:
            ps = self.patch_size
            if ps > self.base_size:
                raise ValueError("patch_size must be <= base_size")
            max_off = self.base_size - ps
            pts_out = np.zeros((0, 2), np.float32)
            # Attempt to find a non-empty patch when requested
            for attempt in range(10) if self.avoid_empty_patches else [0]:
                ox = 0 if max_off <= 0 else random.randint(0, max_off)
                oy = 0 if max_off <= 0 else random.randint(0, max_off)
                crop = img_lb[oy : oy + ps, ox : ox + ps, :]
                if len(pts_tr) > 0:
                    pts_c = pts_tr.copy()
                    pts_c[:, 0] -= ox
                    pts_c[:, 1] -= oy
                    m = (
                        (pts_c[:, 0] >= 0)
                        & (pts_c[:, 0] < ps)
                        & (pts_c[:, 1] >= 0)
                        & (pts_c[:, 1] < ps)
                    )
                    pts_c = pts_c[m]
                else:
                    pts_c = np.zeros((0, 2), np.float32)
                # If avoid_empty_patches is False, we break immediately (no retries)
                if not self.avoid_empty_patches or len(pts_c) > 0 or attempt == 9:
                    pts_out = pts_c
                    img_out = crop
                    break
            grid = ps
        else:
            img_out = img_lb
            pts_out = pts_tr
            grid = self.base_size

        # Build density map
        den = make_density_map(
            pts_out,
            grid_size=grid,
            down=self.down,
            sigma_mode=self.sigma_mode,
        )

        # Optional check: ensure density integrates to the number of points
        if self.mode != "train" or random.random() < 0.02:
            cnt = float(len(pts_out))
            s = float(den.sum())
            if abs(s - cnt) > 0.05:
                print(f"[warn] density sum {s:.3f} != count {cnt:.3f}")

        # Convert to tensors and normalise
        t = self.to_tensor(img_out)
        t = self.normalize(t)
        d = torch.from_numpy(den).unsqueeze(0)
        c = torch.tensor([float(len(pts_out))], dtype=torch.float32)
        return t, d, c


class SpatialEncoder(nn.Module):
    """Simple spatial encoder that propagates information in four directions."""

    def __init__(self, channels: int, k: int = 9) -> None:
        super().__init__()
        p = k // 2
        self.h1 = nn.Conv2d(channels, channels, (1, k), padding=(0, p), groups=channels, bias=False)
        self.h2 = nn.Conv2d(channels, channels, (1, k), padding=(0, p), groups=channels, bias=False)
        self.v1 = nn.Conv2d(channels, channels, (k, 1), padding=(p, 0), groups=channels, bias=False)
        self.v2 = nn.Conv2d(channels, channels, (k, 1), padding=(p, 0), groups=channels, bias=False)
        self.proj = nn.Conv2d(channels * 4, channels, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.h1(x), self.h2(x), self.v1(x), self.v2(x)], dim=1)
        return self.act(self.proj(y))


class SFCN_VGG(nn.Module):
    """Simplified SFCN with VGG-16 backbone and spatial encoder."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        vgg = models.vgg16_bn(
            weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Use features up to conv4_3 (stride 8)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])
        self.senc = SpatialEncoder(512, k=9)
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.senc(x)
        x = self.head(x)
        return torch.nn.functional.softplus(x)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler = None,
    accum_steps: int = 1,
    criterion: str = "mse",
    count_loss_alpha: float = 0.0,
) -> float:
    """Train the model for one epoch and return the mean absolute error.

    Supports optional gradient accumulation and auxiliary count loss. The
    auxiliary loss encourages the sum of the predicted density map to match
    the ground-truth count.
    """
    model.train()
    if criterion == "mse":
        crit = nn.MSELoss()
    elif criterion == "huber":
        crit = nn.SmoothL1Loss()
    else:
        raise ValueError("criterion must be 'mse' or 'huber'")
    running_mae, nimg = 0.0, 0
    total_pred_count, total_gt_count = 0.0, 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, (imgs, dens, _) in enumerate(tqdm(loader, desc="Train", leave=False), 1):
        imgs = imgs.to(device)
        dens = dens.to(device)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = model(imgs)
                map_loss = crit(preds, dens)
                # Count loss: MSE between predicted and ground-truth counts
                if count_loss_alpha > 0.0:
                    pred_cnt = preds.sum(dim=(1, 2, 3))
                    gt_cnt = dens.sum(dim=(1, 2, 3))
                    cnt_loss = F.mse_loss(pred_cnt, gt_cnt)
                    total_loss = (map_loss + count_loss_alpha * cnt_loss) / accum_steps
                else:
                    total_loss = map_loss / accum_steps
            scaler.scale(total_loss).backward()
            if step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            preds = model(imgs)
            map_loss = crit(preds, dens)
            if count_loss_alpha > 0.0:
                pred_cnt = preds.sum(dim=(1, 2, 3))
                gt_cnt = dens.sum(dim=(1, 2, 3))
                cnt_loss = F.mse_loss(pred_cnt, gt_cnt)
                total_loss = (map_loss + count_loss_alpha * cnt_loss) / accum_steps
            else:
                total_loss = map_loss / accum_steps
            total_loss.backward()
            if step % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            pc = preds.sum(dim=(1, 2, 3)).detach().cpu().numpy()
            gc = dens.sum(dim=(1, 2, 3)).detach().cpu().numpy()
            running_mae += np.abs(pc - gc).sum()
            total_pred_count += pc.sum()
            total_gt_count += gc.sum()
            nimg += imgs.size(0)
            # Warn if the difference in count is large for any sample
            if step % 10 == 0:
                for i in range(len(pc)):
                    if abs(pc[i] - gc[i]) > 1.0:
                        print(
                            f"[warn] batch {step} sample {i}: density sum {gc[i]:.1f} != pred {pc[i]:.1f}"
                        )
    avg_pred = total_pred_count / max(1, nimg)
    avg_gt = total_gt_count / max(1, nimg)
    print(f"Avg pred count: {avg_pred:.1f} vs GT {avg_gt:.1f}")
    return running_mae / max(1, nimg)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation set and compute MAE and RMSE."""
    model.eval()
    mae, mse, nimg = 0.0, 0.0, 0
    for imgs, dens, _ in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device)
        dens = dens.to(device)
        pred = model(imgs)
        diff = (pred.sum(dim=(1, 2, 3)) - dens.sum(dim=(1, 2, 3))).detach().cpu().numpy()
        mae += np.abs(diff).sum()
        mse += (diff ** 2).sum()
        nimg += imgs.size(0)
    import math

    return mae / max(1, nimg), math.sqrt(mse / max(1, nimg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="data/train/images")
    parser.add_argument("--lbl_dir", type=str, default="data/train/labels")
    parser.add_argument("--base_size", type=int, default=768)
    parser.add_argument("--down", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=384)
    parser.add_argument("--patches_per_image", type=int, default=4)
    parser.add_argument(
        "--avoid_empty_patches",
        action="store_true",
        help="retry random crops until they contain at least one head",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--criterion", type=str, default="mse", choices=["mse", "huber"]
    )
    parser.add_argument(
        "--count_loss_alpha",
        type=float,
        default=0.0,
        help="weight of the auxiliary count loss (0 to disable)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument(
        "--sigma_mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "constant"],
    )
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--save", type=str, default="sfcn_best.pth")
    args = parser.parse_args()

    # Initialise random seeds and device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build list of images and randomly shuffle before splitting
    all_imgs = sorted(glob(os.path.join(args.img_dir, "*.*")))
    random.shuffle(all_imgs)
    n_images = len(all_imgs)
    if n_images < 2:
        raise ValueError("Need at least 2 images for training and validation")
    n_val = max(1, int(0.1 * n_images))
    n_train = n_images - n_val
    train_imgs = all_imgs[:n_train]
    val_imgs = all_imgs[n_train:]

    # Instantiate datasets
    train_ds = CrowdDataset(
        args.img_dir,
        args.lbl_dir,
        base_size=args.base_size,
        down=args.down,
        aug=True,
        mode="train",
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        sigma_mode=args.sigma_mode,
        avoid_empty_patches=args.avoid_empty_patches,
    )
    val_ds = CrowdDataset(
        args.img_dir,
        args.lbl_dir,
        base_size=args.base_size,
        down=args.down,
        aug=False,
        mode="val",
        patch_size=0,
        patches_per_image=1,
        sigma_mode=args.sigma_mode,
        avoid_empty_patches=False,
    )

    # Override image paths after shuffling
    train_ds.img_paths = train_imgs
    val_ds.img_paths = val_imgs
    # Recompute effective lengths for patch training
    if train_ds.mode == "train" and train_ds.patch_size > 0 and train_ds.patches_per_image > 1:
        train_ds.effective_len = len(train_ds.img_paths) * train_ds.patches_per_image
    else:
        train_ds.effective_len = len(train_ds.img_paths)
    val_ds.effective_len = len(val_ds.img_paths)

    # Data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model, optimizer, scaler, and scheduler
    model = SFCN_VGG(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_mae = float("inf")
    patience = args.early_stop_patience
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_mae = train_epoch(
            model,
            train_loader,
            device,
            opt,
            scaler,
            accum_steps=args.accum_steps,
            criterion=args.criterion,
            count_loss_alpha=args.count_loss_alpha,
        )
        va_mae, va_rmse = evaluate(model, val_loader, device)
        sched.step()
        print(
            f"Train MAE: {tr_mae:.3f} | Val MAE: {va_mae:.3f} | Val RMSE: {va_rmse:.3f}"
        )
        if va_mae + 1e-6 < best_mae:
            best_mae = va_mae
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_mae": va_mae,
                    "args": vars(args),
                },
                args.save,
            )
            print(f"✅ New best: {args.save} (Val MAE={va_mae:.3f})")
        else:
            bad_epochs += 1
            print(f"No improvement for {bad_epochs} epoch(s).")
            if bad_epochs >= patience:
                print(
                    f"Early stopping at epoch {epoch}. Best Val MAE: {best_mae:.3f}"
                )
                break
    print("Training finished. Best Val MAE:", best_mae)


if __name__ == "__main__":
    main()