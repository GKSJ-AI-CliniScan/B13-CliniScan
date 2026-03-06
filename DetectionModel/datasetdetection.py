"""
datasetdetection.py
"""

import os
import random
import math

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image


# ══════════════════════════════════════════════════════════════
# Medical-safe augmentation
# ══════════════════════════════════════════════════════════════
class MedicalAugment:
    """
    All spatial transforms keep bounding boxes in sync.
    Applied only when augment=True (train split).
    """

    def __call__(
        self, image: torch.Tensor, boxes: torch.Tensor
    ):
        _, H, W = image.shape

        # ── 1. Horizontal flip ───────────────────────────────────
        if random.random() < 0.5:
            image = TF.hflip(image)
            if boxes.shape[0] > 0:
                x1_new = W - boxes[:, 2]
                x2_new = W - boxes[:, 0]
                boxes  = torch.stack(
                    [x1_new, boxes[:, 1], x2_new, boxes[:, 3]], dim=1
                )

        # ── 2. Small rotation ±10° ───────────────────────────────
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            if boxes.shape[0] > 0:
                boxes = _rotate_boxes(boxes, angle, W, H)

        # ── 3. Random scaling ±10% ───────────────────────────────
        if random.random() < 0.3:
            scale = random.uniform(0.90, 1.10)
            new_W = max(1, int(W * scale))
            new_H = max(1, int(H * scale))
            image = TF.resize(image, [new_H, new_W])
            image = TF.resize(image, [H, W])   # back to original size
            if boxes.shape[0] > 0:
                boxes = (boxes * scale).clamp(
                    min=torch.tensor([0, 0, 0, 0], dtype=torch.float32),
                    max=torch.tensor([W-1, H-1, W-1, H-1], dtype=torch.float32),
                )

        # ── 4. Brightness  (exposure variation) ─────────────────
        if random.random() < 0.4:
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))

        # ── 5. Contrast  (kVp variation) ─────────────────────────
        if random.random() < 0.4:
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))

        # ── 6. Mild Gaussian noise ───────────────────────────────
        if random.random() < 0.3:
            image = (image + torch.randn_like(image) * 0.01).clamp(0.0, 1.0)

        return image, boxes


def _rotate_boxes(
    boxes: torch.Tensor, angle: float, W: int, H: int
) -> torch.Tensor:
    """
    Rotate bounding boxes around the image centre.
    Returns axis-aligned bounding boxes of the rotated corners.
    """
    cx, cy = W / 2.0, H / 2.0
    rad    = math.radians(-angle)          # PIL rotates counter-clockwise
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        # 4 corners
        corners = [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2)
        ]
        rotated = []
        for x, y in corners:
            xr = cos_a * (x - cx) - sin_a * (y - cy) + cx
            yr = sin_a * (x - cx) + cos_a * (y - cy) + cy
            rotated.append((xr, yr))
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        nx1 = max(0,   min(xs))
        ny1 = max(0,   min(ys))
        nx2 = min(W-1, max(xs))
        ny2 = min(H-1, max(ys))
        if nx2 > nx1 and ny2 > ny1:
            new_boxes.append([nx1, ny1, nx2, ny2])
        else:
            new_boxes.append([x1, y1, x2, y2])   # keep original if degenerate

    return torch.tensor(new_boxes, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════
class ChestXrayDetectionDataset(Dataset):
    """
    Parameters
    ──────────
    csv_file   : labels_scaled_224.csv
                 Required columns: image_id, class_name, class_id,
                                   x_min, y_min, x_max, y_max
    image_dir  : folder containing <image_id>.png (224×224 images)
    augment    : True  → apply MedicalAugment  (train split only)
    max_images : cap on unique images (e.g. 2000 for fast runs)
    cache      : pre-load all images into RAM for faster training
    """

    def __init__(
        self,
        csv_file:   str,
        image_dir:  str,
        augment:    bool = False,
        max_images: int  = None,
        cache:      bool = True,
    ):
        df = pd.read_csv(csv_file)

        # ── Drop degenerate boxes once at load time ──────────────
        has_box  = df["x_min"].notna()
        bad_geom = has_box & (
            (df["x_max"] - df["x_min"] <= 1) |
            (df["y_max"] - df["y_min"] <= 1)
        )
        if bad_geom.sum():
            print(f"[Dataset] Removed {bad_geom.sum()} degenerate boxes.")
        self.df = df[~bad_geom].reset_index(drop=True)

        # ── Image list ───────────────────────────────────────────
        all_ids = self.df["image_id"].unique().tolist()
        if max_images:
            all_ids = all_ids[:max_images]
        self.image_ids = all_ids
        self.image_dir = image_dir
        self.augmentor = MedicalAugment() if augment else None

        # ── Class map  (0 = background, classes start at 1) ─────
        classes = sorted(
            self.df[self.df["class_name"] != "No finding"]["class_name"]
            .dropna().unique().tolist()
        )
        self.class_to_id = {c: i + 1 for i, c in enumerate(classes)}
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        self.id_to_class[0] = "background"

        print(f"[Dataset] {len(self.image_ids)} images | "
              f"{len(self.class_to_id)} classes | "
              f"augment={augment}")

        # ── Image cache ──────────────────────────────────────────
        self._cache: dict = {}
        if cache:
            print(f"[Dataset] Caching {len(self.image_ids)} images …",
                  flush=True)
            for img_id in self.image_ids:
                self._cache[img_id] = self._load_image(img_id)
            print("[Dataset] Cache ready.")

    # ── Image loader ──────────────────────────────────────────────
    def _load_image(self, image_id: str) -> torch.Tensor:
        path = os.path.join(self.image_dir, image_id + ".png")
        img  = Image.open(path).convert("RGB")

        # MONOCHROME1 inversion guard
        # Some X-rays store bright = air (needs inversion to bright = tissue)
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.mean() > 0.7:          # likely inverted  (mostly bright)
            arr = 1.0 - arr

        return torch.from_numpy(arr).permute(2, 0, 1)  # C×H×W  float32

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # ── Image ────────────────────────────────────────────────
        image = (
            self._cache[image_id].clone()
            if image_id in self._cache
            else self._load_image(image_id)
        )

        # ── Annotations ──────────────────────────────────────────
        records = self.df[self.df["image_id"] == image_id]
        boxes, labels = [], []

        for _, row in records.iterrows():
            if pd.notna(row["x_min"]) and row["class_name"] != "No finding":
                x1, y1 = float(row["x_min"]), float(row["y_min"])
                x2, y2 = float(row["x_max"]), float(row["y_max"])
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(row["class_id"]) + 1)  # 0 = background

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        # ── Augment (train only) ──────────────────────────────────
        if self.augmentor is not None:
            image, boxes_t = self.augmentor(image, boxes_t)

        # ── area + iscrowd  (required by mAP evaluator) ──────────
        if boxes_t.shape[0] > 0:
            area = (
                (boxes_t[:, 2] - boxes_t[:, 0]) *
                (boxes_t[:, 3] - boxes_t[:, 1])
            )
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  torch.zeros(len(labels_t), dtype=torch.int64),
        }
        return image, target