"""
CliniScan - Data Verification Script
Run this BEFORE training to ensure your dataset is ready.
"""

import os
import random
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# ─────────────────────────────────────────────
# CONFIG  (edit these paths)
# ─────────────────────────────────────────────
IMAGES_DIR = "dataset/images/train"       # folder containing .jpg / .png files
LABELS_DIR = "dataset/labels/train"       # folder containing .txt (YOLO) files
IMG_SIZE   = (512, 512)             # expected (W, H)
NUM_SAMPLE = 10                     # images to visualise
# ─────────────────────────────────────────────


def check_dataset(images_dir: str, labels_dir: str, expected_size: tuple):
    img_paths = sorted(Path(images_dir).glob("*.png"))
    print(f"\n{'='*50}")
    print(f"Total images found : {len(img_paths)}")

    missing_labels, size_mismatches, no_annotations = [], [], []

    for img_path in img_paths:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")

        # Check label exists
        if not label_path.exists():
            missing_labels.append(img_path.name)
            continue

        # Check annotation content
        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            no_annotations.append(img_path.name)

        # Check image size
        with Image.open(img_path) as im:
            if im.size != expected_size:
                size_mismatches.append((img_path.name, im.size))

    print(f"Missing labels     : {len(missing_labels)}")
    print(f"Empty annotations  : {len(no_annotations)}")
    print(f"Size mismatches    : {len(size_mismatches)}")

    if missing_labels:
        print("\n[!] Missing labels (first 5):", missing_labels[:5])
    if size_mismatches:
        print("\n[!] Size mismatches (first 5):", size_mismatches[:5])

    print("\n✓ Data check complete." if not (missing_labels or size_mismatches) else
          "\n✗ Issues found — fix before training.")
    return img_paths


def visualise_samples(img_paths, labels_dir: str, n: int = 10):
    """Draw bounding boxes on n random images (YOLO format labels)."""
    samples = random.sample(list(img_paths), min(n, len(img_paths)))
    cols = 5
    rows = max(1, (len(samples) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for ax, img_path in zip(axes, samples):
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        ax.imshow(img, cmap="gray")
        ax.set_title(img_path.stem[:20], fontsize=7)
        ax.axis("off")

        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, bw, bh = map(float, parts)
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        rect = patches.Rectangle(
                            (x1, y1), bw * w, bh * h,
                            linewidth=1.5, edgecolor="red", facecolor="none"
                        )
                        ax.add_patch(rect)

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("sample_visualisation.png", dpi=150)
    plt.show()
    print("Saved: sample_visualisation.png")


if __name__ == "__main__":
    img_paths = check_dataset(IMAGES_DIR, LABELS_DIR, IMG_SIZE)
    visualise_samples(img_paths, LABELS_DIR, NUM_SAMPLE)
