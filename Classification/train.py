"""
train.py  –  Multi-label chest X-ray classification + Grad-CAM grid
═══════════════════════════════════════════════════════════════════════
Grad-CAM output is ONE grid figure saved as a single PNG:

    gradcam_outputs/gradcam_<stem>_grid.png

Grid layout  (one ROW per predicted-positive class):
  Col 0 │ Original X-ray
  Col 1 │ Grad-CAM heatmap overlaid on X-ray
  Col 2 │ Raw heatmap  (jet colormap, no X-ray)

A shared colourbar is placed to the right of every row.
The figure title shows the image filename.
Each row title shows: class name + confidence score.
═══════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import ChestXrayDataset
from model import get_model


# ──────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")


# ──────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(p=0.3),
    A.GaussNoise(p=0.2),
    A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ──────────────────────────────────────────────────────────────────────
# Datasets & loaders
# ──────────────────────────────────────────────────────────────────────
train_dataset = ChestXrayDataset("train.csv", "images", train_transform)
val_dataset   = ChestXrayDataset("val.csv",   "images", val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=0)


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────
num_classes = 15
model = get_model(num_classes).to(device)

if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("[Info] Loaded checkpoint: best_model.pth")
else:
    print("[Info] No checkpoint found – training from scratch.")


# ──────────────────────────────────────────────────────────────────────
# Grad-CAM hooks  (registered ONCE before any forward pass)
# ──────────────────────────────────────────────────────────────────────
_gradients:   torch.Tensor | None = None
_activations: torch.Tensor | None = None


def _forward_hook(module, inp, out):
    global _activations
    _activations = out.detach()


def _backward_hook(module, grad_in, grad_out):
    global _gradients
    _gradients = grad_out[0].detach()


_target_layer = model.features[-1]
_target_layer.register_forward_hook(_forward_hook)
_target_layer.register_full_backward_hook(_backward_hook)


# ──────────────────────────────────────────────────────────────────────
# Helper: compute one Grad-CAM heatmap for a given class index
# ──────────────────────────────────────────────────────────────────────
def _compute_heatmap(input_tensor: torch.Tensor, cls_idx: int) -> np.ndarray:
    """
    Returns a float32 array of shape (224, 224) normalised to [0, 1].
    Runs one forward + one backward pass for *cls_idx*.
    """
    model.zero_grad()
    out = model(input_tensor)
    out[0, cls_idx].backward()

    if _gradients is None or _activations is None:
        return np.zeros((224, 224), dtype=np.float32)

    # Grad-CAM: weight each channel by its pooled gradient
    pooled   = _gradients.mean(dim=(2, 3), keepdim=True)       # (1, C, 1, 1)
    weighted = (_activations * pooled).sum(dim=1).squeeze()     # (H_f, W_f)
    hmap     = torch.relu(weighted).cpu().numpy()

    hmap -= hmap.min()
    if hmap.max() > 0:
        hmap /= hmap.max()

    # Bilinear upsample to 224×224
    hmap_t  = torch.from_numpy(hmap).unsqueeze(0).unsqueeze(0)
    hmap_up = F.interpolate(hmap_t, size=(224, 224),
                            mode="bilinear", align_corners=False)
    return hmap_up.squeeze().numpy()


# ──────────────────────────────────────────────────────────────────────
# Grad-CAM grid  –  ONE figure, all positive classes as rows
# ──────────────────────────────────────────────────────────────────────
def run_gradcam_grid(img_path: str, threshold: float = 0.4, out_dir: str = "."):
    """
    Saves a single PNG grid:

        <out_dir>/gradcam_<stem>_grid.png

    Grid structure
    ──────────────
    Rows  : one per predicted-positive class  (min 1, falls back to top-1)
    Col 0 : Original X-ray                    (grey, no overlay)
    Col 1 : Grad-CAM overlay                  (heatmap blended over X-ray)
    Col 2 : Raw heatmap                        (jet colormap only)
    Col 3 : Shared colour bar for that row

    Row labels (left y-axis): class name + confidence
    Column headers (top): "Original", "Grad-CAM Overlay", "Activation Map"
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Load image ───────────────────────────────────────────────────
    raw_pil    = Image.open(img_path).convert("RGB")
    raw_np_224 = np.array(raw_pil.resize((224, 224)), dtype=np.float32) / 255.0

    aug_out      = val_transform(image=np.array(raw_pil))
    input_tensor = aug_out["image"].unsqueeze(0).to(device)   # (1,3,224,224)

    # ── Forward pass → class probabilities ───────────────────────────
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    probs = torch.sigmoid(output)[0]                          # (num_classes,)

    positive_indices = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()
    if not positive_indices:
        positive_indices = [int(probs.argmax())]              # top-1 fallback

    label_cols = train_dataset.label_columns
    n_classes  = len(positive_indices)

    # ── Figure layout ─────────────────────────────────────────────────
    # 3 image columns + 1 narrow colourbar column per row
    # Each cell is ~3.5 inches wide × 3.5 inches tall
    CELL_W, CELL_H = 3.5, 3.5
    CB_W           = 0.35          # colourbar column width (inches)
    fig_w = CELL_W * 3 + CB_W + 0.6
    fig_h = CELL_H * n_classes + 0.9   # + room for column headers

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=130)
    fig.patch.set_facecolor("#1a1a2e")   # dark navy background

    stem = os.path.splitext(os.path.basename(img_path))[0]
    fig.suptitle(
        f"Grad-CAM Grid  ·  {stem}",
        fontsize=13, fontweight="bold", color="white", y=1.002,
    )

    # Column header positions (fractional figure coords)
    col_centres = [
        (0.02 + CELL_W * 0.5) / fig_w,
        (0.02 + CELL_W * 1.5) / fig_w,
        (0.02 + CELL_W * 2.5) / fig_w,
    ]
    col_titles = ["Original X-ray", "Grad-CAM Overlay", "Activation Map"]
    for cx, ct in zip(col_centres, col_titles):
        fig.text(cx, 0.995, ct, ha="center", va="top",
                 fontsize=9, color="#aad4f5", fontweight="bold")

    # ── One row per positive class ────────────────────────────────────
    cmap    = cm.jet
    norm    = mcolors.Normalize(vmin=0, vmax=1)

    for row_idx, cls_idx in enumerate(positive_indices):

        class_name = (label_cols[cls_idx] if cls_idx < len(label_cols)
                      else f"class_{cls_idx}")
        confidence = float(probs[cls_idx])

        # Compute Grad-CAM heatmap (needs grad → can't be inside no_grad)
        hmap_224 = _compute_heatmap(input_tensor, cls_idx)

        # Build RGBA overlay: alpha proportional to activation
        hmap_rgba        = cmap(hmap_224)             # (224, 224, 4)
        hmap_rgba[..., 3] = np.clip(hmap_224 * 0.72, 0.0, 1.0)

        # ── Axes geometry ─────────────────────────────────────────────
        # GridSpec: 3 equal image cols + 1 narrow cb col
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(
            n_classes, 4,
            figure=fig,
            left=0.02  / fig_w + 0.01,
            right=1.0  - (CB_W + 0.15) / fig_w,
            top=0.97,
            bottom=0.03,
            hspace=0.08,
            wspace=0.04,
            width_ratios=[1, 1, 1, CB_W / CELL_W],
        )

        ax_orig  = fig.add_subplot(gs[row_idx, 0])
        ax_over  = fig.add_subplot(gs[row_idx, 1])
        ax_raw   = fig.add_subplot(gs[row_idx, 2])
        ax_cb    = fig.add_subplot(gs[row_idx, 3])

        # ── Col 0: original ───────────────────────────────────────────
        ax_orig.imshow(raw_np_224, vmin=0, vmax=1)
        ax_orig.axis("off")

        # ── Col 1: overlay ────────────────────────────────────────────
        ax_over.imshow(raw_np_224, vmin=0, vmax=1)
        ax_over.imshow(hmap_rgba, interpolation="bilinear")
        ax_over.axis("off")

        # ── Col 2: raw heatmap ────────────────────────────────────────
        ax_raw.imshow(hmap_224, cmap="jet", vmin=0, vmax=1,
                      interpolation="bilinear")
        ax_raw.axis("off")

        # ── Col 3: colour bar ─────────────────────────────────────────
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb)
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.set_ticklabels(["Low", "Mid", "High"], fontsize=6.5)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        cb.outline.set_edgecolor("white")

        # ── Row label on left y-axis of Col 0 ────────────────────────
        ax_orig.set_ylabel(
            f"{class_name}\n{confidence:.2f}",
            fontsize=8, fontweight="bold", color="white",
            rotation=0, labelpad=70, va="center",
        )

        # Dark background for each cell
        for ax in (ax_orig, ax_over, ax_raw):
            ax.set_facecolor("#0d0d1a")

    out_path = os.path.join(out_dir, f"gradcam_{stem}_grid.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [GradCAM grid] → {out_path}  ({n_classes} class(es))")
    return out_path


# ──────────────────────────────────────────────────────────────────────
# Loss & optimiser
# ──────────────────────────────────────────────────────────────────────
labels_df       = train_dataset.df[train_dataset.label_columns]
positive_counts = labels_df.sum().values
pos_weight      = (len(labels_df) - positive_counts) / (positive_counts + 1e-6)
pos_weight_t    = torch.tensor(pos_weight, dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.3, patience=2
)


# ──────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────
epochs     = 18
best_f1    = 0.0
thresholds = np.arange(0.1, 0.9, 0.05)

train_losses   = []
val_losses     = []
val_f1_scores  = []
val_auc_scores = []

last_all_labels = None
last_all_probs  = None

for epoch in range(epochs):

    # ── Train ────────────────────────────────────────────────────────
    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ── Validate ─────────────────────────────────────────────────────
    model.eval()
    running_val_loss = 0.0
    all_labels_list  = []
    all_probs_list   = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            running_val_loss += criterion(outputs, labels).item()
            all_labels_list.append(labels.cpu())
            all_probs_list.append(torch.sigmoid(outputs).cpu())

    all_labels = torch.cat(all_labels_list).numpy()
    all_probs  = torch.cat(all_probs_list).numpy()
    last_all_labels = all_labels
    last_all_probs  = all_probs

    # Threshold sweep
    best_thr, best_thr_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(all_labels, (all_probs > t).astype(int),
                      average="macro", zero_division=0)
        if f1 > best_thr_f1:
            best_thr_f1, best_thr = f1, t

    val_f1  = best_thr_f1
    val_auc = roc_auc_score(all_labels, all_probs, average="macro")
    avg_val_loss = running_val_loss / len(val_loader)

    val_losses.append(avg_val_loss)
    val_f1_scores.append(val_f1)
    val_auc_scores.append(val_auc)

    scheduler.step(val_f1)

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")

    torch.save(model.state_dict(), "classify_model.pth")

    print(
        f"Epoch [{epoch+1:>2}/{epochs}] "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Macro F1: {val_f1:.4f} | "
        f"Macro AUC: {val_auc:.4f} | "
        f"Best Thr: {best_thr:.2f}"
    )

print(f"\nTraining complete.  Best Val Macro F1: {best_f1:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Post-training plots
# ──────────────────────────────────────────────────────────────────────
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend(); plt.tight_layout()
plt.savefig("loss_curve.png"); plt.close()

plt.figure()
plt.plot(val_f1_scores,  label="Val Macro F1")
plt.plot(val_auc_scores, label="Val Macro AUC")
plt.xlabel("Epoch"); plt.ylabel("Score")
plt.title("Validation Metrics")
plt.legend(); plt.tight_layout()
plt.savefig("metrics_curve.png"); plt.close()

if last_all_probs is not None:
    f1_by_thr = [
        f1_score(last_all_labels, (last_all_probs > t).astype(int),
                 average="macro", zero_division=0)
        for t in thresholds
    ]
    plt.figure()
    plt.plot(thresholds, f1_by_thr)
    plt.xlabel("Threshold"); plt.ylabel("Macro F1")
    plt.title("Threshold vs F1 (last epoch)")
    plt.tight_layout()
    plt.savefig("threshold_tuning.png"); plt.close()

print("Plots saved: loss_curve.png | metrics_curve.png | threshold_tuning.png")


# ──────────────────────────────────────────────────────────────────────
# Grad-CAM grid  (all positive classes in one figure)
# ──────────────────────────────────────────────────────────────────────
SAMPLE_IMAGE = os.path.join("images", "0a1aef5326b7b24378c6692f7a454e52.png")

if os.path.exists(SAMPLE_IMAGE):
    print(f"\nRunning Grad-CAM grid on: {SAMPLE_IMAGE}")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    out = run_gradcam_grid(SAMPLE_IMAGE, threshold=0.4, out_dir="gradcam_outputs")
    print(f"Grad-CAM grid saved → {out}")
else:
    print(f"[GradCAM] Sample image not found at {SAMPLE_IMAGE!r} – skipping.")