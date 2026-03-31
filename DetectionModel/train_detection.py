"""
train_detection.py  –  Faster R-CNN chest X-ray detection
═══════════════════════════════════════════════════════════════════════
Fixes in this version
──────────────────────
1.  Heatmap removed from detection entirely — it belongs in train.py
    (classification).  Detection visualisation shows ONLY bounding
    boxes on the raw X-ray.
2.  Smart image selection: always picks images that have ≥1 GT box
    so "Image 1" is never a blank "No finding" scan.
3.  GT label staggering: labels are nudged vertically when multiple
    boxes share the same bottom edge, so text never piles up.
4.  NMS + score-threshold filter applied before drawing.
5.  val_dataset augment=False.
6.  lr_scheduler.step() called exactly once per epoch.
═══════════════════════════════════════════════════════════════════════
"""

import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import nms
from tqdm import tqdm, trange
import glob

from datasetdetection import ChestXrayDetectionDataset
from model_detection import get_detection_model, model_summary


# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")


# ──────────────────────────────────────────────────────────────────────
# IoU helper
# ──────────────────────────────────────────────────────────────────────
def compute_iou(box1, box2) -> float:
    xA = max(box1[0], box2[0]);  yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]);  yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


# ──────────────────────────────────────────────────────────────────────
# Helper: draw boxes with staggered labels on an Axes
# ──────────────────────────────────────────────────────────────────────
def _draw_boxes(ax, boxes, label_names, scores=None,
                color="lime", linestyle="--", label_pos="below"):
    """
    Parameters
    ----------
    boxes       : (N,4) tensor or list of [x1,y1,x2,y2]
    label_names : list of N strings
    scores      : optional (N,) tensor; appended to each label as " (0.XX)"
    color       : edge + text colour
    linestyle   : "--" for GT, "-" for predictions
    label_pos   : "below" → text under box  |  "above" → text above box
    """
    used: list = []   # (x1, base_y) already placed; used to detect collisions

    for i, (box, name) in enumerate(zip(boxes, label_names)):
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        score_str = f" ({float(scores[i]):.2f})" if scores is not None else ""
        text      = name + score_str

        # Box rectangle
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color,
            linestyle=linestyle, facecolor="none",
        )
        ax.add_patch(rect)

        # Compute label anchor; stagger if another label is very close
        base_y    = y2 + 5 if label_pos == "below" else y1 - 5
        collisions = sum(
            1 for (px, py) in used
            if abs(px - x1) < 50 and abs(py - base_y) < 12
        )
        label_y = base_y + collisions * 13 if label_pos == "below" \
                  else base_y - collisions * 13
        used.append((x1, base_y))

        bg  = "black" if label_pos == "below" else "white"
        va  = "top"   if label_pos == "below" else "bottom"
        ax.text(
            x1, label_y, text,
            color=color, fontsize=6.5, fontweight="bold", va=va,
            bbox=dict(facecolor=bg, alpha=0.55, pad=1.5, edgecolor="none"),
        )


# ──────────────────────────────────────────────────────────────────────
# Smart image selector: skip "No finding" images
# ──────────────────────────────────────────────────────────────────────
def _select_with_boxes(dataset, n: int, max_scan: int = 300) -> list:
    """Return up to n dataset indices that have >=1 GT bounding box."""
    selected = []
    for i in range(min(max_scan, len(dataset))):
        _, target = dataset[i]
        if target["boxes"].shape[0] > 0:
            selected.append(i)
        if len(selected) == n:
            break
    # Fallback: if dataset has no annotated images at all
    if not selected:
        selected = list(range(min(n, len(dataset))))
    return selected


# ══════════════════════════════════════════════════════════════════════
# Visualisation: bounding boxes only (no heatmap)
# ══════════════════════════════════════════════════════════════════════
def visualize_predictions(
    model,
    dataset,
    device,
    epoch: int,
    num_images:     int   = 3,
    score_thresh:   float = 0.30,
    nms_iou_thresh: float = 0.50,
    out_dir:        str   = ".",
):
    """
    For each of *num_images* smart-selected images, save ONE PNG:

        epoch_N_prediction_M.png
            Raw X-ray with:
              • green dashed boxes = ground truth
              • red   solid  boxes = model predictions  (score ≥ score_thresh)
            Labels are staggered so they never pile up.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    indices = _select_with_boxes(dataset, num_images)

    for panel_num, idx in enumerate(indices, start=1):

        image, target = dataset[idx]
        img_np = np.clip(image.permute(1, 2, 0).cpu().numpy(), 0, 1)

        # ── Model inference ───────────────────────────────────────────
        with torch.no_grad():
            prediction = model([image.to(device)])

        pred_boxes  = prediction[0]["boxes"].cpu()
        scores      = prediction[0]["scores"].cpu()
        pred_labels = prediction[0]["labels"].cpu()

        # NMS
        if pred_boxes.shape[0] > 0:
            keep        = nms(pred_boxes, scores, iou_threshold=nms_iou_thresh)
            pred_boxes  = pred_boxes[keep]
            scores      = scores[keep]
            pred_labels = pred_labels[keep]

        # Score threshold
        if pred_boxes.shape[0] > 0:
            mask        = scores >= score_thresh
            pred_boxes  = pred_boxes[mask]
            scores      = scores[mask]
            pred_labels = pred_labels[mask]

        gt_boxes  = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()
        n_gt      = gt_boxes.shape[0]
        n_pred    = pred_boxes.shape[0]

        gt_names   = [dataset.id_to_class.get(int(l), f"cls{int(l)}")
                      for l in gt_labels]
        pred_names = [dataset.id_to_class.get(int(l), f"cls{int(l)}")
                      for l in pred_labels]

        # ── Plot ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img_np, vmin=0, vmax=1)

        _draw_boxes(ax, gt_boxes, gt_names,
                    color="lime", linestyle="--", label_pos="below")
        _draw_boxes(ax, pred_boxes, pred_names, scores=scores,
                    color="red",  linestyle="-",  label_pos="above")

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="lime", linewidth=2,
                   linestyle="--", label="Ground Truth"),
            Line2D([0], [0], color="red",  linewidth=2,
                   linestyle="-",  label="Prediction"),
        ]
        ax.legend(handles=legend_elements, loc="upper right",
                  fontsize=7, framealpha=0.7)

        ax.set_title(
            f"Epoch {epoch+1}  |  Sample {panel_num}  "
            f"[GT: {n_gt}   Pred: {n_pred}]",
            fontsize=9, pad=5,
        )
        ax.axis("off")
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"epoch_{epoch+1}_prediction_{panel_num}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Viz] → {out_path}  (GT:{n_gt}  Pred:{n_pred})")

    model.train()


# ──────────────────────────────────────────────────────────────────────
# Collate fn
# ──────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    return tuple(zip(*batch))


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():

    config = {
        "lr":         1e-4,
        "batch_size": 4,
        "epochs":     4,
    }
    lr         = config["lr"]
    batch_size = config["batch_size"]
    epochs     = config["epochs"]

    # ── Dataset ───────────────────────────────────────────────────────
    full_ds     = ChestXrayDetectionDataset(
        "filtered_labels.csv", "images", augment=False
    )
    subset_size  = min(5016, len(full_ds))
    indices      = list(range(subset_size))
    train_size   = int(0.8 * subset_size)
    train_indices = indices[:train_size]
    val_indices   = indices[train_size:]

    train_dataset = Subset(
        ChestXrayDetectionDataset("filtered_labels.csv", "images", augment=True),
        train_indices,
    )
    val_dataset = Subset(
        ChestXrayDetectionDataset("filtered_labels.csv", "images", augment=False),
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────
    num_classes = 16
    model = get_detection_model(num_classes, backbone="mobilenet")
    model_summary(model)

    checkpoints = sorted(glob.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"[Info] Loading checkpoint: {latest}")
        model.load_state_dict(torch.load(latest, map_location=device))

    model.to(device)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1
    )

    logs          = []
    train_losses  = []
    val_losses    = []
    f1_scores     = []
    best_val_loss = float("inf")
    training_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════
    for epoch in trange(epochs, desc="Training Progress", unit="epoch"):

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss   = 0.0
        cls_loss_tot = 0.0
        box_loss_tot = 0.0
        obj_loss_tot = 0.0
        rpn_loss_tot = 0.0
        epoch_start  = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, targets in pbar:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            cls_loss  = loss_dict["loss_classifier"]
            box_loss  = loss_dict["loss_box_reg"]
            obj_loss  = loss_dict["loss_objectness"]
            rpn_loss  = loss_dict["loss_rpn_box_reg"]
            losses    = cls_loss + box_loss + obj_loss + rpn_loss

            if not torch.isfinite(losses):
                print("[Train] NaN/Inf loss – skipping batch")
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss   += losses.item()
            cls_loss_tot += cls_loss.item()
            box_loss_tot += box_loss.item()
            obj_loss_tot += obj_loss.item()
            rpn_loss_tot += rpn_loss.item()

            pbar.set_postfix(loss=f"{losses.item():.3f}")

        train_loss = total_loss / max(len(train_loader), 1)

        # ── Validate ──────────────────────────────────────────────────
        val_loss_sum = 0.0
        TP = FP = FN = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images  = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Loss requires train mode
                model.train()
                loss_dict  = model(images, targets)
                batch_loss = sum(v for v in loss_dict.values())
                if not torch.isfinite(batch_loss):
                    continue
                val_loss_sum += batch_loss.item()

                # Predictions require eval mode
                model.eval()
                preds = model(images)

                for pred, target in zip(preds, targets):
                    pb = pred["boxes"].cpu()
                    pl = pred["labels"].cpu()
                    ps = pred["scores"].cpu()

                    if pb.shape[0] > 0:
                        keep = nms(pb, ps, iou_threshold=0.5)
                        pb, pl, ps = pb[keep], pl[keep], ps[keep]

                    gb = target["boxes"].cpu()
                    gl = target["labels"].cpu()

                    matched = set()
                    for b, lbl, score in zip(pb, pl, ps):
                        if score < 0.5:
                            continue
                        best_iou, best_idx = 0.0, -1
                        for gi, (g, gl_i) in enumerate(zip(gb, gl)):
                            iou = compute_iou(b.numpy(), g.numpy())
                            if iou > best_iou:
                                best_iou, best_idx = iou, gi
                        if (best_iou >= 0.5 and best_idx >= 0
                                and lbl == gl[best_idx]
                                and best_idx not in matched):
                            TP += 1
                            matched.add(best_idx)
                        else:
                            FP += 1
                    FN += len(gb) - len(matched)

        # Single scheduler step per epoch
        lr_scheduler.step()

        val_loss  = val_loss_sum / max(len(val_loader), 1)
        precision = TP / (TP + FP + 1e-6)
        recall    = TP / (TP + FN + 1e-6)
        f1        = 2 * precision * recall / (precision + recall + 1e-6)
        f1_scores.append(f1)

        epoch_time = time.time() - epoch_start
        n = max(len(train_loader), 1)

        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1}/{epochs}   time: {epoch_time:.1f}s")
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val   Loss : {val_loss:.4f}")
        print(f"  Precision  : {precision:.4f}   Recall: {recall:.4f}   F1: {f1:.4f}")
        print(f"  Loss breakdown (train avg):")
        print(f"    Classifier : {cls_loss_tot/n:.4f}")
        print(f"    Box Reg    : {box_loss_tot/n:.4f}")
        print(f"    Objectness : {obj_loss_tot/n:.4f}")
        print(f"    RPN Box Reg: {rpn_loss_tot/n:.4f}")

        print("\nGenerating visualisations …")
        visualize_predictions(
            model, train_dataset.dataset, device, epoch,
            num_images=3, score_thresh=0.30, nms_iou_thresh=0.50,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_detection_model.pth")
            print("  ★ Best model updated.")

        ckpt = f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"  Checkpoint saved: {ckpt}")

        logs.append({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "precision":  precision,
            "recall":     recall,
            "f1":         f1,
            "epoch_time": epoch_time,
        })
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # ── Post-training plots ───────────────────────────────────────────
    with open("training_log.json", "w") as fh:
        json.dump(logs, fh, indent=4)
    print("\nTraining log saved: training_log.json")

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, epochs + 1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses,   marker="s", label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig("loss_curve_detect.png", dpi=120)
    plt.close()

    if f1_scores:
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(f1_scores) + 1), f1_scores,
                 marker="o", color="green")
        plt.xlabel("Epoch"); plt.ylabel("F1 Score")
        plt.title("Validation F1 per Epoch")
        plt.tight_layout()
        plt.savefig("f1_curve_detect.png", dpi=120)
        plt.close()

    total_time = time.time() - training_start
    torch.save(model.state_dict(), "detection_model_final.pth")
    print(f"\nTraining complete in {total_time:.1f}s")
    print("Final model saved: detection_model_final.pth")


if __name__ == "__main__":
    main()