"""
CliniScan - Classification Model Training
Predicts WHICH abnormalities exist in a Chest X-Ray.

Supported models: ResNet50, EfficientNet-B0/B2, DenseNet121
Loss: BCEWithLogitsLoss  (multi-label classification)
Metric: AUC-ROC, F1-score
"""

import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score


CONFIG = {
    "images_dir"  : "dataset/images/train",
    "labels_csv"  : "dataset/labels.csv",   
    "model_name"  : "efficientnet_b0",           
    "num_classes" : 14,
    "class_names" : [f"class_{i}" for i in range(14)],
    "img_size"    : 256,
    "batch_size"  : 4,
    "lr"          : 1e-3,
    "epochs"      : 10,
    "val_split"   : 0.15,
    "test_split"  : 0.10,
    "save_dir"    : "checkpoints/classification",
    "device"      : "cpu",
    "mixed_prec"  : False,
}
class ChestXRayDataset(Dataset):
    TRAIN_TF = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    VAL_TF = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    def __init__(self, df: pd.DataFrame, images_dir: str, split: str = "train"):
        self.df         = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.tf         = self.TRAIN_TF if split == "train" else self.VAL_TF
        self.labels     = CONFIG["class_names"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.images_dir / row["filename"]
        image    = Image.open(img_path).convert("RGB")
        image    = self.tf(image)
        label    = torch.tensor(row[self.labels].values.astype(float),
                                dtype=torch.float32)
        return image, label


def build_dataloaders(config: dict):
    df = pd.read_csv(config["labels_csv"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n      = len(df)
    n_test = int(n * config["test_split"])
    n_val  = int(n * config["val_split"])

    test_df  = df.iloc[:n_test]
    val_df   = df.iloc[n_test : n_test + n_val]
    train_df = df.iloc[n_test + n_val:]

    print(f"Split → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    make = lambda df_, split: DataLoader(
        ChestXRayDataset(df_, config["images_dir"], split),
        batch_size  = config["batch_size"],
        shuffle     = (split == "train"),
        num_workers = 0,
        pin_memory  = False,
    )
    return make(train_df, "train"), make(val_df, "val"), make(test_df, "test")


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {name}")

    print(f"Built model: {name}  →  {num_classes} outputs (no final sigmoid; use BCEWithLogitsLoss)")
    return model


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, steps = 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device, enabled=scaler is not None):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        steps      += 1

    return total_loss / steps


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, steps = 0.0, 0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        steps      += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # AUC-ROC 
    aucs = []
    for i in range(all_labels.shape[1]):
        if all_labels[:, i].sum() > 0:
            aucs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
    mean_auc = float(np.mean(aucs)) if aucs else 0.0

    # F1 (threshold = 0.5)
    preds = (all_probs >= 0.5).astype(int)
    f1    = f1_score(all_labels, preds, average="macro", zero_division=0)

    return total_loss / steps, mean_auc, f1
def plot_curves(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(epochs, history["val_auc"])
    axes[1].set_title("Validation AUC-ROC")

    axes[2].plot(epochs, history["val_f1"])
    axes[2].set_title("Validation F1-score")

    for ax in axes:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    out = Path(save_dir) / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved: {out}")


def main():
    cfg    = CONFIG
    device = cfg["device"]
    print(f"Device: {device}")

    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model     = build_model(cfg["model_name"], cfg["num_classes"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    scaler    = torch.cuda.amp.GradScaler() if (cfg["mixed_prec"] and device == "cuda") else None

    history     = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
    best_f1     = -1
    best_path   = Path(cfg["save_dir"]) / "best_model.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_auc, val_f1 = val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{cfg['epochs']} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"AUC={val_auc:.4f} | F1={val_f1:.4f} | {elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (F1={best_f1:.4f})")

    # ── Final test evaluation ──
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_auc, test_f1 = val_epoch(model, test_loader, criterion, device)
    print(f"\nTest → loss={test_loss:.4f}  AUC={test_auc:.4f}  F1={test_f1:.4f}")

    # Save history
    with open(Path(cfg["save_dir"]) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_curves(history, cfg["save_dir"])


if __name__ == "__main__":
    main()
