import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

from dataset import ChestXrayDataset
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Transforms
# --------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# Dataset
# --------------------
train_dataset = ChestXrayDataset("train.csv", "images", train_transform)
val_dataset = ChestXrayDataset("val.csv", "images", val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# --------------------
# Model
# --------------------
num_classes = 15
model = get_model(num_classes).to(device)

# --------------------
# Class imbalance handling
# --------------------
labels_df = train_dataset.df[train_dataset.label_columns]
num_samples = len(labels_df)
positive_counts = labels_df.sum().values
pos_weight = (num_samples - positive_counts) / (positive_counts + 1e-6)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

epochs = 25
best_f1 = 0

train_losses = []
val_losses = []
val_f1_scores = []
val_auc_scores = []

# --------------------
# Training Loop
# --------------------
for epoch in range(epochs):

    # ===== TRAIN =====
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.3).float()

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    val_f1 = f1_score(all_labels, all_preds, average="macro")
    val_auc = roc_auc_score(all_labels, all_probs, average="macro")

    val_f1_scores.append(val_f1)
    val_auc_scores.append(val_auc)

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Macro F1: {val_f1:.4f} | "
          f"Macro AUC: {val_auc:.4f}")

print("Training Complete")
print("Best Validation Macro F1:", best_f1)

# --------------------
# Plot Training Curves
# --------------------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.savefig("loss_curve.png")
plt.close()

plt.figure()
plt.plot(val_f1_scores, label="Val Macro F1")
plt.plot(val_auc_scores, label="Val Macro AUC")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics")
plt.savefig("metrics_curve.png")
plt.close()