import torch
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from datasetdetection import ChestXrayDetectionDataset
from model_detection import get_detection_model


# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")


# ----------------------------
# Load Dataset
# ----------------------------
full_dataset = ChestXrayDetectionDataset(
    "labels_scaled_224.csv",
    "images"
)

subset_indices = list(range(3000))
subset_dataset = Subset(full_dataset, subset_indices)

train_size = int(0.8 * len(subset_dataset))
val_size = len(subset_dataset) - train_size

train_dataset, val_dataset = random_split(
    subset_dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

print("Train images:", len(train_dataset))
print("Val images:", len(val_dataset))


# ----------------------------
# Model
# ----------------------------
num_classes = 16
model = get_detection_model(num_classes)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 1


# ----------------------------
# Training Logs
# ----------------------------
logs = []

train_losses = []
val_losses = []

best_val_loss = float("inf")

training_start = time.time()


# ==========================================================
# Training Loop
# ==========================================================
for epoch in range(epochs):

    model.train()

    total_loss = 0
    cls_loss_total = 0
    box_loss_total = 0
    obj_loss_total = 0
    rpn_loss_total = 0

    epoch_start = time.time()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_idx, (images, targets) in enumerate(progress_bar):

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        cls_loss = loss_dict["loss_classifier"]
        box_loss = loss_dict["loss_box_reg"]
        obj_loss = loss_dict["loss_objectness"]
        rpn_loss = loss_dict["loss_rpn_box_reg"]

        losses = cls_loss + box_loss + obj_loss + rpn_loss

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        cls_loss_total += cls_loss.item()
        box_loss_total += box_loss.item()
        obj_loss_total += obj_loss.item()
        rpn_loss_total += rpn_loss.item()

        progress_bar.set_postfix({
            "loss": f"{losses.item():.3f}",
            "cls": f"{cls_loss.item():.3f}",
            "box": f"{box_loss.item():.3f}",
            "obj": f"{obj_loss.item():.3f}"
        })

    train_loss = total_loss / len(train_loader)


    # ==========================================================
    # Validation
    # ==========================================================
    model.train()  # detection models need train mode for loss

    val_loss = 0

    with torch.no_grad():

        for images, targets in val_loader:

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            val_loss += losses.item()

    val_loss /= len(val_loader)

    epoch_time = time.time() - epoch_start


    print("\nEpoch", epoch+1, "completed")
    print("Train Loss:", round(train_loss, 4))
    print("Validation Loss:", round(val_loss, 4))
    print("Epoch Time:", round(epoch_time, 2), "seconds\n")


    # ==========================================================
    # Save Best Model
    # ==========================================================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_detection_model.pth")
        print("Best model updated\n")


    # ==========================================================
    # Save Checkpoint
    # ==========================================================
    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")


    # ==========================================================
    # Logging
    # ==========================================================
    logs.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch_time": epoch_time
    })

    train_losses.append(train_loss)
    val_losses.append(val_loss)


# ==========================================================
# Save Logs
# ==========================================================
with open("training_log.json", "w") as f:
    json.dump(logs, f, indent=4)

print("Training log saved")


# ==========================================================
# Plot Loss Curve
# ==========================================================
plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()

plt.savefig("loss_curve.png")

print("Loss curve saved")


# ==========================================================
# Final Model Save
# ==========================================================
total_time = time.time() - training_start

torch.save(model.state_dict(), "detection_model_final.pth")

print("\nTraining Complete")
print("Total Training Time:", round(total_time, 2), "seconds")
print("Final model saved.")