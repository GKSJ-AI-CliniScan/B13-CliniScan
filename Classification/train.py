import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from dataset import ChestXrayDataset
from model import get_model
import cv2
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Transforms
# --------------------
train_transform = A.Compose([
    A.Resize(224,224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10,p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(p=0.3),
    A.GaussNoise(p=0.2),
    A.RandomResizedCrop(size=(224,224), scale=(0.9,1.0), p=0.3),
    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),
    ToTensorV2()
])

# --------------------
# Dataset
# --------------------
train_dataset = ChestXrayDataset("train.csv", "images", train_transform)
val_dataset = ChestXrayDataset("val.csv", "images", val_transform)

train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=0)  

val_loader = DataLoader(val_dataset,batch_size=16,shuffle=False,num_workers=0)

# --------------------
# Model
# --------------------
num_classes = 15
model = get_model(num_classes).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model = model.to(device)
# --------------------
# Class imbalance handling
# --------------------
labels_df = train_dataset.df[train_dataset.label_columns]
num_samples = len(labels_df)
positive_counts = labels_df.sum().values
pos_weight = (num_samples - positive_counts) / (positive_counts + 1e-6)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.3,
    patience=2,
    
)

epochs = 18
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
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        thresholds = np.arange(0.1, 0.9, 0.05)

        best_threshold = 0
        best_threshold_f1 = 0

        for t in thresholds:

            preds = (all_probs > t).astype(int)

            f1 = f1_score(all_labels, preds, average="macro")

            if f1 > best_threshold_f1:
                best_threshold_f1 = f1
                best_threshold = t

        val_f1 = best_threshold_f1
        val_auc = roc_auc_score(all_labels, all_probs, average="macro")

        scheduler.step(val_f1)
    

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)


    val_f1_scores.append(val_f1)
    val_auc_scores.append(val_auc)
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")
    torch.save(model.state_dict(), "classify_model.pth")
    print(f"Epoch [{epoch+1}/{epochs}] "
      f"Train Loss: {avg_train_loss:.4f} | "
      f"Val Loss: {avg_val_loss:.4f} | "
      f"Macro F1: {val_f1:.4f} | "
      f"Macro AUC: {val_auc:.4f} | "
      f"Best Threshold: {best_threshold:.2f}")

print("Training Complete")
print("Best Validation Macro F1:", best_f1)

# --------------------
# Plot Training Curves
# --------------------
f1_scores = []

for t in thresholds:
    preds = (all_probs > t).astype(int)
    f1_scores.append(f1_score(all_labels, preds, average="macro"))

# --------------------
# Grad-CAM Visualization
# --------------------

print("Running Grad-CAM visualization...")
model.eval()
gradients = None
activations = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# Select last convolution layer
target_layer = model.features[-1]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Load a test image
img_path = "images\\0a1aef5326b7b24378c6692f7a454e52.png"  # change to your image

image = Image.open(img_path).convert("RGB")

transform = val_transform

input_tensor = transform(image=np.array(image))["image"].unsqueeze(0).to(device)

# Forward pass
output = model(input_tensor)

pred_class = torch.argmax(output, dim=1).item()

# Backprop
model.zero_grad()
output[0, pred_class].backward()

# GradCAM calculation
pooled_gradients = torch.mean(gradients, dim=[0,2,3])

activations = activations.squeeze(0)

for i in range(pooled_gradients.shape[0]):
    activations[i,:,:] *= pooled_gradients[i]

heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()

heatmap = np.maximum(heatmap, 0)
heatmap = heatmap - heatmap.min()
heatmap = heatmap / (heatmap.max() + 1e-8)

# Overlay heatmap
heatmap = cv2.resize(heatmap, (224,224))
heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

img = cv2.imread(img_path)
img = cv2.resize(img,(224,224))

superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

cv2.imwrite("gradcam_result.jpg", superimposed_img)
cv2.imwrite("gradcam_heatmap.jpg", heatmap)

probs = torch.sigmoid(output)
confidence = probs[0][pred_class].item()

print("Predicted class:", pred_class)
print("Confidence:", round(confidence,3))

print("Grad-CAM saved as gradcam_result.jpg")

plt.figure()
plt.plot(thresholds, f1_scores)
plt.xlabel("Threshold")
plt.ylabel("Macro F1")
plt.title("Threshold vs F1 Score")
plt.savefig("threshold_tuning.png")
plt.close()

print("Best Threshold:", best_threshold)
print("Best F1 at this threshold:", best_threshold_f1)

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