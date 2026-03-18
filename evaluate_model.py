import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import os

# =============================
# EXPERIMENT SETTINGS
# =============================

MODEL_NAME = "ResNet50"
LEARNING_RATE = 0.001
BATCH_SIZE = 16
OPTIMIZER = "Adam"
EPOCHS = 20

# =============================
# DEVICE
# =============================

device = torch.device("cpu")

# =============================
# IMAGE TRANSFORM
# =============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =============================
# LOAD VALIDATION DATASET
# =============================

val_data = datasets.ImageFolder("dataset_merged/val", transform=transform)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", val_data.classes)

# =============================
# LOAD MODEL
# =============================

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(val_data.classes))
model = model.to(device)

model.load_state_dict(torch.load("best_resnet50_final.pth", map_location=device))

print("Model Loaded Successfully!")

# =============================
# EVALUATION
# =============================

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# =============================
# METRICS
# =============================

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
macro_f1 = f1_score(all_labels, all_preds, average='macro')

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Macro F1 Score:", macro_f1)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))

# =============================
# SAVE RESULTS TO CSV
# =============================

results = {
    "Model": MODEL_NAME,
    "Learning_Rate": LEARNING_RATE,
    "Batch_Size": BATCH_SIZE,
    "Optimizer": OPTIMIZER,
    "Epochs": EPOCHS,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Macro_F1": macro_f1
}

df = pd.DataFrame([results])

csv_file = "experiment_results.csv"

if os.path.exists(csv_file):
    df.to_csv(csv_file, mode='a', header=False, index=False)
else:
    df.to_csv(csv_file, index=False)

print("\nExperiment results saved to experiment_results.csv")