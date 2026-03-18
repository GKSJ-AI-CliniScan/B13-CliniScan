import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import os

# =============================
# EXPERIMENT SETTINGS
# =============================

MODEL_NAME = "ResNet50"
LEARNING_RATE = 0.0005
BATCH_SIZE = 16
OPTIMIZER_NAME = "AdamW"
EPOCHS = 20
IMAGE_SIZE = 224

# =============================
# DEVICE
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# =============================
# DATA AUGMENTATION
# =============================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85,1.0)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# =============================
# LOAD DATASET
# =============================

train_data = datasets.ImageFolder("dataset_merged/train", transform=train_transform)
val_data = datasets.ImageFolder("dataset_merged/val", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_data.classes)

print("Classes:", train_data.classes)

# =============================
# COMPUTE CLASS WEIGHTS
# =============================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.targets),
    y=train_data.targets
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print("Class Weights:", class_weights)

# =============================
# LOAD PRETRAINED MODEL
# =============================

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model = model.to(device)

# =============================
# LOAD PREVIOUS CHECKPOINT
# =============================

if os.path.exists("best_resnet50_final.pth"):
    print("Loading previous checkpoint...")
    model.load_state_dict(torch.load("best_resnet50_final.pth", map_location=device))

# =============================
# LOSS FUNCTION
# =============================

criterion = nn.CrossEntropyLoss(weight=class_weights)

# =============================
# OPTIMIZER
# =============================

if OPTIMIZER_NAME == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

elif OPTIMIZER_NAME == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# =============================
# LR SCHEDULER
# =============================

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

# =============================
# TRAINING SETTINGS
# =============================

best_f1 = 0.0

# =============================
# TRAINING LOOP
# =============================

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print("Training Loss:", round(epoch_loss,4))

    # =============================
    # VALIDATION
    # =============================

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs,1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print("Validation Macro F1:", round(macro_f1,4))

    # =============================
    # SAVE BEST MODEL
    # =============================

    if macro_f1 > best_f1:

        best_f1 = macro_f1

        torch.save(model.state_dict(),"best_resnet50_final.pth")

        print("Best Model Saved!")

    scheduler.step()

# =============================
# TRAINING COMPLETE
# =============================

print("\nTraining Completed!")
print("Best Macro F1 Achieved:", best_f1)