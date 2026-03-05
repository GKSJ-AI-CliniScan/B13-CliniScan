import os
print("Status: 1/6 - Loading heavy AI libraries (this takes a few seconds)...")
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. Real Dataset Loader (Reads YOLO .txt labels for Classification)
class CliniScanDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.txt'))
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # Create a tensor of 14 zeros
        labels = torch.zeros(14)
        
        # If the text file exists and has data, change the 0 to a 1 for that disease
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id = int(float(line.split()[0]))
                    if class_id < 14:
                        labels[class_id] = 1.0 # 1.0 means the disease is present
                        
        return image, labels

print("Status: 2/6 - Setting up image transformations...")
# 2. Setup Transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Status: 3/6 - Loading training and validation images from folders...")
# Load Data
train_dataset = CliniScanDataset('vinbigdata_yolo_preprocessed/train/images', 'vinbigdata_yolo_preprocessed/train/labels', transform)
val_dataset = CliniScanDataset('vinbigdata_yolo_preprocessed/val/images', 'vinbigdata_yolo_preprocessed/val/labels', transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Status: 4/6 - Downloading and setting up ResNet50 Model (this may take a minute)...")
# 3. Setup Model (ResNet50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, 14)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Training & Validation Loop
# Updated to 20 epochs to strictly match instructor's requirements
epochs = 20 
best_f1 = 0.0
train_losses, val_losses = [], []

print(f"Status: 5/6 - Starting Training on {device} for {epochs} epochs...")

for epoch in range(epochs):
    # --- TRAINING ---
    model.train()
    running_train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # --- VALIDATION ---
    model.eval()
    running_val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            
            # Apply Sigmoid to get probabilities between 0 and 1
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())
            
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Calculate Metrics
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Threshold at 0.5 for F1 calculation
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Use macro to average the score across all 14 diseases
    val_f1 = f1_score(all_targets, binary_preds, average='macro', zero_division=0)
    
    print(f"Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
    
    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'sample.hcl')
        print(f"New best F1! Model saved to 'sample.hcl'")

print("Status: 6/6 - Training complete! Generating and saving the loss curve graph...")
# 5. Plot Training vs Testing Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Classification Model Loss Curve')
plt.legend()
plt.savefig('classification_loss_curve.png')

print("Script completely finished! You can now check your folder for the 'sample.hcl' and 'classification_loss_curve.png' files.")