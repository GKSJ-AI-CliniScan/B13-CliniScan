import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import VinDrCXRClassificationDataset, get_train_val_transforms
from model import CliniScanClassifier
from sklearn.metrics import roc_auc_score

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    loop = tqdm(loader, desc="Training Epoch")
    for images, targets in loop:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())
        
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    for images, targets in tqdm(loader, desc="Evaluating"):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        
        all_targets.append(targets.cpu())
        all_outputs.append(torch.sigmoid(outputs).cpu())
        
    all_targets = torch.cat(all_targets).numpy()
    all_outputs = torch.cat(all_outputs).numpy()
    
    # Calculate Macro AUC
    try:
        auc = roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovr')
    except ValueError:
        auc = 0.0 # handles edge case where batch has only one class
        
    return running_loss / len(loader.dataset), auc

def main():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Dummy paths for local
    train_csv = './data/train.csv'
    val_csv = './data/train.csv' # For demonstration, reusing same dataset for val
    img_dir = './data/images'
    
    train_transform, val_transform = get_train_val_transforms()
    
    try:
        train_dataset = VinDrCXRClassificationDataset(train_csv, img_dir, transform=train_transform)
        val_dataset = VinDrCXRClassificationDataset(val_csv, img_dir, transform=val_transform)
    except FileNotFoundError:
        print("Data files not found. Skipping dataset initialization for demo purposes.")
        return

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = CliniScanClassifier(num_classes=15).to(device)
    # Using BCEWithLogitsLoss for Multi-Label Classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 10
    best_auc = 0.0
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_resnet_classification.pth")
            print("Saved Best Model!")

if __name__ == '__main__':
    main()
