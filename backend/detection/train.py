import math
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import VinDrCXRBoxesDataset, collate_fn
from model import get_detection_model

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(data_loader, desc=f"Detection Training Epoch {epoch}")
    
    for i, (images, targets) in enumerate(loop):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        running_loss += loss_value
        loop.set_postfix(loss=loss_value)
        
        if (i + 1) % 10 == 0:
            print(f"Epoch {epoch} - Batch {i+1}/{len(data_loader)} - Loss: {loss_value:.4f}", flush=True)
            
    return running_loss / len(data_loader)

def main():
    # Force CPU for detection as MPS is currently unstable for Faster R-CNN on this machine
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 14 classes + 1 No Finding + 1 background = 16
    num_classes = 16
    train_csv = './data/train.csv'
    img_dir = './data/images'
    
    try:
        dataset = VinDrCXRBoxesDataset(train_csv, img_dir, transform=None)
    except FileNotFoundError:
        print("Data files not found. Skipping dataset initialization for demo purposes.")
        return

    data_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=collate_fn
    )

    model = get_detection_model(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    # Optional learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
        lr_scheduler.step()
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_faster_rcnn_detection.pth")
            print("Saved Best Detection Model!")

if __name__ == '__main__':
    main()
