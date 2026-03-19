import torch
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import glob
from datasetdetection import ChestXrayDetectionDataset
import model
from model_detection import get_detection_model


# ==========================================================
# Reproducibility
# ==========================================================
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)

    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = box1_area + box2_area - inter_area + 1e-6

    return inter_area / union

# ==========================================================
# Visualization Function
# ==========================================================
def visualize_predictions(model, dataset, device, epoch, num_images=3):

    model.eval()

    for i in range(num_images):

        idx = random.randint(0, len(dataset)-1)
        image, target = dataset[idx]

        with torch.no_grad():
            prediction = model([image.to(device)])

        pred_boxes = prediction[0]["boxes"].cpu()
        scores = prediction[0]["scores"].cpu()
        labels = prediction[0]["labels"].cpu()

        img = image.permute(1,2,0).cpu().numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(img)
        gt_boxes = target["boxes"].cpu()

        for box in gt_boxes:
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle(
                (x1, y1),
                x2-x1,
                y2-y1,
                linewidth=2,
                edgecolor="lime",
                linestyle="--",
                facecolor="none"
            )
            ax.add_patch(rect)

        for box, score, label in zip(pred_boxes, scores, labels):

            if score < 0.4:   # slightly lower threshold for debugging
                continue

            x1, y1, x2, y2 = box.tolist()

            class_name = dataset.id_to_class.get(int(label), "unknown")
            rect = patches.Rectangle(
                (x1, y1),
                x2-x1,
                y2-y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )

            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 5,
                f"{class_name} ({score:.2f})",
                color="red",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.6)
            )
        gt_boxes = target["boxes"].cpu()

        # Draw GT in GREEN
        
        plt.title(f"Epoch {epoch+1} Prediction {i+1}")

        # Save image
        plt.savefig(f"epoch_{epoch+1}_prediction_{i+1}.png")

        plt.close()

    model.train()

def collate_fn(batch):
    return tuple(zip(*batch))

def main():

    # ==========================================================
    # Load Dataset (with transforms)
    # ==========================================================
    config = {
    "lr": 1e-4,
    "batch_size": 4,
    "epochs": 2,
    "optimizer": "AdamW"
    }
    
    batch_size=config["batch_size"]
    epochs=config["epochs"]
    
    subset_size = min(5016, len(ChestXrayDetectionDataset("filtered_labels.csv", "images")))
    indices = list(range(subset_size))

    train_size = int(0.8 * subset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]


    # Separate datasets (IMPORTANT)
    train_dataset = Subset(
        ChestXrayDetectionDataset(
            "filtered_labels.csv",
            "images",
            augment=True  
        ),
        train_indices
    )

    val_dataset = Subset(
        ChestXrayDetectionDataset(
            "filtered_labels.csv",
            "images",
            augment=True 
            ),
        val_indices
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    # ==========================================================
    # Model
    # ==========================================================
    num_classes = 16

    model = get_detection_model(num_classes, backbone="mobilenet")

    checkpoints = glob.glob("checkpoint_epoch_*.pth")

    if checkpoints:
        latest_checkpoint = sorted(checkpoints)[-1]
        print("Loading:", latest_checkpoint)
        model.load_state_dict(torch.load(latest_checkpoint))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
    )
    model.to(device)
    

    # ==========================================================
    # Training Logs
    # ==========================================================
    logs = []

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    training_start = time.time()


    # ==========================================================
    # Training Loop
    # ==========================================================
    from tqdm import trange
    from tqdm import trange
    f1_scores = []

    for epoch in trange(epochs, desc="Training Progress", unit="epoch"):
        f1_scores = []
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
                "loss": f"{losses.item():.3f}"
            })

        train_loss = total_loss / len(train_loader)


        # ==========================================================
        # Validation
        # ==========================================================
        model.train()  # FasterRCNN returns loss only in train mode

        val_loss = 0
        TP, FP, FN = 0, 0, 0
        with torch.no_grad():

            for images, targets in val_loader:

                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # -------- LOSS (train mode) --------
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                if not torch.isfinite(losses):
                    print("NaN loss detected, skipping batch")
                    continue

                val_loss += losses.item()

                # -------- PREDICTIONS (eval mode) --------
                model.eval()
                preds = model(images)

                for pred, target in zip(preds, targets):

                    pred_boxes = pred["boxes"].cpu()
                    pred_labels = pred["labels"].cpu()
                    pred_scores = pred["scores"].cpu()

                    gt_boxes = target["boxes"].cpu()
                    gt_labels = target["labels"].cpu()

                    matched = set()

                    for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):

                        if ps < 0.5:
                            continue

                        best_iou = 0
                        best_gt_idx = -1

                        for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):

                            iou = compute_iou(pb.numpy(), gb.numpy())

                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = i

                        if best_iou >= 0.5 and best_gt_idx != -1 and pl == gt_labels[best_gt_idx]:
                            TP += 1
                            matched.add(best_gt_idx)
                        else:
                            FP += 1

                    FN += len(gt_boxes) - len(matched)

            lr_scheduler.step()

    
        val_loss /= len(val_loader)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        f1_scores.append(f1_score)
        print("F1 Score:", round(f1_score, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        
        lr_scheduler.step()

        epoch_time = time.time() - epoch_start


        # ==========================================================
        # Print Epoch Results
        # ==========================================================
        print("\nEpoch", epoch+1, "completed")
        print("Train Loss:", round(train_loss, 4))
        print("Validation Loss:", round(val_loss, 4))
        print("Epoch Time:", round(epoch_time, 2), "seconds")

        print("\nLoss Breakdown")
        print("Classifier:", round(cls_loss_total/len(train_loader),4))
        print("Box Reg:", round(box_loss_total/len(train_loader),4))
        print("Objectness:", round(obj_loss_total/len(train_loader),4))
        print("RPN:", round(rpn_loss_total/len(train_loader),4))


        # ==========================================================
        # Show Predictions
        # ==========================================================
        print("\nSample Predictions:")
        visualize_predictions(model, full_dataset, device, epoch, num_images=3)


        # ==========================================================
        # Save Best Model
        # ==========================================================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_detection_model.pth")
            print("Best model updated")


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

    plt.savefig("loss_curve_detect.png")

    print("Loss curve saved")


    # ==========================================================
    # Final Model Save
    # ==========================================================
    total_time = time.time() - training_start

    torch.save(model.state_dict(), "detection_model_final.pth")

    print("\nTraining Complete")
    print("Total Training Time:", round(total_time, 2), "seconds")
    print("Final model saved.")



if __name__ == "__main__":
    main()