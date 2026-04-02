import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

# ---------------------------
# 1. Load YOLO model
# ---------------------------
yolo_model = YOLO("models/best.pt")

# ---------------------------
# 2. Rebuild EfficientNet-B0 (CORRECT)
# ---------------------------
NUM_CLASSES = 15

def load_classification_model():
    # Use B0 (IMPORTANT FIX)
    model = models.efficientnet_b0(pretrained=True)

    # Modify final layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    # Load saved weights
    checkpoint = torch.load("models/classification_model.pth", map_location="cpu")

    # Handle different saving formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    # Load weights safely
    model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    return model

classification_model = load_classification_model()

def get_models():
    return yolo_model, classification_model