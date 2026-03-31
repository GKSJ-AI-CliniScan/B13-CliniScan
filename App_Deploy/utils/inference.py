import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import numpy as np
import torchvision.models as models
import torch.nn as nn
import tempfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import gdown

os.makedirs("models", exist_ok=True)

if not os.path.exists("models/best_classifier.pth"):
    gdown.download(id="1jK6PQ1W7Jnzm5PYpfy9-2H3B3GvohCRO", output="models/best_classifier.pth", quiet=False)

if not os.path.exists("models/best.pt"):
    gdown.download(id="1x1mmhxMrczhlTgAspARhtHIAbak6yR8M", output="models/best.pt", quiet=False)

# -------- LOAD CLASSIFIER --------
classifier = models.resnet18(pretrained=False)
classifier.fc = nn.Linear(classifier.fc.in_features, 2)

state_dict = torch.load("models/best_classifier.pth", map_location=device)
classifier.load_state_dict(state_dict)
classifier.to(device)
classifier.eval()

# -------- LOAD DETECTOR --------
detector = YOLO("models/best.pt")

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------- PIPELINE --------
def run_pipeline(image):
    # Classification
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classifier(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = "Abnormal" if pred.item() == 1 else "Normal"

    # If normal → return
    if label == "Normal":
        return {
            "label": label,
            "confidence": float(conf.item()),
            "image": image
        }

    # -------- DETECTION FIX --------
    # Save temp image (IMPORTANT FIX)
    # Save temporary file losslessly
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Run YOLO with lower confidence threshold and matching training resolution
    results = detector(temp_path, conf=0.0009, iou=0.7, max_det=20, imgsz=704)

    # Plot bounding boxes with smaller lines and text
    output_img = results[0].plot(conf=True, labels=True, line_width=2, font_size=10)
    
    # YOLO returns a BGR numpy array. Convert to RGB for Streamlit.
    output_img = output_img[:, :, ::-1]
    
    box_count = len(results[0].boxes)

    return {
        "label": label,
        "confidence": float(conf.item()),
        "image": output_img,
        "box_count": box_count
    }
