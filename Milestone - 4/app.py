from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import io
import base64
import torch.nn.functional as F

app = FastAPI()

# ------------------ Load Models ------------------
device = torch.device("cpu")

# YOLO
yolo_model = YOLO("best.pt")

# Classification
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 14)
)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

classes = [
"Aortic enlargement","Atelectasis","Calcification","Cardiomegaly",
"Consolidation","ILD","Infiltration","Lung Opacity",
"Nodule/Mass","Other lesion","Pleural effusion",
"Pleural thickening","Pneumothorax","Pulmonary fibrosis"
]

# ------------------ Routes ------------------

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Convert original image to base64
    orig_buffer = io.BytesIO()
    image.save(orig_buffer, format="PNG")
    original_str = base64.b64encode(orig_buffer.getvalue()).decode()

    # YOLO
    results = yolo_model(image)
    annotated = results[0].plot()

    # Convert annotated image to base64
    img = Image.fromarray(annotated)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    annotated_str = base64.b64encode(buffer.getvalue()).decode()

    # Classification
    img_tensor = transform(image).unsqueeze(0)
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)

    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probs, 3, dim=1)
    top_predictions = [
        {
            "class": classes[top_indices[0][i].item()],
            "confidence": float(top_probs[0][i].item()) * 100
        }
        for i in range(3)
    ]

    return {
        "original_image": f"data:image/png;base64,{original_str}",
        "annotated_image": f"data:image/png;base64,{annotated_str}",
        "top_predictions": top_predictions
    }