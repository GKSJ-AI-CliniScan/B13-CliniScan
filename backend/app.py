import os
import io
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import models
from classification.model import CliniScanClassifier
from detection.model import get_detection_model
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = FastAPI(title="MediScanAI API")

# Allow CORS for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to the Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cpu') # Hugging Face Free Tier uses CPU

# Global model variables
class_model = None
detect_model = None

# Class names
CLASS_NAMES_CLASSIFY = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", 
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", 
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening", 
    "Pneumothorax", "Pulmonary fibrosis", "No finding"
]

CLASS_NAMES_DETECT = [
    "Background", "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", 
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", 
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening", 
    "Pneumothorax", "Pulmonary fibrosis", "No finding"
]

def get_inference_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

@app.on_event("startup")
async def load_models():
    global class_model, detect_model
    try:
        print("Loading Classification Model...")
        class_model = CliniScanClassifier(15)
        class_model.load_state_dict(torch.load('models/best_resnet_classification.pth', map_location=device))
        class_model.to(device)
        class_model.eval()
        print("Classification Model Loaded.")

        print("Loading Detection Model...")
        detect_model = get_detection_model(16)
        detect_model.load_state_dict(torch.load('models/best_faster_rcnn_detection.pth', map_location=device))
        detect_model.to(device)
        detect_model.eval()
        print("Detection Model Loaded.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/")
def read_root():
    return {"status": "MediScanAI API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), mode: str = Form("classify")):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # OpenCV format

    if mode == "classify":
        transform = get_inference_transforms()
        transformed = transform(image=image_bgr)
        image_tensor = transformed['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = class_model(image_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        results = []
        for i, prob in enumerate(probs):
            if prob > 0.1:  # Lower threshold for frontend to filter
                # Convert names to match frontend (e.g., "Pleural effusion" -> "Effusion")
                # We will map standard names to match the frontend expectations
                name = CLASS_NAMES_CLASSIFY[i]
                if name == "Pleural effusion": name = "Effusion"
                elif name == "Nodule/Mass": name = "Mass"
                elif name == "Pleural thickening": name = "Pleural_Thickening"

                results.append({"name": name, "score": float(prob)})
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return {"mode": "classify", "predictions": results}

    elif mode == "detect":
        orig_h, orig_w = image_np.shape[:2]
        image_resized = cv2.resize(image_np, (256, 256))
        image_tensor = torch.as_tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = detect_model(image_tensor)[0]

        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        # Scale boxes back to 0-100 percentage for the frontend SVG viewBox
        boxes_pct = boxes.copy()
        boxes_pct[:, [0, 2]] = (boxes_pct[:, [0, 2]] / 256.0) * 100
        boxes_pct[:, [1, 3]] = (boxes_pct[:, [1, 3]] / 256.0) * 100

        results = []
        for i in range(len(scores)):
            if scores[i] > 0.1: # Return lower bounds, let frontend threshold filter
                name = CLASS_NAMES_DETECT[labels[i]]
                if name == "Pleural effusion": name = "Effusion"
                elif name == "Nodule/Mass": name = "Mass"
                elif name == "Pleural thickening": name = "Pleural_Thickening"

                box = boxes_pct[i].tolist()
                results.append({
                    "name": name,
                    "score": float(scores[i]),
                    "box": {
                        "xMin": box[0], "yMin": box[1], "xMax": box[2], "yMax": box[3]
                    }
                })

        # Deduplicate overlapping boxes for the same class (NMS simplified)
        # Not strictly necessary if the model already runs NMS, but good for cleanup
        results.sort(key=lambda x: x['score'], reverse=True)
        return {"mode": "detect", "predictions": results}

    return {"error": "Invalid mode"}
