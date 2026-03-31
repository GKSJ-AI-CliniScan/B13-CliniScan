from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # NEW: Added for Cloud Deployment
from ultralytics import YOLO
import shutil
import os
import io

# RESNET IMPORTS
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🧠 Loading AI Ensemble Pipeline...")

# --- 1. LOAD THE SPECIALIST (YOLOv8) ---
try:
    yolo_model = YOLO('models/best.pt') # Ensure this path points to your YOLO weights
    print("✅ Specialist (YOLOv8) Loaded!")
except Exception as e:
    print(f"❌ Error loading YOLOv8: {e}")

# --- 2. LOAD THE CHIEF DOCTOR (ResNet18) ---
try:
    # Set device (CPU is safest for web servers unless configured otherwise)
    device = torch.device("cpu")
    
    # Build the ResNet18 skeleton
    resnet_model = models.resnet18(weights=None)
    resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)
    
    # Load your custom weights
    resnet_model.load_state_dict(torch.load('models/resnet_medical.pth', map_location=device))
    resnet_model = resnet_model.to(device)
    resnet_model.eval() # Set to evaluation mode

    # Define the exact preprocessing recipe from your gradcam script
    resnet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("✅ Chief Doctor (ResNet18) Loaded!")
except Exception as e:
    print(f"❌ Error loading ResNet: {e}")


@app.get("/health")
def health_check():
    return {"status": "AI-CliniScan Dual-Engine is online!"}

@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    # 1. Save file temporarily for YOLO
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = f"temp_uploads/{file.filename}"
    
    # Read file into memory for ResNet AND save for YOLO
    file_bytes = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(file_bytes)

    # ==========================================
    # 🧠 CHIEF DOCTOR (RESNET) VERDICT
    # ==========================================
    try:
        # Prepare image
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        input_tensor = resnet_transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = resnet_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get the winning class and confidence
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Assuming Class 1 is "Abnormal/Sick" and Class 0 is "Normal" based on your code
            class_idx = predicted_class.item()
            chief_diagnosis = "ABNORMAL" if class_idx == 1 else "NORMAL"
            chief_confidence = round(confidence.item() * 100, 2)
            
    except Exception as e:
        print(f"ResNet Error: {e}")
        chief_diagnosis = "UNKNOWN"
        chief_confidence = 0.0

    # ==========================================
    # 🔍 SPECIALIST (YOLO) FINDINGS
    # ==========================================
    yolo_results = yolo_model(file_path, conf=0.15)
    best_findings = {}

    for r in yolo_results:
        for box in r.boxes:
            pathology = yolo_model.names[int(box.cls)]
            yolo_conf = round(float(box.conf) * 100, 2)
            coords = box.xyxy[0].tolist()

            if pathology not in best_findings or yolo_conf > best_findings[pathology]["confidence"]:
                best_findings[pathology] = {
                    "pathology": pathology,
                    "confidence": yolo_conf,
                    "bounding_box": coords
                }
    
    filtered_results = list(best_findings.values())

    # ==========================================
    # 📦 SEND COMBINED DATA TO WEBSITE
    # ==========================================
    return {
        "filename": file.filename,
        "status": "success",
        "chief_verdict": {
            "diagnosis": chief_diagnosis,
            "confidence": chief_confidence
        },
        "specialist_findings": filtered_results
    }

# ==========================================
# 🌐 CLOUD DEPLOYMENT: SERVE FRONTEND
# ==========================================
# This mounts your Frontend folder so Hugging Face can display your website
app.mount("/", StaticFiles(directory="Frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Changed port to 7860 for Hugging Face Spaces compatibility
    uvicorn.run(app, host="0.0.0.0", port=7860)