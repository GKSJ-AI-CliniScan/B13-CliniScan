import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from utils.load_models import get_models
from utils.gradcam import generate_gradcam

# ---------------------------
# Load models
# ---------------------------
yolo_model = YOLO("models/best_model.pt")
_, classification_model = get_models()

CLASS_NAMES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural Thickening","Hernia","No Finding"
]

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(image):
    image = image.resize((260, 260))
    image = np.array(image) / 255.0

    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return torch.tensor(image, dtype=torch.float32)

# ---------------------------
# Draw boxes (FINAL CLEAN VERSION)
# ---------------------------
def draw_boxes(image_np, boxes, min_conf=0.1, max_boxes=2):
    img = image_np.copy()

    if boxes is None or len(boxes) == 0:
        return Image.fromarray(img), False

    box_list = []
    for box in boxes:
        conf = float(box.conf)
        xyxy = box.xyxy[0].cpu().numpy()
        box_list.append((conf, xyxy))

    # Sort by confidence
    box_list = sorted(box_list, key=lambda x: x[0], reverse=True)

    # Keep only top boxes
    box_list = box_list[:max_boxes]

    drawn = False

    for i, (conf, xyxy) in enumerate(box_list):
        x1, y1, x2, y2 = map(int, xyxy)

        # Skip very low confidence (except first box)
        if conf < min_conf and i != 0:
            continue

        # ✅ Draw ONLY bounding box (no misleading label)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        drawn = True

    return Image.fromarray(img), drawn

# ---------------------------
# Main pipeline
# ---------------------------
def run_models(pil_image):
    image_np = np.array(pil_image).astype(np.uint8)

    # ---------------------------
    # 1. Classification
    # ---------------------------
    input_tensor = preprocess(pil_image)

    with torch.no_grad():
        output = classification_model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

        class_result = CLASS_NAMES[pred.item()]
        confidence_score = confidence.item()

    # ---------------------------
    # 2. YOLO Detection
    # ---------------------------
    results = yolo_model.predict(
        source=image_np,
        conf=0.01,   # Low threshold to capture weak detections
        iou=0.3,
        max_det=5,
        save=False,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    print("Total boxes detected:", 0 if boxes is None else len(boxes))

    detection_image, drawn = draw_boxes(image_np, boxes)

    no_detection = not drawn

    # ---------------------------
    # 3. Grad-CAM
    # ---------------------------
    try:
        target_layer = classification_model.features[-1]

        gradcam_image = generate_gradcam(
            classification_model,
            pil_image,
            target_layer
        )
    except Exception as e:
        print("Grad-CAM error:", e)
        gradcam_image = pil_image

    # ---------------------------
    # 4. Smart Medical Note (FINAL)
    # ---------------------------
    if confidence_score < 0.5:
        note = "Low confidence prediction – interpret with caution."

    elif no_detection and class_result != "No Finding":
        note = "Diffuse abnormality detected. Region highlighted using Grad-CAM."

    elif no_detection:
        note = "No abnormalities detected."

    else:
        note = "Localized region detected (approximate). Refer Grad-CAM for precise localization."

    # ---------------------------
    # Return results
    # ---------------------------
    return {
        "classification": class_result,
        "confidence": confidence_score,
        "detection_image": detection_image,
        "gradcam_image": gradcam_image,
        "no_detection": no_detection,
        "note": note
    }