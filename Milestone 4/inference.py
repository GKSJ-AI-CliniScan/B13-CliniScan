"""
inference.py  –  ChestAI inference pipeline
Runs classification (EfficientNet-B0) + detection (Faster R-CNN)
and returns results ready for the frontend.
"""

import os
import traceback

import numpy as np
import torch
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import get_model
from model_detection import get_detection_model

# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════
class Config:
    CLASSIFICATION_MODEL = "best_model.pth"
    DETECTION_MODEL      = "best_detection_model.pth"
    DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE             = 224
    CLS_THRESHOLD        = 0.40   # multi-label threshold for classification
    DET_THRESHOLD        = 0.35   # confidence threshold for detections
    DET_NMS_IOU          = 0.50   # NMS IoU threshold
    MAX_DET_BOXES        = 3      # return at most N boxes (keep it clean)

config = Config()

# ── Albumentations transform (classification) ─────────────────────────
_cls_transform = A.Compose([
    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ── Class names ────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis", "No finding",
]

# id_to_class for the detection model (0 = background, 1-15 = pathologies)
DETECTION_ID_TO_CLASS = {0: "background"}
DETECTION_ID_TO_CLASS.update({
    i + 1: name for i, name in enumerate(CLASS_NAMES[:-1])  # exclude "No finding"
})


# ══════════════════════════════════════════════════════════════════════
# Classification
# ══════════════════════════════════════════════════════════════════════
class ChestXrayClassifier:
    def __init__(self):
        self.model = None
        candidates = [
            config.CLASSIFICATION_MODEL,
            "classify_model.pth",
            "best_model_advanced.pth",
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)

        if path:
            self._load(path)
        else:
            print(f"❌ Classification model not found. Searched: {candidates}")

    def _load(self, path: str):
        try:
            self.model = get_model(len(CLASS_NAMES)).to(config.DEVICE)
            ckpt = torch.load(path, map_location=config.DEVICE)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            self.model.load_state_dict(ckpt, strict=False)
            self.model.eval()
            print(f"✅ Classification model loaded: {path}")
        except Exception as e:
            print(f"❌ Classification load error: {e}")
            traceback.print_exc()
            self.model = None

    def predict(self, image_path: str) -> list[dict]:
        """Returns list of {class, confidence} sorted by confidence desc."""
        if self.model is None:
            return [{"class": "Model not loaded", "confidence": 0.0}]

        img    = Image.open(image_path).convert("RGB")
        arr    = np.array(img)
        tensor = _cls_transform(image=arr)["image"].unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]

        results = [
            {"class": CLASS_NAMES[i], "confidence": float(p)}
            for i, p in enumerate(probs)
            if p >= config.CLS_THRESHOLD
        ]

        if not results:
            # Fallback: always return the top-1 class
            best = int(np.argmax(probs))
            results = [{"class": CLASS_NAMES[best], "confidence": float(probs[best])}]

        return sorted(results, key=lambda x: x["confidence"], reverse=True)


# ══════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════
class ChestXrayDetector:
    def __init__(self):
        self.model = None
        candidates = [
            config.DETECTION_MODEL,
            "checkpoints_detection/best_detection_model.pth",
            "detection_model_final.pth",
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)

        if path:
            self._load(path)
        else:
            print(f"⚠  Detection model not found. Searched: {candidates}. Skipping detection.")

    def _load(self, path: str):
        try:
            self.model = get_detection_model(num_classes=16, backbone="mobilenet").to(config.DEVICE)
            ckpt = torch.load(path, map_location=config.DEVICE)
            if isinstance(ckpt, dict) and "model" in ckpt:
                self.model.load_state_dict(ckpt["model"])
            else:
                self.model.load_state_dict(ckpt)
            self.model.eval()
            print(f"✅ Detection model loaded: {path}")
        except Exception as e:
            print(f"❌ Detection load error: {e}. Detection disabled.")
            self.model = None

    def predict(self, image_path: str) -> list[dict]:
        """
        Returns list of:
          {"box": [x1,y1,x2,y2], "confidence": float, "label": str}
        Boxes are in pixel coordinates of the original image.
        """
        if self.model is None:
            return []

        try:
            img = Image.open(image_path).convert("RGB")
            W, H = img.size

            # Resize to model input size, keeping scale factors for back-projection
            img_resized = img.resize((config.IMG_SIZE, config.IMG_SIZE))
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float().to(config.DEVICE)

            with torch.no_grad():
                preds = self.model([tensor])[0]

            boxes  = preds["boxes"].cpu()
            scores = preds["scores"].cpu()
            labels = preds["labels"].cpu()

            if boxes.shape[0] == 0:
                return []

            # ── NMS ────────────────────────────────────────────────────
            from torchvision.ops import nms
            keep   = nms(boxes, scores, iou_threshold=config.DET_NMS_IOU)
            boxes  = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # ── Score threshold ────────────────────────────────────────
            mask   = scores >= config.DET_THRESHOLD
            boxes  = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            if boxes.shape[0] == 0:
                return []

            # ── Top-K (keep it clean in the UI) ───────────────────────
            topk  = min(config.MAX_DET_BOXES, boxes.shape[0])
            boxes  = boxes[:topk]
            scores = scores[:topk]
            labels = labels[:topk]

            # ── Back-project to original image coordinates ─────────────
            scale_x = W / config.IMG_SIZE
            scale_y = H / config.IMG_SIZE

            results = []
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "box": [
                        x1 * scale_x,
                        y1 * scale_y,
                        x2 * scale_x,
                        y2 * scale_y,
                    ],
                    "confidence": float(score),
                    "label": DETECTION_ID_TO_CLASS.get(int(label), f"class_{int(label)}"),
                })

            return results

        except Exception as e:
            print(f"⚠  Detection predict error: {e}")
            traceback.print_exc()
            return []


# ══════════════════════════════════════════════════════════════════════
# Main Analyzer
# ══════════════════════════════════════════════════════════════════════
class ChestXrayAnalyzer:
    def __init__(self):
        print("Initializing ChestXrayAnalyzer …")
        self.classifier = ChestXrayClassifier()
        self.detector   = ChestXrayDetector()

        if self.classifier.model is None:
            raise RuntimeError(
                "Classification model failed to load. Cannot start analyzer."
            )

    def analyze(self, image_path: str) -> dict:
        """
        Returns:
        {
          "classification": [{"class": str, "confidence": float}, ...],
          "detection":      [{"box": [x1,y1,x2,y2], "confidence": float, "label": str}, ...],
          "summary": {
              "status":         "Positive" | "Negative",
              "findings":       [str, ...],
              "avg_confidence": float,
          }
        }
        """
        classification = self.classifier.predict(image_path)
        detection      = self.detector.predict(image_path)

        findings   = [c["class"] for c in classification]
        is_finding = findings[0] != "No finding" if findings else False
        top_conf   = classification[0]["confidence"] if classification else 0.0

        summary = {
            "status":         "Positive" if is_finding else "Negative",
            "findings":       findings,
            "avg_confidence": top_conf,
        }

        return {
            "classification": classification,
            "detection":      detection,
            "summary":        summary,
        }