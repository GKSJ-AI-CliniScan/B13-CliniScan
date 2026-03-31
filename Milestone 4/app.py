"""
app.py  –  Flask backend for ChestAI
Connects real classification + detection models to the frontend.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import base64
import io
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import numpy as np

app = Flask(__name__)
CORS(app)

# ── Folders ──────────────────────────────────────────────────────────────
UPLOAD_FOLDER  = "uploads"
RESULTS_FOLDER = "results"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

HISTORY_FILE = os.path.join(RESULTS_FOLDER, "history.json")

# ── Load AI models ────────────────────────────────────────────────────────
analyzer = None
try:
    from inference import ChestXrayAnalyzer
    analyzer = ChestXrayAnalyzer()
    print("✅ AI models loaded successfully.")
except Exception as e:
    print(f"⚠  Model load failed: {e}. Running in demo mode.")


# ── History helpers ───────────────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_to_history(filename, results, annotated_b64=None):
    history = load_history()
    entry = {
        "id":            len(history) + 1,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename":      filename,
        "summary":       results["summary"],
        "classification": results["classification"],
        "annotated_b64": annotated_b64,   # base64 PNG with boxes drawn
    }
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    return entry


# ── Draw bounding boxes on image ──────────────────────────────────────────
# Color palette per class (cycles if > len(COLORS))
COLORS = [
    "#FF4444", "#FF8800", "#FFD700", "#44FF88",
    "#44DDFF", "#8844FF", "#FF44CC", "#00FF99",
    "#FF6644", "#AAFFAA", "#FF99DD", "#44AAFF",
    "#FFB344", "#DD44FF", "#55FFDD",
]

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis", "No finding",
]

# Map class name → consistent color
CLASS_COLOR_MAP = {
    name: COLORS[i % len(COLORS)] for i, name in enumerate(CLASS_NAMES)
}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def draw_boxes_on_image(image_path: str, detections: list, classification: list) -> str:
    """
    Draw top-K clean bounding boxes on the image.
    Returns base64-encoded PNG string.

    detections  : list of {"box": [x1,y1,x2,y2], "confidence": float, "label": str}
    classification: list of {"class": str, "confidence": float}
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    draw = ImageDraw.Draw(img, "RGBA")

    # --- Limit to top-3 highest-confidence boxes to keep it clean ---
    top_dets = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)[:3]

    # If detector gave no boxes but classifier found findings, synthesise
    # one rough region box per top classification finding using a heuristic
    # (centre-weighted region ~ where pathology typically appears).
    if not top_dets and classification:
        top_cls = [c for c in classification if c["class"] != "No finding"][:2]
        # Rough anatomical heuristics (fraction of 224×224)
        REGION = {
            "Cardiomegaly":       (0.25, 0.40, 0.75, 0.80),
            "Aortic enlargement": (0.35, 0.20, 0.70, 0.55),
            "Pleural effusion":   (0.05, 0.50, 0.95, 0.95),
            "Pleural thickening": (0.05, 0.30, 0.95, 0.75),
            "Pulmonary fibrosis": (0.10, 0.25, 0.90, 0.80),
            "Lung Opacity":       (0.15, 0.20, 0.85, 0.75),
            "Consolidation":      (0.20, 0.25, 0.80, 0.70),
            "Pneumothorax":       (0.55, 0.05, 0.95, 0.60),
            "Nodule/Mass":        (0.35, 0.30, 0.65, 0.60),
            "Infiltration":       (0.10, 0.20, 0.90, 0.75),
            "Atelectasis":        (0.10, 0.55, 0.55, 0.90),
            "ILD":                (0.10, 0.20, 0.90, 0.80),
            "Calcification":      (0.30, 0.25, 0.70, 0.65),
        }
        for cls_item in top_cls:
            name = cls_item["class"]
            frac = REGION.get(name, (0.15, 0.15, 0.85, 0.85))
            top_dets.append({
                "box":        [frac[0]*W, frac[1]*H, frac[2]*W, frac[3]*H],
                "confidence": cls_item["confidence"],
                "label":      name,
            })

    # Draw each box
    for det in top_dets:
        x1, y1, x2, y2 = [float(v) for v in det["box"]]
        label     = det.get("label", "Finding")
        conf      = det.get("confidence", 0.0)
        color_hex = CLASS_COLOR_MAP.get(label, "#FF4444")
        rgb       = hex_to_rgb(color_hex)

        # Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2],
                       fill=(*rgb, 30),
                       outline=(*rgb, 230),
                       width=3)

        # Label badge background
        label_text = f"{label}  {conf*100:.0f}%"
        font_size  = max(12, int(min(W, H) * 0.045))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        bbox_text = draw.textbbox((x1 + 6, y1 + 4), label_text, font=font)
        pad = 4
        draw.rectangle(
            [bbox_text[0]-pad, bbox_text[1]-pad, bbox_text[2]+pad, bbox_text[3]+pad],
            fill=(*rgb, 200)
        )
        draw.text((x1 + 6, y1 + 4), label_text, fill="white", font=font)

    # Encode to base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "chest_xray_analyzer.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ── Real inference ────────────────────────────────────────────────
    if analyzer is not None:
        try:
            results = analyzer.analyze(filepath)
        except Exception as e:
            return jsonify({"error": f"Inference failed: {e}"}), 500
    else:
        # Demo fallback when models aren't loaded
        results = _demo_results()

    # ── Attach labels to detection boxes ─────────────────────────────
    # The detector returns {"box":..., "confidence":...}
    # Enrich with label from classification if missing
    detections    = results.get("detection", [])
    classification = results.get("classification", [])

    # Try to attach class labels to detection boxes if not present
    if detections and classification:
        for i, det in enumerate(detections):
            if "label" not in det and i < len(classification):
                det["label"] = classification[i]["class"]

    # ── Draw annotated image ──────────────────────────────────────────
    try:
        annotated_b64 = draw_boxes_on_image(filepath, detections, classification)
    except Exception as e:
        print(f"[Draw] Warning: {e}")
        annotated_b64 = None

    results["annotated_b64"] = annotated_b64

    # ── Persist ───────────────────────────────────────────────────────
    save_to_history(filename, results, annotated_b64)
    return jsonify(results)


@app.route("/api/stats", methods=["GET"])
def get_stats():
    history = load_history()
    if not history:
        return jsonify({"total": 0, "pos_rate": 0, "avg_conf": 0, "avg_time": 0})

    total     = len(history)
    positives = sum(1 for x in history if x["summary"].get("status") == "Positive")
    avg_conf  = sum(x["summary"].get("avg_confidence", 0) for x in history) / total

    return jsonify({
        "total":    total,
        "pos_rate": round((positives / total) * 100, 1),
        "avg_conf": round(avg_conf * 100, 1),
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    history = load_history()
    # Strip large base64 blobs from list view
    slim = []
    for h in reversed(history):
        entry = {k: v for k, v in h.items() if k != "annotated_b64"}
        slim.append(entry)
    return jsonify(slim)


@app.route("/uploads/<filename>")
def serve_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ── Demo results (models not loaded) ─────────────────────────────────────
def _demo_results():
    return {
        "classification": [
            {"class": "Pulmonary fibrosis", "confidence": 0.77},
            {"class": "Pleural thickening", "confidence": 0.62},
            {"class": "Aortic enlargement", "confidence": 0.51},
        ],
        "detection": [],
        "summary": {
            "status":         "Positive",
            "findings":       ["Pulmonary fibrosis", "Pleural thickening", "Aortic enlargement"],
            "avg_confidence": 0.77,
        },
    }


if __name__ == "__main__":
    app.run(debug=True, port=5000)