from __future__ import annotations

import gc
import io
import os
import sys
import base64
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image

def _pip(pkg: str) -> None:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    _pip("albumentations")
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

try:
    from ultralytics import YOLO
except ImportError:
    _pip("ultralytics")
    from ultralytics import YOLO

try:
    from fpdf import FPDF
except ImportError:
    _pip("fpdf2")
    from fpdf import FPDF

BASE       = Path(__file__).parent
CLF_PATH   = BASE / "best_efficientnet_b3.pth"
YOLO_PATH  = BASE / "best.pt"

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis",
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = 384
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN_AUC    = 91.93

PER_CLASS_AUC: dict[str, float] = {
    "Aortic enlargement": 96.33,
    "Atelectasis":        93.62,
    "Calcification":      89.31,
    "Cardiomegaly":       96.04,
    "Consolidation":      92.03,
    "ILD":                89.24,
    "Infiltration":       92.73,
    "Lung Opacity":       94.39,
    "Nodule/Mass":        92.80,
    "Other lesion":       89.78,
    "Pleural effusion":   90.79,
    "Pleural thickening": 92.53,
    "Pneumothorax":       84.46,
    "Pulmonary fibrosis": 92.93,
}

PALETTE = [
    (45,212,191), (251,146,60),  (251,191,36), (244,114,182), (167,139,250),
    (52,211,153), (110,231,183), (252,165,165),(203,213,225), (148,163,184),
    (96,165,250), (74,222,128),  (190,242,100),(253,164,175),
]

val_tfm = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

print(f"[CliniScan] Device: {DEVICE}")

clf_model:  nn.Module | None = None
yolo_model: YOLO | None      = None

def load_classifier() -> None:
    global clf_model
    try:
        m     = models.efficientnet_b3(weights=None)
        in_f  = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),      nn.Linear(in_f, 512),
            nn.ReLU(inplace=True),nn.Dropout(0.15),
            nn.Linear(512, NUM_CLASSES),
        )
        if CLF_PATH.exists():
            ckpt  = torch.load(CLF_PATH, map_location=DEVICE, weights_only=False)
            state = ckpt.get("model_state", ckpt)
            m.load_state_dict(state)
            print("[CliniScan] EfficientNet-B3 loaded from checkpoint.")
        else:
            print(f"[CliniScan] WARNING: Model not found at {CLF_PATH} — running in DEMO mode.")
        clf_model = m.eval().to(DEVICE)
    except Exception as exc:
        print(f"[CliniScan] Classifier load error: {exc}")

def load_detector() -> None:
    global yolo_model
    try:
        if YOLO_PATH.exists():
            yolo_model = YOLO(str(YOLO_PATH))
            print("[CliniScan] YOLOv8m loaded from checkpoint.")
        else:
            print(f"[CliniScan] WARNING: YOLO weights not found — detection disabled.")
    except Exception as exc:
        print(f"[CliniScan] YOLO load error: {exc}")

load_classifier()
load_detector()

class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self.grads: torch.Tensor | None = None
        self.acts:  torch.Tensor | None = None
        layer.register_forward_hook(lambda _m, _i, o: setattr(self, "acts", o.detach()))
        layer.register_full_backward_hook(lambda _m, _gi, go: setattr(self, "grads", go[0].detach()))

    def generate(self, tensor: torch.Tensor, cls_idx: int) -> np.ndarray:
        self.model.eval()
        out = self.model(tensor)
        self.model.zero_grad()
        out[0, cls_idx].backward()
        w   = self.grads[0].mean(dim=(1, 2))
        cam = (w[:, None, None] * self.acts[0]).sum(0)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy()

def to_rgb(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def classify(rgb: np.ndarray) -> list[tuple[str, float]]:
    if clf_model is None:
        import random; random.seed(42)
        return sorted([(c, round(random.uniform(.05, .85), 4)) for c in CLASS_NAMES], key=lambda x: -x[1])
    t = val_tfm(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(clf_model(t)).squeeze().cpu().numpy()
    return sorted(
        [(CLASS_NAMES[i], float(round(float(probs[i]), 4))) for i in range(NUM_CLASSES)],
        key=lambda x: -x[1],
    )

def detect(rgb: np.ndarray, conf: float = 0.15, iou: float = 0.45) -> tuple[np.ndarray, list[dict]]:
    if yolo_model is None:
        return rgb.copy(), []
    try:
        tmp = tempfile.mktemp(suffix=".png")
        cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        res = yolo_model.predict(tmp, conf=conf, iou=iou, verbose=False)[0]
        det_img = cv2.resize(rgb, (640, 640)).copy()
        boxes: list[dict] = []
        if res.boxes and len(res.boxes) > 0:
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cid   = int(b.cls[0])
                cf    = float(b.conf[0])
                name  = CLASS_NAMES[cid] if cid < NUM_CLASSES else f"cls{cid}"
                color = PALETTE[cid % len(PALETTE)]
                cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 3)
                label = f"{name} {cf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)
                cv2.rectangle(det_img, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
                cv2.putText(det_img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
                boxes.append({"class": name, "conf": round(cf, 4), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return det_img, boxes
    except Exception as exc:
        print(f"[CliniScan] Detection error: {exc}")
        return rgb.copy(), []

def compute_gradcam(rgb: np.ndarray, preds: list[tuple[str, float]]) -> tuple[np.ndarray | None, str | None]:
    if clf_model is None:
        return None, None
    try:
        top_idx = int(np.argmax([p for _, p in preds]))
        gc_obj  = GradCAM(clf_model, clf_model.features[-1])
        t = val_tfm(image=rgb)["image"].unsqueeze(0).to(DEVICE)
        t.requires_grad_(True)
        cam = gc_obj.generate(t, top_idx)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        hm  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        hm  = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        i384    = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        overlay = (0.6 * i384 + 0.4 * hm).astype(np.uint8)
        return overlay, CLASS_NAMES[top_idx]
    except Exception as exc:
        print(f"[CliniScan] Grad-CAM error: {exc}")
        return None, None

def ndarray_to_b64(img: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode image.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

def build_pdf(preds, filename, boxes, elapsed, det_img=None, gcam_img=None,
              patient_name="", patient_age="", patient_phone="") -> bytes:
    def save_tmp_img(arr):
        path = tempfile.mktemp(suffix=".png")
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return path

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_fill_color(13, 148, 136)
    pdf.rect(0, 0, 210, 36, "F")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 8);  pdf.cell(0, 10, "CliniScan AI", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 20); pdf.cell(0, 7, "Chest X-Ray Abnormality Detection Report", ln=True)
    pdf.set_xy(10, 27); pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(167, 243, 208)
    pdf.cell(0, 6, "EfficientNet-B3 + YOLOv8m  |  VinDr-CXR  |  Research Use Only", ln=True)

    pdf.set_text_color(40, 40, 40)
    pdf.set_xy(10, 42); pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6,
             f"File: {filename}   "
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}   "
             f"Time: {elapsed:.1f}s   "
             f"Device: {str(DEVICE).upper()}", ln=True)
    pdf.set_draw_color(13, 148, 136); pdf.set_line_width(.4)
    pdf.line(10, 50, 200, 50)

    # Patient Details Block
    y_pt = 53
    has_patient = any([patient_name, patient_age, patient_phone])
    if has_patient:
        pdf.set_fill_color(230, 255, 250)
        pdf.rect(10, y_pt, 190, 22, "F")
        pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(13, 148, 136)
        pdf.set_xy(14, y_pt + 2); pdf.cell(0, 5, "PATIENT INFORMATION", ln=True)
        pdf.set_font("Helvetica", "", 9); pdf.set_text_color(50, 50, 50)
        pdf.set_xy(14, y_pt + 9)
        if patient_name:  pdf.cell(65, 5, f"Name: {patient_name}", border=0)
        if patient_age:   pdf.cell(50, 5, f"Age: {patient_age}", border=0)
        if patient_phone: pdf.cell(0,  5, f"Phone: {patient_phone}", border=0, ln=True)
        pdf.line(10, y_pt + 24, 200, y_pt + 24)
        y_pt += 26

    sorted_p  = sorted(preds, key=lambda x: -x[1])
    positives = [(n, c) for n, c in sorted_p if c >= .5]
    top_n, top_c = sorted_p[0] if sorted_p else ("N/A", 0.0)

    pdf.set_fill_color(240, 253, 250)
    pdf.rect(10, y_pt, 190, 22, "F")
    pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(13, 148, 136)
    pdf.set_xy(14, y_pt + 2); pdf.cell(0, 5, "SUMMARY", ln=True)
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(50, 50, 50)
    pdf.set_xy(14, y_pt + 8)
    pdf.cell(58, 5, f"Detections: {len(boxes)}", border=0)
    pdf.cell(62, 5, f"Positive Classes: {len(positives)}", border=0)
    pdf.cell(0,  5, f"Top: {top_n} ({top_c*100:.1f}%)", border=0, ln=True)
    pdf.set_xy(14, y_pt + 14)
    pdf.cell(0, 5, f"Mean AUC: {MEAN_AUC:.2f}%  |  Model: EfficientNet-B3  |  Dataset: VinDr-CXR", ln=True)
    pdf.line(10, y_pt + 25, 200, y_pt + 25)

    y = y_pt + 29
    img_paths = []
    if det_img is not None: img_paths.append(save_tmp_img(det_img))
    if gcam_img is not None: img_paths.append(save_tmp_img(gcam_img))

    if img_paths:
        pdf.set_xy(10, y); pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(13, 148, 136)
        pdf.cell(0, 7, "VISUAL ANALYSIS", ln=True)
        pdf.line(10, y + 8, 200, y + 8); y += 12
        img_w = 88
        labels = ["Detection Results (YOLOv8m)", "Grad-CAM Heatmap"]
        for idx, path in enumerate(img_paths[:2]):
            x_pos = 10 + idx * (img_w + 10)
            pdf.image(path, x=x_pos, y=y, w=img_w)
            pdf.set_xy(x_pos, y + img_w + 1)
            pdf.set_font("Helvetica", "I", 7); pdf.set_text_color(100, 100, 100)
            pdf.cell(img_w, 5, labels[idx], align="C")
        y += img_w + 10
        if y > 260: pdf.add_page(); y = 20

    pdf.set_xy(10, y); pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(13, 148, 136)
    pdf.cell(0, 7, "CLASSIFICATION RESULTS", ln=True)
    pdf.line(10, y + 8, 200, y + 8); y += 11

    pdf.set_fill_color(13, 148, 136); pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 8); pdf.set_xy(10, y)
    pdf.cell(72, 6, "Disease Class",  fill=True, border=0)
    pdf.cell(32, 6, "Probability",    fill=True, border=0)
    pdf.cell(30, 6, "Status",         fill=True, border=0)
    pdf.cell(0,  6, "Model AUC",      fill=True, border=0, ln=True)
    y += 7

    for ri, (name, cv_) in enumerate(sorted_p):
        is_pos = cv_ >= .5
        pdf.set_fill_color(240, 253, 250) if ri % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(180, 0, 0) if is_pos else pdf.set_text_color(50, 50, 50)
        pdf.set_font("Helvetica", "B" if is_pos else "", 8)
        pdf.set_xy(10, y)
        pdf.cell(72, 5.5, name,                             fill=True, border=0)
        pdf.cell(32, 5.5, f"{cv_*100:.2f}%",               fill=True, border=0)
        pdf.cell(30, 5.5, "POSITIVE" if is_pos else "-",   fill=True, border=0)
        auc_val = PER_CLASS_AUC.get(name, 0)
        pdf.cell(0,  5.5, f"{auc_val:.2f}%",               fill=True, border=0, ln=True)
        y += 5.5
        if y > 270: pdf.add_page(); y = 20

    if boxes:
        y += 6
        if y > 240: pdf.add_page(); y = 20
        pdf.set_xy(10, y); pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(13, 148, 136)
        pdf.cell(0, 7, "DETECTION RESULTS", ln=True)
        pdf.line(10, y + 8, 200, y + 8); y += 11

        pdf.set_fill_color(13, 148, 136); pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8); pdf.set_xy(10, y)
        pdf.cell(68, 6, "Class",                       fill=True, border=0)
        pdf.cell(30, 6, "Confidence",                  fill=True, border=0)
        pdf.cell(0,  6, "Bounding Box [x1,y1,x2,y2]", fill=True, border=0, ln=True)
        y += 7

        for di, b in enumerate(boxes):
            pdf.set_fill_color(240, 253, 250) if di % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(50, 50, 50); pdf.set_font("Helvetica", "", 8)
            pdf.set_xy(10, y)
            pdf.cell(68, 5.5, b["class"],                           fill=True, border=0)
            pdf.cell(30, 5.5, f"{b['conf']*100:.1f}%",             fill=True, border=0)
            pdf.cell(0,  5.5, f"[{b['x1']},{b['y1']},{b['x2']},{b['y2']}]", fill=True, border=0, ln=True)
            y += 5.5

    y += 8
    if y > 240: pdf.add_page(); y = 20
    pdf.set_xy(10, y); pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(13, 148, 136)
    pdf.cell(0, 7, "PER-CLASS AUC (Training Results)", ln=True)
    pdf.line(10, y + 8, 200, y + 8); y += 11

    pdf.set_fill_color(13, 148, 136); pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 8); pdf.set_xy(10, y)
    pdf.cell(80, 6, "Disease Class", fill=True, border=0)
    pdf.cell(30, 6, "AUC Score",     fill=True, border=0)
    pdf.cell(0,  6, "Grade",         fill=True, border=0, ln=True)
    y += 7

    for ri, (name, auc_v) in enumerate(PER_CLASS_AUC.items()):
        grade = ("Excellent" if auc_v >= 95 else
                 "Very Good" if auc_v >= 92 else
                 "Good"      if auc_v >= 90 else "Moderate")
        pdf.set_fill_color(240, 253, 250) if ri % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(50, 50, 50); pdf.set_font("Helvetica", "", 8)
        pdf.set_xy(10, y)
        pdf.cell(80, 5.5, name,              fill=True, border=0)
        pdf.cell(30, 5.5, f"{auc_v:.2f}%",  fill=True, border=0)
        pdf.cell(0,  5.5, grade,             fill=True, border=0, ln=True)
        y += 5.5

    pdf.set_y(-18)
    pdf.set_draw_color(13, 148, 136); pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_font("Helvetica", "", 7); pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 7,
             f"CliniScan AI  |  EfficientNet-B3 + YOLOv8m  |  VinDr-CXR  |  Mean AUC {MEAN_AUC:.2f}%  |  Research Only",
             align="C")

    return bytes(pdf.output())

app = FastAPI(title="CliniScan AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HTML_PAGE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CliniScan AI — Chest X-Ray Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}

/* ══════════════ LIGHT THEME ══════════════ */
:root,[data-theme="light"]{
  --bg:#F4F6F9;--surface:#FFFFFF;--surface2:#F0F3F7;--border:#DDE2EA;
  --text:#0D1B2A;--text2:#4A5568;--text3:#8FA0B4;
  --teal:#0D9488;--teal2:#0F766E;--teal3:#CCFBF1;--teal4:#99F6E4;
  --red:#DC2626;--red2:#FEF2F2;--green:#16A34A;--green2:#F0FDF4;
  --amber:#D97706;
  --shadow:0 1px 3px rgba(0,0,0,.08),0 4px 16px rgba(0,0,0,.05);
  --shadow2:0 2px 8px rgba(0,0,0,.1),0 8px 32px rgba(0,0,0,.08);
  --hero-grad:linear-gradient(135deg,#0D9488 0%,#0F766E 50%,#065F46 100%);
  --nav-bg:#FFFFFF;--footer-bg:linear-gradient(180deg,#0D1521 0%,#091018 100%);
  --input-bg:#FFFFFF;--input-border:#DDE2EA;
  --theme-toggle-bg:#E2E8F0;--theme-icon-color:#64748B;
  --report-header-bg:linear-gradient(135deg,#0D9488,#0F766E);
  --table-even:#FAFBFC;--alert-warn-bg:#FEF2F2;--alert-warn-border:#FCA5A5;--alert-warn-text:#991B1B;
  --alert-ok-bg:#F0FDF4;--alert-ok-border:#86EFAC;--alert-ok-text:#14532D;
  --patient-block-bg:#E6FFFB;--patient-block-border:#99F6E4;
}

/* ══════════════ DARK THEME ══════════════ */
[data-theme="dark"]{
  --bg:#0A0F1A;--surface:#111827;--surface2:#1A2332;--border:#1E2D3D;
  --text:#E2E8F0;--text2:#94A3B8;--text3:#4A5568;
  --teal:#2DD4BF;--teal2:#14B8A6;--teal3:rgba(13,148,136,.15);--teal4:rgba(13,148,136,.3);
  --red:#F87171;--red2:rgba(239,68,68,.12);--green:#4ADE80;--green2:rgba(74,222,128,.12);
  --amber:#FCD34D;
  --shadow:0 1px 3px rgba(0,0,0,.3),0 4px 16px rgba(0,0,0,.2);
  --shadow2:0 2px 8px rgba(0,0,0,.4),0 8px 32px rgba(0,0,0,.3);
  --hero-grad:linear-gradient(135deg,#0A1628 0%,#0D1F38 50%,#091018 100%);
  --nav-bg:#111827;--footer-bg:linear-gradient(180deg,#060B12 0%,#040810 100%);
  --input-bg:#1A2332;--input-border:#1E2D3D;
  --theme-toggle-bg:#1A2332;--theme-icon-color:#94A3B8;
  --report-header-bg:linear-gradient(135deg,#0A1628,#0D1F38);
  --table-even:#141E2D;--alert-warn-bg:rgba(239,68,68,.1);--alert-warn-border:rgba(239,68,68,.3);--alert-warn-text:#FCA5A5;
  --alert-ok-bg:rgba(74,222,128,.1);--alert-ok-border:rgba(74,222,128,.3);--alert-ok-text:#86EFAC;
  --patient-block-bg:rgba(13,148,136,.1);--patient-block-border:rgba(13,148,136,.3);
}

body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;transition:background .3s,color .3s}
.page{display:none;min-height:100vh}.page.active{display:block}

/* ── NAV ── */
nav{background:var(--nav-bg);border-bottom:1.5px solid var(--border);padding:0 32px;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;transition:background .3s,border-color .3s}
.nav-logo{display:flex;align-items:center;gap:10px;cursor:pointer}
.nav-logo-icon{width:34px;height:34px;background:linear-gradient(135deg,#0D9488,#0F766E);border-radius:8px;display:flex;align-items:center;justify-content:center}
.nav-logo span{font-size:16px;font-weight:700;letter-spacing:-.3px;color:var(--text)}
.nav-logo em{color:var(--teal);font-style:italic}
.nav-links{display:flex;align-items:center;gap:4px}
.nav-link{padding:6px 14px;border-radius:7px;font-size:13px;font-weight:500;cursor:pointer;transition:all .15s;color:var(--text2);border:none;background:none}
.nav-link:hover{background:var(--surface2);color:var(--text)}
.nav-link.active{background:var(--teal3);color:var(--teal2);font-weight:600}
.nav-badge{background:var(--teal);color:#fff;font-size:9px;font-weight:700;padding:2px 6px;border-radius:99px;letter-spacing:.3px}

/* ── THEME TOGGLE ── */
.theme-toggle{display:flex;align-items:center;gap:8px;margin-left:8px}
.theme-btn{width:44px;height:24px;border-radius:99px;background:var(--theme-toggle-bg);border:1.5px solid var(--border);cursor:pointer;position:relative;transition:all .25s;flex-shrink:0}
.theme-btn-knob{width:18px;height:18px;border-radius:50%;background:linear-gradient(135deg,#0D9488,#0F766E);position:absolute;top:2px;left:2px;transition:all .25s;box-shadow:0 1px 3px rgba(0,0,0,.2);display:flex;align-items:center;justify-content:center;font-size:10px}
[data-theme="dark"] .theme-btn-knob{left:22px;background:linear-gradient(135deg,#1A2332,#0D1F38)}
.theme-btn-icon{font-size:11px;line-height:1}

/* ── HERO ── */
.hero{background:var(--hero-grad);padding:64px 32px;position:relative;overflow:hidden;transition:background .3s}
[data-theme="dark"] .hero{border-bottom:1px solid rgba(13,148,136,.2)}
.hero::before{content:'';position:absolute;top:-60px;right:-80px;width:400px;height:400px;background:rgba(255,255,255,.04);border-radius:50%}
.hero-content{max-width:900px;margin:0 auto;position:relative;z-index:1}
.hero-eyebrow{display:inline-flex;align-items:center;gap:7px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);border-radius:99px;padding:4px 14px;font-size:11px;font-weight:600;color:rgba(167,243,208,.95);letter-spacing:.8px;text-transform:uppercase;margin-bottom:20px}
.hero-eyebrow::before{content:'';width:6px;height:6px;background:#4ADE80;border-radius:50%;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.hero h1{font-size:46px;font-weight:700;line-height:1.1;letter-spacing:-2px;color:#fff;margin-bottom:16px}
[data-theme="dark"] .hero h1{color:#E2E8F0}
.hero h1 em{font-style:italic;color:#99F6E4}
.hero p{font-size:15px;color:rgba(167,243,208,.85);max-width:560px;line-height:1.7;margin-bottom:28px;font-weight:300}
[data-theme="dark"] .hero p{color:rgba(148,163,184,.8)}
.hero-actions{display:flex;gap:12px;flex-wrap:wrap}
.btn-hero{background:#fff;color:var(--teal2);padding:11px 24px;border-radius:9px;font-size:14px;font-weight:700;cursor:pointer;border:none;transition:all .2s}
[data-theme="dark"] .btn-hero{background:var(--teal);color:#fff}
.btn-hero:hover{transform:translateY(-1px);box-shadow:0 4px 20px rgba(255,255,255,.25)}
.btn-outline-hero{background:rgba(255,255,255,.1);color:#fff;border:1px solid rgba(255,255,255,.25);padding:11px 24px;border-radius:9px;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;font-family:'DM Sans',sans-serif}
.btn-outline-hero:hover{background:rgba(255,255,255,.18)}
.hero-stats{display:flex;gap:32px;margin-top:40px;padding-top:32px;border-top:1px solid rgba(255,255,255,.12)}
.hero-stat span{display:block;font-size:28px;font-weight:700;color:#fff;letter-spacing:-1px}
[data-theme="dark"] .hero-stat span{color:#E2E8F0}
.hero-stat small{font-size:11px;color:rgba(167,243,208,.7);font-weight:500;letter-spacing:.3px;text-transform:uppercase}

/* ── CONTAINERS ── */
.container{max-width:960px;margin:0 auto;padding:0 32px}
.section{padding:56px 0}
.section-label{font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--teal);margin-bottom:8px}
.section-title{font-size:28px;font-weight:700;letter-spacing:-.8px;color:var(--text);margin-bottom:6px}
.section-sub{font-size:14px;color:var(--text2);line-height:1.6;margin-bottom:36px}

/* ── TECH CARDS ── */
.tech-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
.tech-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;transition:all .2s}
.tech-card:hover{transform:translateY(-2px);box-shadow:var(--shadow2);border-color:var(--teal4)}
.tech-icon{width:42px;height:42px;border-radius:10px;display:flex;align-items:center;justify-content:center;margin-bottom:14px;font-size:18px}
.tech-card h4{font-size:14px;font-weight:600;margin-bottom:4px;color:var(--text)}
.tech-card p{font-size:12px;color:var(--text2);line-height:1.5}
.tech-tag{display:inline-block;font-size:10px;font-weight:600;padding:2px 8px;border-radius:99px;margin-top:10px;letter-spacing:.3px}

/* ── WORKFLOW ── */
.workflow{display:grid;grid-template-columns:repeat(4,1fr);gap:0;position:relative}
.workflow::before{content:'';position:absolute;top:28px;left:calc(12.5%);right:calc(12.5%);height:1.5px;background:linear-gradient(to right,var(--teal),var(--teal4));z-index:0}
.workflow-step{text-align:center;position:relative;z-index:1;padding:0 8px}
.workflow-num{width:56px;height:56px;border-radius:50%;background:var(--surface);border:2px solid var(--teal);display:flex;align-items:center;justify-content:center;margin:0 auto 14px;font-size:18px;font-weight:700;color:var(--teal)}
.workflow-num.filled{background:var(--teal);color:#fff}
.workflow-step h4{font-size:13px;font-weight:600;margin-bottom:4px;color:var(--text)}
.workflow-step p{font-size:11px;color:var(--text2);line-height:1.5}

/* ── CLASSES & AUC ── */
.classes-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}
.class-chip{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:9px 12px;font-size:12px;font-weight:500;color:var(--text);display:flex;align-items:center;gap:8px;transition:all .15s}
.class-chip:hover{border-color:var(--teal4);background:var(--teal3);color:var(--teal2)}
.class-chip .auc{margin-left:auto;font-size:10px;font-weight:700;color:var(--text3)}
.chip-dot{width:7px;height:7px;border-radius:50%;background:var(--teal);flex-shrink:0}
.auc-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.auc-name{font-size:11px;color:var(--text);width:160px;flex-shrink:0}
.auc-bar-wrap{flex:1;height:5px;background:var(--border);border-radius:99px;overflow:hidden}
.auc-bar{height:100%;border-radius:99px;background:linear-gradient(to right,var(--teal),var(--teal4))}
.auc-val{font-size:10px;font-weight:700;font-family:'DM Mono',monospace;color:var(--teal2);min-width:48px;text-align:right}

/* ── DISCLAIMER ── */
.disclaimer{background:linear-gradient(135deg,#FFFBEB,#FEF3C7);border:1px solid #FDE68A;border-radius:12px;padding:16px 20px;display:flex;gap:12px;align-items:flex-start;margin:32px 0}
[data-theme="dark"] .disclaimer{background:rgba(217,119,6,.1);border-color:rgba(217,119,6,.3)}
[data-theme="dark"] .disclaimer p{color:#FCD34D}
.disclaimer p{font-size:12.5px;color:#92400E;line-height:1.6}

/* ── ANALYZE LAYOUT ── */
.upload-layout{display:grid;grid-template-columns:340px 1fr;gap:24px;padding:28px 32px;max-width:1100px;margin:0 auto;align-items:start}
@media(max-width:768px){.upload-layout{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden;transition:background .3s,border-color .3s}
.panel-header{padding:14px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:9px}
.panel-header h3{font-size:13px;font-weight:700;letter-spacing:-.2px;color:var(--text)}
.panel-dot{width:8px;height:8px;border-radius:50%}
.panel-body{padding:18px}

/* ── PATIENT FORM ── */
.patient-form-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.patient-form-full{grid-column:1/-1}
.form-group{display:flex;flex-direction:column;gap:6px}
.form-label{font-size:11px;font-weight:700;color:var(--text2);letter-spacing:.6px;text-transform:uppercase;display:flex;align-items:center;gap:4px;line-height:1}
.form-label .req{color:#F43F5E;font-size:13px;line-height:1;font-weight:700}
.form-input{background:var(--input-bg);border:1.5px solid var(--input-border);border-radius:8px;padding:9px 12px;font-size:13px;font-family:'DM Sans',sans-serif;color:var(--text);outline:none;transition:all .2s;width:100%}
.form-input:focus{border-color:var(--teal);box-shadow:0 0 0 3px var(--teal3)}
.form-input::placeholder{color:var(--text3);font-size:12px}
.form-input.error{border-color:#F43F5E;box-shadow:0 0 0 3px rgba(244,63,94,.15)}
.form-error-msg{font-size:10px;color:#F43F5E;font-weight:500;display:none;margin-top:1px}
.form-group.has-error .form-input{border-color:#F43F5E;box-shadow:0 0 0 3px rgba(244,63,94,.15)}
.form-group.has-error .form-error-msg{display:block}
.mandatory-note{font-size:10px;color:var(--text3);line-height:1.5;display:flex;align-items:center;gap:4px}
.mandatory-note .req{color:#F43F5E;font-weight:700;font-size:12px}

/* ── PATIENT INFO BLOCK (results/report) ── */
.patient-block{background:var(--patient-block-bg);border:1px solid var(--patient-block-border);border-radius:10px;padding:12px 16px;margin-bottom:16px;display:flex;gap:20px;flex-wrap:wrap;align-items:center}
.patient-block-item{display:flex;flex-direction:column;gap:2px}
.patient-block-item label{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--teal2)}
.patient-block-item span{font-size:13px;font-weight:600;color:var(--text)}

/* ── UPLOAD ZONE ── */
.upload-zone{border:2px dashed var(--border);border-radius:10px;padding:32px 20px;text-align:center;cursor:pointer;transition:all .2s;position:relative;overflow:hidden}
.upload-zone:hover,.upload-zone.drag{border-color:var(--teal);background:var(--teal3)}
.upload-zone.has-image{border-style:solid;border-color:var(--teal);padding:0}
.upload-zone.has-image img{width:100%;border-radius:8px;display:block}
.upload-icon{width:48px;height:48px;background:var(--surface2);border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 12px}
.upload-zone h4{font-size:13px;font-weight:600;margin-bottom:4px;color:var(--text)}
.upload-zone p{font-size:11px;color:var(--text3)}
.upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}

/* ── SETTINGS ── */
.setting-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.setting-label{font-size:12px;font-weight:500;display:flex;align-items:center;gap:6px;color:var(--text)}
.setting-label small{font-size:10px;color:var(--text3);font-weight:400}
.setting-val{font-size:12px;font-weight:700;color:var(--teal);font-family:'DM Mono',monospace;min-width:36px;text-align:right}
input[type=range]{width:100%;height:3px;background:var(--border);border-radius:99px;outline:none;margin:6px 0 14px;accent-color:var(--teal)}
.toggle-row{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-top:1px solid var(--border)}
.toggle{width:36px;height:20px;background:var(--border);border-radius:99px;position:relative;cursor:pointer;transition:all .2s;flex-shrink:0}
.toggle.on{background:var(--teal)}
.toggle::after{content:'';position:absolute;width:14px;height:14px;background:#fff;border-radius:50%;top:3px;left:3px;transition:all .2s;box-shadow:0 1px 3px rgba(0,0,0,.2)}
.toggle.on::after{left:19px}
.toggle-label{font-size:12px;font-weight:500;color:var(--text)}
.toggle-sub{font-size:10px;color:var(--text3)}

/* ── BUTTONS ── */
.btn-analyze{width:100%;background:linear-gradient(135deg,var(--teal),var(--teal2));color:#fff;border:none;border-radius:10px;padding:13px;font-size:14px;font-weight:700;cursor:pointer;transition:all .2s;margin-top:14px;display:flex;align-items:center;justify-content:center;gap:8px;font-family:'DM Sans',sans-serif}
.btn-analyze:hover{transform:translateY(-1px);box-shadow:0 4px 20px rgba(13,148,136,.35)}
.btn-analyze:disabled{opacity:.5;cursor:not-allowed;transform:none}

/* ── RESULTS ── */
.results-tabs{display:flex;gap:2px;padding:4px;background:var(--surface2);border-radius:8px;margin-bottom:16px}
.rtab{flex:1;padding:7px;text-align:center;font-size:12px;font-weight:600;color:var(--text2);border-radius:6px;cursor:pointer;transition:all .15s}
.rtab.active{background:var(--surface);color:var(--teal2);box-shadow:0 1px 3px rgba(0,0,0,.08)}
.result-image{width:100%;background:var(--surface2);border-radius:8px;overflow:hidden;position:relative;min-height:200px;display:flex;align-items:center;justify-content:center}
.result-image img{width:100%;display:block}
.hm-label{position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,.6);color:#fff;font-size:10px;font-weight:600;padding:3px 8px;border-radius:4px}
.summary-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}
.sum-card{background:var(--surface2);border-radius:8px;padding:12px;text-align:center}
.sum-card .num{font-size:24px;font-weight:700;font-family:'DM Mono',monospace;color:var(--text)}
.sum-card .lbl{font-size:10px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
.sum-card.warn .num{color:var(--red)}
.sum-card.ok .num{color:var(--green)}
.finding-row{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--border)}
.finding-row:last-child{border-bottom:none}
.finding-name{font-size:12px;font-weight:500;flex:1;min-width:0;color:var(--text)}
.finding-bar-wrap{flex:1.5;height:4px;background:var(--border);border-radius:99px;overflow:hidden}
.finding-bar{height:100%;border-radius:99px}
.finding-pct{font-size:11px;font-weight:700;font-family:'DM Mono',monospace;min-width:36px;text-align:right}
.finding-status{font-size:9px;font-weight:700;padding:2px 7px;border-radius:99px;letter-spacing:.4px}
.pos{background:var(--red2);color:var(--red)}
.neg{background:var(--green2);color:var(--green)}
.download-bar{background:linear-gradient(135deg,#0D1521,#111C2C);border-radius:10px;padding:14px 18px;display:flex;align-items:center;justify-content:space-between;margin-top:14px}
[data-theme="dark"] .download-bar{background:linear-gradient(135deg,#060C16,#0A1628);border:1px solid var(--border)}
.download-bar p{font-size:12px;color:rgba(255,255,255,.6)}
.download-bar strong{color:#fff;font-size:13px;display:block}
.btn-dl{background:var(--teal);color:#fff;border:none;border-radius:7px;padding:9px 18px;font-size:12px;font-weight:700;cursor:pointer;font-family:'DM Sans',sans-serif;transition:all .2s}
.btn-dl:hover{background:var(--teal2)}

/* ── SPINNER ── */
.spinner{width:20px;height:20px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}
.loading .spinner{display:block}
.loading .btn-label{display:none}

/* ── REPORT PAGE ── */
.report-layout{max-width:820px;margin:0 auto;padding:28px 32px}
.report-doc{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden;box-shadow:var(--shadow2)}
.report-header{background:var(--report-header-bg);padding:28px 32px;display:flex;align-items:center;justify-content:space-between}
.report-header h2{font-size:24px;font-weight:700;color:#fff;letter-spacing:-.5px}
[data-theme="dark"] .report-header h2{color:#E2E8F0}
.report-header p{font-size:11px;color:rgba(167,243,208,.8);margin-top:4px}
[data-theme="dark"] .report-header p{color:rgba(148,163,184,.7)}
.report-logo{font-size:11px;font-weight:700;color:rgba(167,243,208,.7);text-align:right}
.report-logo span{display:block;font-size:26px;font-weight:700;color:#fff;letter-spacing:-1px}
.report-meta{padding:16px 32px;background:var(--surface2);border-bottom:1px solid var(--border);display:flex;gap:32px;flex-wrap:wrap}
.meta-item{font-size:11px}
.meta-item label{display:block;color:var(--text3);font-weight:500;letter-spacing:.3px;text-transform:uppercase;margin-bottom:1px}
.meta-item strong{color:var(--text);font-weight:600;font-size:12px}
.report-body{padding:28px 32px}
.report-section{margin-bottom:28px}
.report-section-title{font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--teal);margin-bottom:12px;padding-bottom:8px;border-bottom:1.5px solid var(--teal3)}
.report-summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.rsm{background:var(--surface2);border-radius:8px;padding:12px;text-align:center;border:1px solid var(--border)}
.rsm .rv{font-size:20px;font-weight:700;font-family:'DM Mono',monospace;color:var(--text)}
.rsm .rl{font-size:9px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
.rsm.alert .rv{color:var(--red)}
table{width:100%;border-collapse:collapse;font-size:12px}
table th{text-align:left;padding:8px 10px;background:var(--surface2);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--text2);border:1px solid var(--border)}
table td{padding:8px 10px;border:1px solid var(--border);color:var(--text)}
table tr:nth-child(even) td{background:var(--table-even)}
.badge{font-size:9px;font-weight:700;padding:2px 8px;border-radius:99px;letter-spacing:.3px;text-transform:uppercase}
.badge.positive{background:var(--red2);color:var(--red)}
.badge.negative{background:var(--green2);color:var(--green)}
.badge.high{background:#EFF6FF;color:#1D4ED8}
[data-theme="dark"] .badge.high{background:rgba(29,78,216,.2);color:#93C5FD}
.heatmap-demo{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.hm-img{border-radius:8px;overflow:hidden;background:#0D1117;position:relative;min-height:160px;display:flex;align-items:center;justify-content:center}
.hm-img img{width:100%;display:block}
.report-footer{padding:16px 32px;background:var(--surface2);border-top:1px solid var(--border);text-align:center}
.report-footer p{font-size:10px;color:var(--text3);line-height:1.6}
.report-actions{display:flex;gap:10px;justify-content:flex-end;margin-bottom:16px}
.btn-action{padding:9px 20px;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;border:none;display:inline-flex;align-items:center;gap:6px;font-family:'DM Sans',sans-serif;transition:all .2s}
.btn-action.primary{background:var(--teal);color:#fff}
.btn-action.primary:hover{background:var(--teal2)}
.btn-action.sec{background:var(--surface);color:var(--text);border:1px solid var(--border)}
.btn-action.sec:hover{background:var(--surface2)}
.nav-back{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text2);cursor:pointer;padding:6px 0;margin-bottom:4px;width:fit-content}
.nav-back:hover{color:var(--teal)}

/* ── TOAST ── */
.toast{position:fixed;bottom:24px;right:24px;background:#1A2332;color:#fff;padding:12px 20px;border-radius:10px;font-size:13px;font-weight:500;z-index:999;transform:translateY(100px);opacity:0;transition:all .3s;border-left:3px solid var(--teal)}
[data-theme="dark"] .toast{background:#0A1628;border-color:var(--teal)}
.toast.show{transform:translateY(0);opacity:1}

/* ── HISTORY PAGE ── */
.history-layout{max-width:1060px;margin:0 auto;padding:32px}
.history-topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;flex-wrap:wrap;gap:12px}
.history-topbar h2{font-size:22px;font-weight:700;letter-spacing:-.5px;color:var(--text)}
.history-topbar p{font-size:12px;color:var(--text2);margin-top:2px}
.history-actions{display:flex;gap:8px;align-items:center}
.btn-sm{padding:7px 14px;border-radius:7px;font-size:12px;font-weight:600;cursor:pointer;border:none;font-family:'DM Sans',sans-serif;transition:all .15s}
.btn-sm.primary{background:var(--teal);color:#fff}
.btn-sm.primary:hover{background:var(--teal2)}
.btn-sm.danger{background:var(--red2);color:var(--red);border:1px solid rgba(220,38,38,.3)}
.btn-sm.danger:hover{opacity:.8}
.btn-sm.sec{background:var(--surface);color:var(--text2);border:1px solid var(--border)}
.btn-sm.sec:hover{background:var(--surface2)}
.history-stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
.hstat{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.hstat .hv{font-size:22px;font-weight:700;font-family:'DM Mono',monospace;color:var(--text);letter-spacing:-.5px}
.hstat .hl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--text3);margin-top:2px}
.hstat.teal .hv{color:var(--teal)}
.hstat.red .hv{color:var(--red)}
.history-filter-bar{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;align-items:center}
.filter-chip{padding:5px 12px;border-radius:99px;font-size:11px;font-weight:600;cursor:pointer;border:1px solid var(--border);background:var(--surface);color:var(--text2);transition:all .15s}
.filter-chip.active{background:var(--teal3);border-color:var(--teal4);color:var(--teal2)}
.history-search{flex:1;max-width:260px;padding:6px 12px;border:1px solid var(--border);border-radius:7px;font-size:12px;font-family:'DM Sans',sans-serif;background:var(--input-bg);color:var(--text);outline:none}
.history-search:focus{border-color:var(--teal)}
.history-empty{text-align:center;padding:80px 20px;background:var(--surface);border:1px solid var(--border);border-radius:14px}
.history-empty h4{font-size:15px;font-weight:600;color:var(--text2);margin-bottom:6px}
.history-empty p{font-size:12px;color:var(--text3);margin-bottom:20px}
.history-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.hcard{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;transition:all .2s;cursor:pointer;position:relative}
.hcard:hover{transform:translateY(-2px);box-shadow:var(--shadow2);border-color:var(--teal4)}
.hcard-thumb{width:100%;height:130px;object-fit:cover;background:var(--surface2);display:block}
.hcard-thumb-placeholder{width:100%;height:130px;background:linear-gradient(135deg,#0D1521,#1A2740);display:flex;align-items:center;justify-content:center}
.hcard-body{padding:14px}
.hcard-header{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:8px;gap:8px}
.hcard-name{font-size:12px;font-weight:700;color:var(--text);word-break:break-all;line-height:1.3}
.hcard-patient{font-size:11px;font-weight:600;color:var(--teal2);margin-bottom:6px;display:flex;align-items:center;gap:4px}
.hcard-date{font-size:10px;color:var(--text3);white-space:nowrap;flex-shrink:0}
.hcard-badges{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px}
.hcard-badge{font-size:9px;font-weight:700;padding:2px 7px;border-radius:99px;letter-spacing:.3px}
.hcard-badge.pos{background:var(--red2);color:var(--red)}
.hcard-badge.neg{background:var(--green2);color:var(--green)}
.hcard-badge.info{background:#EFF6FF;color:#1D4ED8}
[data-theme="dark"] .hcard-badge.info{background:rgba(29,78,216,.2);color:#93C5FD}
.hcard-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:10px}
.hcard-stat{background:var(--surface2);border-radius:6px;padding:6px 8px;text-align:center}
.hcard-stat .v{font-size:14px;font-weight:700;font-family:'DM Mono',monospace;color:var(--text)}
.hcard-stat .l{font-size:9px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.3px}
.hcard-top-finding{font-size:11px;color:var(--text2);padding:7px 10px;background:var(--surface2);border-radius:6px;margin-bottom:10px;font-weight:500}
.hcard-top-finding strong{color:var(--teal2)}
.hcard-actions{display:flex;gap:6px}
.hcard-btn{flex:1;padding:6px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;border:none;font-family:'DM Sans',sans-serif;transition:all .15s;text-align:center}
.hcard-btn.view{background:var(--teal3);color:var(--teal2)}
.hcard-btn.view:hover{background:var(--teal4)}
.hcard-btn.del{background:var(--red2);color:var(--red)}
.hcard-btn.del:hover{opacity:.8}
.hcard-pos-indicator{position:absolute;top:10px;right:10px;background:var(--red);color:#fff;font-size:9px;font-weight:700;padding:3px 8px;border-radius:99px;letter-spacing:.3px}
.hcard-pos-indicator.clean{background:var(--green)}

/* ── HISTORY MODAL ── */
.hmodal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:200;display:none;align-items:center;justify-content:center;padding:20px}
[data-theme="dark"] .hmodal-overlay{background:rgba(0,0,0,.75)}
.hmodal-overlay.open{display:flex}
.hmodal{background:var(--surface);border-radius:16px;width:100%;max-width:780px;max-height:90vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,.3);border:1px solid var(--border)}
.hmodal-header{padding:20px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;background:var(--surface);z-index:1}
.hmodal-header h3{font-size:16px;font-weight:700;color:var(--text)}
.hmodal-close{width:30px;height:30px;border-radius:50%;background:var(--surface2);border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:16px;color:var(--text2)}
.hmodal-body{padding:24px}
.hmodal-imgs{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}
.hmodal-img-wrap{border-radius:8px;overflow:hidden;background:var(--surface2);aspect-ratio:1;display:flex;align-items:center;justify-content:center}
.hmodal-img-wrap img{width:100%;display:block}
.hmodal-img-label{font-size:10px;font-weight:600;color:var(--text3);text-align:center;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}
.alert-box{border-radius:8px;padding:12px 16px;font-size:12px;line-height:1.6;margin-bottom:16px}
.alert-box.warn{background:var(--alert-warn-bg);border:1px solid var(--alert-warn-border);color:var(--alert-warn-text)}
.alert-box.ok{background:var(--alert-ok-bg);border:1px solid var(--alert-ok-border);color:var(--alert-ok-text)}

/* ── FOOTER ── */
#site-footer{background:var(--footer-bg);border-top:1px solid rgba(13,148,136,.25);margin-top:0;transition:background .3s}
.footer-inner{max-width:960px;margin:0 auto;padding:52px 32px 40px;display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:40px}
@media(max-width:768px){.footer-inner{grid-template-columns:1fr 1fr;gap:32px}.footer-brand{grid-column:1 / -1}}
.footer-logo{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.footer-logo-icon{width:32px;height:32px;background:linear-gradient(135deg,#0D9488,#0F766E);border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.footer-logo span{font-size:16px;font-weight:700;letter-spacing:-.3px;color:#fff}
.footer-logo em{color:#5EEAD4;font-style:italic}
.footer-tagline{font-size:12px;color:rgba(148,163,184,.7);line-height:1.7;margin-bottom:16px;font-weight:300}
.footer-badges{display:flex;flex-wrap:wrap;gap:6px}
.footer-badge{font-size:10px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;padding:3px 9px;border-radius:99px;background:rgba(13,148,136,.15);border:1px solid rgba(13,148,136,.3);color:#5EEAD4}
.footer-col-title{font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#5EEAD4;margin-bottom:16px}
.footer-links{list-style:none;display:flex;flex-direction:column;gap:10px}
.footer-links a{font-size:12px;color:rgba(148,163,184,.7);text-decoration:none;transition:color .15s;display:inline-flex;align-items:center;gap:4px}
.footer-links a:hover{color:#fff}
.footer-links a[target="_blank"]::after{content:'↗';font-size:9px;opacity:.5}
.footer-stats{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.footer-stat{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:8px;padding:10px;text-align:center}
.fstat-val{display:block;font-size:17px;font-weight:700;font-family:'DM Mono',monospace;color:#fff;letter-spacing:-.5px}
.fstat-lbl{display:block;font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:rgba(148,163,184,.5);margin-top:2px}
.footer-bottom{border-top:1px solid rgba(255,255,255,.06)}
.footer-bottom-inner{max-width:960px;margin:0 auto;padding:16px 32px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
.footer-copy{font-size:11px;color:rgba(148,163,184,.4)}
.footer-disclaimer-inline{font-size:11px;color:rgba(234,179,8,.6);font-weight:500}
.footer-device{display:flex;align-items:center;gap:6px;font-size:11px;color:rgba(148,163,184,.4);font-family:'DM Mono',monospace}
.device-dot{width:6px;height:6px;border-radius:50%;background:#4ADE80;animation:blink 2s infinite}

/* ── SURFACE SECTION BG ── */
.surface-section{background:var(--surface);border-top:1px solid var(--border);border-bottom:1px solid var(--border);transition:background .3s,border-color .3s}
</style>
</head>
<body>
<nav>
  <div class="nav-logo" onclick="goPage('home')">
    <div class="nav-logo-icon">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="3" width="18" height="18" rx="3" stroke="white" stroke-width="1.5"/>
        <path d="M7 12h2l2-4 2 8 2-4h2" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <span>Clini<em>Scan</em> AI</span>
  </div>
  <div class="nav-links">
    <button class="nav-link active" id="nav-home"    onclick="goPage('home')">Overview</button>
    <button class="nav-link"        id="nav-analyze" onclick="goPage('analyze')">Analyze X-Ray</button>
    <button class="nav-link"        id="nav-report"  onclick="goPage('report')">View Report</button>
    <button class="nav-link"        id="nav-history" onclick="goPage('history')">History <span id="historyCount" style="background:var(--teal);color:#fff;font-size:9px;font-weight:700;padding:1px 5px;border-radius:99px;margin-left:2px;display:none">0</span></button>
    <span class="nav-badge">RESEARCH</span>
    <!-- THEME TOGGLE -->
    <div class="theme-toggle" title="Toggle dark/light mode">
      <button class="theme-btn" id="themeBtn" onclick="toggleTheme()" aria-label="Toggle theme">
        <div class="theme-btn-knob" id="themeKnob">
          <span class="theme-btn-icon" id="themeIcon">☀️</span>
        </div>
      </button>
    </div>
  </div>
</nav>

<!-- ═══════════════ HOME ═══════════════ -->
<div class="page active" id="page-home">
  <div class="hero">
    <div class="hero-content">
      <div class="hero-eyebrow">AI-Powered Chest X-Ray Analysis</div>
      <h1>Clini<em>Scan</em> AI<br>Diagnostic Platform</h1>
      <p>Automated detection of 14 chest pathologies using deep learning. EfficientNet-B3 classification with YOLOv8m localization and Grad-CAM visual explainability.</p>
      <div class="hero-actions">
        <button class="btn-hero" onclick="goPage('analyze')">Start Analysis</button>
        <button class="btn-outline-hero" onclick="goPage('report')">View Sample Report</button>
      </div>
      <div class="hero-stats">
        <div class="hero-stat"><span>91.93%</span><small>Mean AUC</small></div>
        <div class="hero-stat"><span>14</span><small>Pathologies</small></div>
        <div class="hero-stat"><span>18K</span><small>Training Images</small></div>
        <div class="hero-stat"><span>VinDr-CXR</span><small>Dataset</small></div>
      </div>
    </div>
  </div>

  <div class="container">
    <div class="section">
      <div class="section-label">How It Works</div>
      <div class="section-title">Four-Step Analysis Pipeline</div>
      <div class="section-sub">From image upload to downloadable clinical report in seconds.</div>
      <div class="workflow">
        <div class="workflow-step"><div class="workflow-num filled">1</div><h4>Upload X-Ray</h4><p>JPEG/PNG chest radiograph</p></div>
        <div class="workflow-step"><div class="workflow-num">2</div><h4>AI Classification</h4><p>EfficientNet-B3 scores 14 classes</p></div>
        <div class="workflow-step"><div class="workflow-num">3</div><h4>Detection + Heatmap</h4><p>YOLOv8m boxes + Grad-CAM</p></div>
        <div class="workflow-step"><div class="workflow-num">4</div><h4>PDF Report</h4><p>Structured clinical report</p></div>
      </div>
    </div>
  </div>

  <div class="surface-section">
    <div class="container">
      <div class="section">
        <div class="section-label">Technology Stack</div>
        <div class="section-title">Models &amp; Libraries</div>
        <div class="section-sub">Built on state-of-the-art computer vision frameworks.</div>
        <div class="tech-grid">
          <div class="tech-card"><div class="tech-icon" style="background:#EFF6FF"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a5 5 0 0 1 5 5c0 1.8-.96 3.37-2.4 4.22C16.5 12.13 18 14.38 18 17H6c0-2.62 1.5-4.87 3.4-5.78A5 5 0 0 1 12 2z"/><path d="M9 21h6M12 17v4"/><circle cx="12" cy="7" r="2"/></svg></div><h4>EfficientNet-B3</h4><p>Multi-label classifier predicting probability for each of 14 pathology classes.</p><span class="tech-tag" style="background:#EFF6FF;color:#1D4ED8">PyTorch · torchvision</span></div>
          <div class="tech-card"><div class="tech-icon" style="background:#F0FDF4"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9l6 6M15 9l-6 6"/><rect x="8" y="3" width="8" height="3" rx="1" fill="#166534" stroke="none" opacity=".25"/></svg></div><h4>YOLOv8m</h4><p>Lesion localization with bounding boxes for each detected abnormality region.</p><span class="tech-tag" style="background:#F0FDF4;color:#166534">Ultralytics</span></div>
          <div class="tech-card"><div class="tech-icon" style="background:#FFF7ED"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#92400E" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/><path d="M12 13v4M10 17h4"/></svg></div><h4>Grad-CAM</h4><p>Gradient-weighted Class Activation Maps highlighting model attention regions.</p><span class="tech-tag" style="background:#FFF7ED;color:#92400E">Explainability</span></div>
          <div class="tech-card"><div class="tech-icon" style="background:#FDF4FF"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6B21A8" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg></div><h4>fpdf2</h4><p>Structured PDF report generation with findings tables and model metadata.</p><span class="tech-tag" style="background:#FDF4FF;color:#6B21A8">PDF Export</span></div>
          <div class="tech-card"><div class="tech-icon" style="background:#ECFDF5"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#065F46" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg></div><h4>Albumentations</h4><p>Advanced image augmentation pipeline for preprocessing and validation transforms.</p><span class="tech-tag" style="background:#ECFDF5;color:#065F46">Preprocessing</span></div>
          <div class="tech-card"><div class="tech-icon" style="background:#FFF1F2"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#9F1239" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v6c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 11v6c0 1.66 4.03 3 9 3s9-1.34 9-3v-6"/></svg></div><h4>VinDr-CXR</h4><p>18,000 annotated chest X-rays. 70/20/10 split from Vietnamese hospitals.</p><span class="tech-tag" style="background:#FFF1F2;color:#9F1239">Dataset</span></div>
        </div>
      </div>
    </div>
  </div>

  <div class="container">
    <div class="section">
      <div class="section-label">Detection Targets</div>
      <div class="section-title">14 Pathology Classes</div>
      <div class="section-sub">Each class independently scored 0–100%. Threshold ≥ 50% = positive finding.</div>
      <div class="classes-grid" id="classGrid"></div>
    </div>
  </div>

  <div class="surface-section">
    <div class="container">
      <div class="section">
        <div class="section-label">Model Performance</div>
        <div class="section-title">Per-Class AUC Scores</div>
        <div class="section-sub">Area Under ROC Curve on held-out test set (VinDr-CXR, 10% split).</div>
        <div id="aucBars"></div>
      </div>
    </div>
  </div>

  <div class="container">
    <div class="disclaimer">
      <div style="color:#D97706;font-size:18px;flex-shrink:0">⚠</div>
      <p><strong>Research Use Only.</strong> CliniScan AI is not a certified medical device. Outputs are not a substitute for radiologist review. Always consult a qualified healthcare professional.</p>
    </div>
    <div style="height:40px"></div>
  </div>
</div>

<!-- ═══════════════ ANALYZE ═══════════════ -->
<div class="page" id="page-analyze">
  <div class="upload-layout">
    <div>
      <!-- Patient Details Panel -->
      <div class="panel" style="margin-bottom:14px">
        <div class="panel-header">
          <div class="panel-dot" style="background:#F472B6"></div>
          <h3>Patient Details</h3>
        </div>
        <div class="panel-body">
          <div class="patient-form-grid">
            <div class="form-group patient-form-full" id="grpName">
              <label class="form-label" for="patientName">
                Full Name <span class="req">*</span>
              </label>
              <input class="form-input" type="text" id="patientName"
                     placeholder="Enter patient full name" autocomplete="off"
                     oninput="clearFieldError('grpName')">
              <span class="form-error-msg">Full name is required</span>
            </div>
            <div class="form-group" id="grpAge">
              <label class="form-label" for="patientAge">
                Age <span class="req">*</span>
              </label>
              <input class="form-input" type="number" id="patientAge"
                     placeholder="Enter age" min="0" max="130" maxlength="3"
                     oninput="clearFieldError('grpAge')">
              <span class="form-error-msg">Age is required</span>
            </div>
            <div class="form-group" id="grpPhone">
              <label class="form-label" for="patientPhone">
                Phone No. <span class="req">*</span>
              </label>
              <input class="form-input" type="tel" id="patientPhone"
                     placeholder="Enter phone number" autocomplete="off"
                     oninput="clearFieldError('grpPhone')">
              <span class="form-error-msg">Phone number is required</span>
            </div>
          </div>
          <div class="mandatory-note"><span class="req">*</span> All fields are mandatory</div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <div class="panel-dot" style="background:#22D3EE"></div>
          <h3>Input Image</h3>
        </div>
        <div class="panel-body">
          <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
            <div id="uploadPlaceholder">
              <div class="upload-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="#8FA0B4" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
              </div>
              <h4>Upload Chest X-Ray</h4>
              <p>Click or drag &amp; drop<br>JPEG · PNG</p>
            </div>
            <img id="previewImg" style="display:none">
            <input type="file" id="fileInput" accept="image/*" onchange="handleFile(event)">
          </div>
          <div id="uploadedName" style="font-size:11px;color:var(--text3);margin-top:8px;text-align:center;display:none"></div>
        </div>
      </div>

      <div class="panel" style="margin-top:14px">
        <div class="panel-header">
          <div class="panel-dot" style="background:#A78BFA"></div>
          <h3>Detection Settings</h3>
        </div>
        <div class="panel-body">
          <div class="setting-row">
            <div class="setting-label">Confidence <small>YOLO threshold</small></div>
            <div class="setting-val" id="confVal">0.15</div>
          </div>
          <input type="range" min="5" max="50" value="15" step="5"
                 oninput="document.getElementById('confVal').textContent='0.'+String(this.value).padStart(2,'0')">
          <div class="setting-row">
            <div class="setting-label">IoU <small>overlap threshold</small></div>
            <div class="setting-val" id="iouVal">0.45</div>
          </div>
          <input type="range" min="30" max="70" value="45" step="5"
                 oninput="document.getElementById('iouVal').textContent='0.'+this.value">
          <div class="toggle-row">
            <div><div class="toggle-label">Grad-CAM Heatmap</div><div class="toggle-sub">Attention visualization</div></div>
            <div class="toggle on" id="tGradcam" onclick="this.classList.toggle('on')"></div>
          </div>
          <div class="toggle-row">
            <div><div class="toggle-label">Bounding Boxes</div><div class="toggle-sub">YOLO detections</div></div>
            <div class="toggle on" id="tBbox" onclick="this.classList.toggle('on')"></div>
          </div>
        </div>
      </div>

      <button class="btn-analyze" id="analyzeBtn" onclick="runAnalysis()" disabled>
        <div class="spinner" id="spinner"></div>
        <span class="btn-label">Run AI Analysis</span>
      </button>
    </div>

    <div>
      <div class="panel">
        <div class="panel-header">
          <div class="panel-dot" style="background:#34D399"></div>
          <h3>Analysis Results</h3>
        </div>
        <div class="panel-body" id="resultsPanel">
          <div id="emptyState" style="text-align:center;padding:60px 20px">
            <div style="width:60px;height:60px;background:var(--surface2);border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 16px">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="18" height="18" rx="3" stroke="#8FA0B4" stroke-width="1.5"/><path d="M7 12h2l2-4 2 8 2-4h2" stroke="#8FA0B4" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <h4 style="font-size:14px;font-weight:600;color:var(--text2);margin-bottom:4px">No Analysis Yet</h4>
            <p style="font-size:12px;color:var(--text3)">Upload a chest X-ray and click Run AI Analysis.</p>
          </div>
          <div id="resultsContent" style="display:none">
            <div id="resultPatientBlock"></div>
            <div class="summary-grid">
              <div class="sum-card warn"><div class="num" id="rDetections">—</div><div class="lbl">Detections</div></div>
              <div class="sum-card warn"><div class="num" id="rPositives">—</div><div class="lbl">Positive</div></div>
              <div class="sum-card ok"><div class="num" id="rTime">—</div><div class="lbl">Time</div></div>
            </div>
            <div class="results-tabs">
              <div class="rtab active" onclick="switchTab('detection',this)">Detection</div>
              <div class="rtab" onclick="switchTab('heatmap',this)">Grad-CAM</div>
              <div class="rtab" onclick="switchTab('findings',this)">All Findings</div>
            </div>
            <div id="tab-detection">
              <div class="result-image" id="detectionImgWrap">
                <p style="font-size:12px;color:var(--text3)">No detections available</p>
                <div class="hm-label">Detection View — YOLOv8m</div>
              </div>
            </div>
            <div id="tab-heatmap" style="display:none">
              <div class="result-image" id="heatmapImgWrap">
                <p style="font-size:12px;color:var(--text3)">Grad-CAM not available</p>
                <div class="hm-label" id="gcamLabel">Grad-CAM</div>
              </div>
            </div>
            <div id="tab-findings" style="display:none">
              <div id="findingsList" style="max-height:300px;overflow-y:auto"></div>
            </div>
            <div class="download-bar">
              <div>
                <strong>PDF Report Ready</strong>
                <p>Full structured report with all findings</p>
              </div>
              <button class="btn-dl" id="reportBtn" onclick="downloadReport()">Download PDF</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════ REPORT ═══════════════ -->
<div class="page" id="page-report">
  <div class="report-layout">
    <div class="nav-back" onclick="goPage('analyze')">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M19 12H5M5 12l7 7M5 12l7-7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Back to Analysis
    </div>
    <div class="report-actions">
      <button class="btn-action sec" onclick="window.print()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M6 9V2h12v7M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2M6 14h12v8H6v-8z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
        Print
      </button>
      <button class="btn-action primary" onclick="downloadReport()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
        Download PDF
      </button>
    </div>
    <div class="report-doc" id="reportDoc">
      <div style="padding:60px;text-align:center;color:var(--text3)">
        <p style="font-size:14px">Run an analysis first to generate the report.</p>
        <button class="btn-hero" style="margin-top:16px;color:var(--teal2)" onclick="goPage('analyze')">Go to Analysis →</button>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════ HISTORY ═══════════════ -->
<div class="page" id="page-history">
  <div class="history-layout">
    <div class="history-topbar">
      <div>
        <h2>Analysis History</h2>
        <p>All X-ray analyses from this session — stored locally in your browser.</p>
      </div>
      <div class="history-actions">
        <button class="btn-sm sec" onclick="exportHistoryCSV()">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style="display:inline;margin-right:4px"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
          Export CSV
        </button>
        <button class="btn-sm danger" onclick="clearHistory()">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style="display:inline;margin-right:4px"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
          Clear All
        </button>
      </div>
    </div>

    <div class="history-stats-row">
      <div class="hstat"><div class="hv" id="hsTotalScans">0</div><div class="hl">Total Scans</div></div>
      <div class="hstat red"><div class="hv" id="hsTotalPositive">0</div><div class="hl">Positive Findings</div></div>
      <div class="hstat teal"><div class="hv" id="hsTotalDetections">0</div><div class="hl">Total Detections</div></div>
      <div class="hstat"><div class="hv" id="hsAvgTime">—</div><div class="hl">Avg Analysis Time</div></div>
    </div>

    <div class="history-filter-bar">
      <div class="filter-chip active" onclick="filterHistory('all',this)">All</div>
      <div class="filter-chip" onclick="filterHistory('positive',this)">Positive Findings</div>
      <div class="filter-chip" onclick="filterHistory('clean',this)">No Findings</div>
      <div class="filter-chip" onclick="filterHistory('detected',this)">With Detections</div>
      <input class="history-search" type="text" placeholder="Search by name, filename or class…" oninput="searchHistory(this.value)" id="historySearchInput">
    </div>

    <div id="historyGridWrap">
      <div class="history-empty" id="historyEmpty">
        <div style="width:64px;height:64px;background:var(--surface2);border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 16px">
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none"><path d="M12 8v4l3 3M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" stroke="#8FA0B4" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </div>
        <h4>No Analysis History Yet</h4>
        <p>Run your first X-ray analysis and it will appear here automatically.</p>
        <button class="btn-sm primary" onclick="goPage('analyze')">Start Analysis</button>
      </div>
      <div class="history-grid" id="historyGrid" style="display:none"></div>
    </div>
  </div>
</div>

<!-- History Detail Modal -->
<div class="hmodal-overlay" id="hmodalOverlay" onclick="closeHistoryModal(event)">
  <div class="hmodal" id="hmodal">
    <div class="hmodal-header">
      <h3 id="hmodalTitle">Analysis Detail</h3>
      <button class="hmodal-close" onclick="closeHistoryModal()">✕</button>
    </div>
    <div class="hmodal-body" id="hmodalBody"></div>
  </div>
</div>

<!-- ═══════════════ FOOTER ═══════════════ -->
<footer id="site-footer">
  <div class="footer-inner">
    <div class="footer-brand">
      <div class="footer-logo">
        <div class="footer-logo-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <rect x="3" y="3" width="18" height="18" rx="3" stroke="white" stroke-width="1.5"/>
            <path d="M7 12h2l2-4 2 8 2-4h2" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <span>Clini<em>Scan</em> AI</span>
      </div>
      <p class="footer-tagline">AI-powered chest X-ray analysis platform. EfficientNet-B3 classification with YOLOv8m localization and Grad-CAM explainability.</p>
      <div class="footer-badges">
        <span class="footer-badge">Research Only</span>
        <span class="footer-badge">Not FDA Cleared</span>
        <span class="footer-badge">VinDr-CXR</span>
      </div>
    </div>
    <div class="footer-col">
      <div class="footer-col-title">Platform</div>
      <ul class="footer-links">
        <li><a href="#" onclick="goPage('home');return false">Overview</a></li>
        <li><a href="#" onclick="goPage('analyze');return false">Analyze X-Ray</a></li>
        <li><a href="#" onclick="goPage('report');return false">View Report</a></li>
        <li><a href="#" onclick="goPage('history');return false">History</a></li>
        <li><a href="/health" target="_blank">Health Check</a></li>
        <li><a href="/docs" target="_blank">API Docs</a></li>
      </ul>
    </div>
    <div class="footer-col">
      <div class="footer-col-title">Technology</div>
      <ul class="footer-links">
        <li><a href="https://arxiv.org/abs/1905.11946" target="_blank">EfficientNet-B3</a></li>
        <li><a href="https://docs.ultralytics.com" target="_blank">YOLOv8m</a></li>
        <li><a href="https://arxiv.org/abs/1610.02391" target="_blank">Grad-CAM</a></li>
        <li><a href="https://vindr.ai/datasets/cxr" target="_blank">VinDr-CXR Dataset</a></li>
        <li><a href="https://fastapi.tiangolo.com" target="_blank">FastAPI Backend</a></li>
      </ul>
    </div>
    <div class="footer-col">
      <div class="footer-col-title">Model Stats</div>
      <div class="footer-stats">
        <div class="footer-stat"><span class="fstat-val">91.93%</span><span class="fstat-lbl">Mean AUC</span></div>
        <div class="footer-stat"><span class="fstat-val">14</span><span class="fstat-lbl">Pathologies</span></div>
        <div class="footer-stat"><span class="fstat-val">18K</span><span class="fstat-lbl">Train Images</span></div>
        <div class="footer-stat"><span class="fstat-val">384px</span><span class="fstat-lbl">Input Res.</span></div>
      </div>
    </div>
  </div>
  <div class="footer-bottom">
    <div class="footer-bottom-inner">
      <div class="footer-copy">© 2025 CliniScan AI · Built with PyTorch, Ultralytics &amp; FastAPI</div>
      <div class="footer-disclaimer-inline">⚠ Research use only · Not a substitute for radiologist review</div>
      <div class="footer-device" id="footerDevice">
        <span class="device-dot"></span>
        <span id="footerDeviceText">System Ready</span>
      </div>
    </div>
  </div>
</footer>

<div class="toast" id="toast"></div>

<script>
const AUC_DATA={
  "Aortic enlargement": 96.33,
  "Atelectasis":        93.62,
  "Calcification":      89.31,
  "Cardiomegaly":       96.04,
  "Consolidation":      92.03,
  "ILD":                89.24,
  "Infiltration":       92.73,
  "Lung Opacity":       94.39,
  "Nodule/Mass":        92.80,
  "Other lesion":       89.78,
  "Pleural effusion":   90.79,
  "Pleural thickening": 92.53,
  "Pneumothorax":       84.46,
  "Pulmonary fibrosis": 92.93
};

let lastResult      = null;
let uploadedFile    = null;
let analysisHistory = [];
let historyFilter   = 'all';
let historySearch   = '';

// ══════════════ THEME SYSTEM ══════════════
function getTheme(){return document.documentElement.getAttribute('data-theme')||'light'}

function applyTheme(theme){
  document.documentElement.setAttribute('data-theme',theme);
  const icon=document.getElementById('themeIcon');
  if(icon) icon.textContent=theme==='dark'?'🌙':'☀️';
  try{localStorage.setItem('cliniscan-theme',theme);}catch(e){}
}

function toggleTheme(){
  applyTheme(getTheme()==='dark'?'light':'dark');
}

// Load saved theme on start
(function(){
  try{
    const saved=localStorage.getItem('cliniscan-theme');
    if(saved==='dark'||saved==='light') applyTheme(saved);
    else if(window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches) applyTheme('dark');
  }catch(e){}
})();

// ══════════════ PAGE NAV ══════════════
function goPage(p){
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(x=>x.classList.remove('active'));
  document.getElementById('page-'+p).classList.add('active');
  document.getElementById('nav-'+p).classList.add('active');
  if(p==='history') renderHistoryPage();
  window.scrollTo(0,0);
}

// ══════════════ PATIENT HELPERS ══════════════
function getPatientInfo(){
  return {
    name:  document.getElementById('patientName').value.trim(),
    age:   document.getElementById('patientAge').value.trim(),
    phone: document.getElementById('patientPhone').value.trim(),
  };
}

function patientBlockHTML(p){
  if(!p||(!p.name&&!p.age&&!p.phone)) return '';
  const items=[];
  if(p.name)  items.push(`<div class="patient-block-item"><label>Patient Name</label><span>${escHtml(p.name)}</span></div>`);
  if(p.age)   items.push(`<div class="patient-block-item"><label>Age</label><span>${escHtml(p.age)} yrs</span></div>`);
  if(p.phone) items.push(`<div class="patient-block-item"><label>Phone</label><span>${escHtml(p.phone)}</span></div>`);
  return `<div class="patient-block">${items.join('')}</div>`;
}

function escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

// ══════════════ HISTORY ══════════════
function saveToHistory(data,filename,thumbB64,patient){
  const entry={
    id:           Date.now(),
    filename:     filename||'xray.png',
    timestamp:    new Date().toISOString(),
    elapsed:      data.elapsed,
    device:       data.device,
    predictions:  data.predictions,
    positives:    data.positives,
    boxes:        data.boxes,
    top_class:    data.top_class,
    det_img_b64:  data.det_img_b64  ||null,
    gcam_img_b64: data.gcam_img_b64 ||null,
    thumb:        thumbB64           ||data.det_img_b64||null,
    patient:      patient||{name:'',age:'',phone:''},
  };
  analysisHistory.unshift(entry);
  updateHistoryBadge();
  updateHistoryStats();
}

function updateHistoryBadge(){
  const c=document.getElementById('historyCount');
  if(!c) return;
  const n=analysisHistory.length;
  c.textContent=n;
  c.style.display=n>0?'inline':'none';
}

function updateHistoryStats(){
  const total    = analysisHistory.length;
  const posCount = analysisHistory.filter(e=>e.positives.length>0).length;
  const detTotal = analysisHistory.reduce((s,e)=>s+e.boxes.length,0);
  const avgTime  = total?(analysisHistory.reduce((s,e)=>s+e.elapsed,0)/total).toFixed(1)+'s':'—';
  document.getElementById('hsTotalScans').textContent      = total;
  document.getElementById('hsTotalPositive').textContent   = posCount;
  document.getElementById('hsTotalDetections').textContent = detTotal;
  document.getElementById('hsAvgTime').textContent         = avgTime;
}

function filterHistory(f,el){
  historyFilter=f;
  document.querySelectorAll('.filter-chip').forEach(x=>x.classList.remove('active'));
  el.classList.add('active');
  renderHistoryGrid();
}

function searchHistory(q){
  historySearch=q.toLowerCase();
  renderHistoryGrid();
}

function getFilteredHistory(){
  return analysisHistory.filter(e=>{
    const matchFilter=
      historyFilter==='all'      ?true:
      historyFilter==='positive' ?e.positives.length>0:
      historyFilter==='clean'    ?e.positives.length===0:
      historyFilter==='detected' ?e.boxes.length>0:true;
    const p=e.patient||{};
    const matchSearch=!historySearch||
      e.filename.toLowerCase().includes(historySearch)||
      (p.name&&p.name.toLowerCase().includes(historySearch))||
      (p.phone&&p.phone.includes(historySearch))||
      e.positives.some(([n])=>n.toLowerCase().includes(historySearch))||
      e.predictions.some(([n])=>n.toLowerCase().includes(historySearch));
    return matchFilter&&matchSearch;
  });
}

function renderHistoryPage(){
  updateHistoryStats();
  renderHistoryGrid();
}

function renderHistoryGrid(){
  const grid =document.getElementById('historyGrid');
  const empty=document.getElementById('historyEmpty');
  const items=getFilteredHistory();

  if(items.length===0){
    grid.style.display='none';
    empty.style.display='block';
    if(analysisHistory.length>0){
      empty.querySelector('h4').textContent='No results match your filter';
      empty.querySelector('p').textContent='Try adjusting the filter or search term.';
      empty.querySelector('button').style.display='none';
    } else {
      empty.querySelector('h4').textContent='No Analysis History Yet';
      empty.querySelector('p').textContent='Run your first X-ray analysis and it will appear here automatically.';
      empty.querySelector('button').style.display='inline-block';
    }
    return;
  }

  empty.style.display='none';
  grid.style.display='grid';
  grid.innerHTML=items.map(e=>{
    const dt=new Date(e.timestamp);
    const dateStr=dt.toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'})
                 +' '+dt.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'});
    const isPos=e.positives.length>0;
    const topPos=e.positives.slice(0,3).map(([n,v])=>
      `<span class="hcard-badge pos">${n} ${(v*100).toFixed(0)}%</span>`).join('');
    const thumbHtml=e.thumb
      ?`<img class="hcard-thumb" src="${e.thumb}" alt="thumb">`
      :`<div class="hcard-thumb-placeholder"><svg width="36" height="36" viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="18" height="18" rx="3" stroke="rgba(255,255,255,.2)" stroke-width="1.5"/><path d="M7 12h2l2-4 2 8 2-4h2" stroke="rgba(255,255,255,.2)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></div>`;
    const p=e.patient||{};
    const patientLine=p.name?`<div class="hcard-patient">
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" stroke="currentColor" stroke-width="2"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      ${escHtml(p.name)}${p.age?' · '+escHtml(p.age)+' yrs':''}
    </div>`:'';
    return `<div class="hcard" onclick="openHistoryModal(${e.id})">
      <div style="position:relative">
        ${thumbHtml}
        <div class="hcard-pos-indicator ${isPos?'':'clean'}">${isPos?e.positives.length+' POSITIVE':'CLEAN'}</div>
      </div>
      <div class="hcard-body">
        <div class="hcard-header">
          <div class="hcard-name">${escHtml(e.filename)}</div>
          <div class="hcard-date">${dateStr}</div>
        </div>
        ${patientLine}
        <div class="hcard-stats">
          <div class="hcard-stat"><div class="v">${e.positives.length}</div><div class="l">Positive</div></div>
          <div class="hcard-stat"><div class="v">${e.boxes.length}</div><div class="l">Detected</div></div>
          <div class="hcard-stat"><div class="v">${e.elapsed.toFixed(1)}s</div><div class="l">Time</div></div>
        </div>
        ${isPos?`<div class="hcard-top-finding">Top: <strong>${escHtml(e.top_class)}</strong> · ${(e.predictions[0][1]*100).toFixed(1)}%</div>`:`<div class="hcard-top-finding" style="color:var(--green)">✓ No significant findings above threshold</div>`}
        <div class="hcard-badges">${topPos}${e.boxes.length?`<span class="hcard-badge info">${e.boxes.length} bbox</span>`:''}</div>
        <div class="hcard-actions">
          <button class="hcard-btn view" onclick="event.stopPropagation();openHistoryModal(${e.id})">View Details</button>
          <button class="hcard-btn del"  onclick="event.stopPropagation();deleteHistoryEntry(${e.id})">Delete</button>
        </div>
      </div>
    </div>`;
  }).join('');
}

function openHistoryModal(id){
  const e=analysisHistory.find(x=>x.id===id);
  if(!e) return;
  const dt=new Date(e.timestamp);
  const dateStr=dt.toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'})+' '+dt.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'});
  document.getElementById('hmodalTitle').textContent=e.filename;

  const positives=e.predictions.filter(([,v])=>v>=0.5);
  const sorted=e.predictions.slice().sort((a,b)=>b[1]-a[1]);
  const p=e.patient||{};

  const patientSection=p.name||p.age||p.phone?`
    <div style="background:var(--patient-block-bg);border:1px solid var(--patient-block-border);border-radius:8px;padding:12px 16px;margin-bottom:16px;display:flex;gap:20px;flex-wrap:wrap">
      ${p.name?`<div><div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--teal2);margin-bottom:2px">Patient</div><div style="font-size:13px;font-weight:600;color:var(--text)">${escHtml(p.name)}</div></div>`:''}
      ${p.age?`<div><div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--teal2);margin-bottom:2px">Age</div><div style="font-size:13px;font-weight:600;color:var(--text)">${escHtml(p.age)} yrs</div></div>`:''}
      ${p.phone?`<div><div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--teal2);margin-bottom:2px">Phone</div><div style="font-size:13px;font-weight:600;color:var(--text)">${escHtml(p.phone)}</div></div>`:''}
    </div>`:'';

  const detImgHtml =e.det_img_b64 ?`<img src="${e.det_img_b64}"  alt="Detection" style="width:100%">` :`<div style="height:180px;display:flex;align-items:center;justify-content:center;color:var(--text3);font-size:12px">No detection image</div>`;
  const gcamImgHtml=e.gcam_img_b64?`<img src="${e.gcam_img_b64}" alt="Grad-CAM"  style="width:100%">` :`<div style="height:180px;display:flex;align-items:center;justify-content:center;color:var(--text3);font-size:12px">No heatmap</div>`;

  const topRows=sorted.slice(0,14).map(([n,v])=>{
    const pos=v>=0.5;
    const clr=v>=0.7?'var(--red)':v>=0.5?'var(--amber)':'var(--teal)';
    return `<div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border)">
      <div style="font-size:11px;font-weight:500;flex:1;color:var(--text)">${n}</div>
      <div style="flex:1.5;height:4px;background:var(--border);border-radius:99px;overflow:hidden"><div style="width:${Math.round(v*100)}%;height:100%;border-radius:99px;background:${clr}"></div></div>
      <div style="font-size:11px;font-weight:700;font-family:'DM Mono',monospace;min-width:40px;text-align:right;color:${clr}">${(v*100).toFixed(1)}%</div>
      <span style="font-size:9px;font-weight:700;padding:2px 7px;border-radius:99px;${pos?'background:var(--red2);color:var(--red)':'background:var(--green2);color:var(--green)'}">${pos?'POS':'neg'}</span>
    </div>`;
  }).join('');

  const boxRows=e.boxes.length
    ?e.boxes.map(b=>`<div style="display:flex;gap:10px;padding:5px 0;border-bottom:1px solid var(--border);font-size:11px"><div style="flex:1;font-weight:500;color:var(--text)">${b.class}</div><div style="font-weight:700;color:var(--red)">${(b.conf*100).toFixed(1)}%</div><div style="font-family:'DM Mono',monospace;color:var(--text3)">[${b.x1},${b.y1},${b.x2},${b.y2}]</div></div>`).join('')
    :`<div style="font-size:11px;color:var(--text3);padding:8px 0">No bounding box detections at analysis threshold.</div>`;

  document.getElementById('hmodalBody').innerHTML=`
    ${patientSection}
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px">
      <div style="background:var(--surface2);border-radius:8px;padding:12px;text-align:center"><div style="font-size:18px;font-weight:700;font-family:'DM Mono',monospace;color:${positives.length?'var(--red)':'var(--green)'}">${positives.length}</div><div style="font-size:9px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-top:2px">Positive</div></div>
      <div style="background:var(--surface2);border-radius:8px;padding:12px;text-align:center"><div style="font-size:18px;font-weight:700;font-family:'DM Mono',monospace;color:var(--text)">${e.boxes.length}</div><div style="font-size:9px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-top:2px">Detections</div></div>
      <div style="background:var(--surface2);border-radius:8px;padding:12px;text-align:center"><div style="font-size:18px;font-weight:700;font-family:'DM Mono',monospace;color:var(--text)">${e.elapsed.toFixed(2)}s</div><div style="font-size:9px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-top:2px">Time</div></div>
      <div style="background:var(--surface2);border-radius:8px;padding:12px;text-align:center"><div style="font-size:11px;font-weight:700;color:var(--teal)">${dateStr}</div><div style="font-size:9px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-top:2px">Date</div></div>
    </div>
    <div class="hmodal-imgs">
      <div><div class="hmodal-img-wrap">${detImgHtml}</div><div class="hmodal-img-label">YOLOv8m Detection</div></div>
      <div><div class="hmodal-img-wrap">${gcamImgHtml}</div><div class="hmodal-img-label">Grad-CAM Heatmap</div></div>
    </div>
    <div style="margin-bottom:20px">
      <div style="font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:var(--teal);margin-bottom:10px;padding-bottom:6px;border-bottom:1.5px solid var(--teal3)">Classification Results</div>
      ${topRows}
    </div>
    <div>
      <div style="font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:var(--teal);margin-bottom:10px;padding-bottom:6px;border-bottom:1.5px solid var(--teal3)">Detection Boxes</div>
      ${boxRows}
    </div>`;

  document.getElementById('hmodalOverlay').classList.add('open');
}

function closeHistoryModal(e){
  if(!e||e.target===document.getElementById('hmodalOverlay')){
    document.getElementById('hmodalOverlay').classList.remove('open');
  }
}

function deleteHistoryEntry(id){
  analysisHistory=analysisHistory.filter(x=>x.id!==id);
  updateHistoryBadge();
  updateHistoryStats();
  renderHistoryGrid();
  toast('Entry removed from history.');
}

function clearHistory(){
  if(!analysisHistory.length){toast('History is already empty.');return;}
  if(!confirm('Clear all '+analysisHistory.length+' history entries?')) return;
  analysisHistory=[];
  updateHistoryBadge();
  updateHistoryStats();
  renderHistoryGrid();
  toast('History cleared.');
}

function exportHistoryCSV(){
  if(!analysisHistory.length){toast('No history to export.');return;}
  const rows=[['ID','Filename','PatientName','PatientAge','PatientPhone','Timestamp','Elapsed(s)','Device','PositiveCount','DetectionCount','TopClass','TopConfidence(%)']];
  analysisHistory.forEach(e=>{
    const p=e.patient||{};
    rows.push([e.id,e.filename,p.name||'',p.age||'',p.phone||'',e.timestamp,e.elapsed,e.device,e.positives.length,e.boxes.length,e.top_class,e.predictions.length?(e.predictions[0][1]*100).toFixed(2):0]);
  });
  const csv=rows.map(r=>r.map(c=>'"'+String(c).replace(/"/g,'""')+'"').join(',')).join('\\n');
  const blob=new Blob([csv],{type:'text/csv'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url; a.download=`CliniScan_History_${Date.now()}.csv`; a.click();
  URL.revokeObjectURL(url);
  toast('CSV exported successfully.');
}

// ══════════════ HOME PAGE INIT ══════════════
const cg=document.getElementById('classGrid');
Object.keys(AUC_DATA).forEach(c=>{
  cg.innerHTML+=`<div class="class-chip"><div class="chip-dot"></div>${c}<span class="auc">${AUC_DATA[c].toFixed(2)}%</span></div>`;
});

const ab=document.getElementById('aucBars');
Object.entries(AUC_DATA).sort((a,b)=>b[1]-a[1]).forEach(([n,v])=>{
  const pct=Math.round((v-80)/(100-80)*100);
  ab.innerHTML+=`<div class="auc-row"><div class="auc-name">${n}</div><div class="auc-bar-wrap"><div class="auc-bar" style="width:${pct}%"></div></div><div class="auc-val">${v.toFixed(2)}%</div></div>`;
});

// ══════════════ FILE UPLOAD ══════════════
function handleFile(e){
  const f=e.target.files[0]; if(!f) return;
  uploadedFile=f;
  const r=new FileReader();
  r.onload=ev=>{
    const img=document.getElementById('previewImg');
    img.src=ev.target.result; img.style.display='block';
    document.getElementById('uploadPlaceholder').style.display='none';
    document.getElementById('uploadZone').classList.add('has-image');
    document.getElementById('uploadedName').textContent=f.name;
    document.getElementById('uploadedName').style.display='block';
    document.getElementById('analyzeBtn').disabled=false;
  };
  r.readAsDataURL(f);
}

const uz=document.getElementById('uploadZone');
uz.addEventListener('dragover',e=>{e.preventDefault();uz.classList.add('drag')});
uz.addEventListener('dragleave',()=>uz.classList.remove('drag'));
uz.addEventListener('drop',e=>{
  e.preventDefault(); uz.classList.remove('drag');
  const f=e.dataTransfer.files[0]; if(!f) return;
  const dt=new DataTransfer(); dt.items.add(f);
  document.getElementById('fileInput').files=dt.files;
  handleFile({target:{files:[f]}});
});

// ══════════════ ANALYSIS ══════════════
function clearFieldError(groupId){
  document.getElementById(groupId)&&document.getElementById(groupId).classList.remove('has-error');
}

function validatePatientForm(){
  const fields=[
    {id:'patientName', group:'grpName',  msg:'Full name is required'},
    {id:'patientAge',  group:'grpAge',   msg:'Age is required'},
    {id:'patientPhone',group:'grpPhone', msg:'Phone number is required'},
  ];
  let valid=true;
  fields.forEach(f=>{
    const el=document.getElementById(f.id);
    const grp=document.getElementById(f.group);
    if(!el||!el.value.trim()){
      grp&&grp.classList.add('has-error');
      valid=false;
    } else {
      grp&&grp.classList.remove('has-error');
    }
  });
  return valid;
}

async function runAnalysis(){
  if(!uploadedFile){toast('Please upload an X-ray image first.');return;}
  if(!validatePatientForm()){
    toast('Please fill in all patient details.', true);
    document.getElementById('patientName').scrollIntoView({behavior:'smooth',block:'center'});
    return;
  }
  const btn=document.getElementById('analyzeBtn');
  btn.disabled=true; btn.classList.add('loading');
  const conf=document.getElementById('confVal').textContent;
  const iou=document.getElementById('iouVal').textContent;
  const doGcam=document.getElementById('tGradcam').classList.contains('on');
  const doBbox=document.getElementById('tBbox').classList.contains('on');
  const patient=getPatientInfo();
  const fd=new FormData();
  fd.append('file',uploadedFile);
  fd.append('conf_thresh',conf);
  fd.append('iou_thresh',iou);
  fd.append('show_gradcam',doGcam);
  fd.append('show_bbox',doBbox);
  fd.append('patient_name',patient.name);
  fd.append('patient_age',patient.age);
  fd.append('patient_phone',patient.phone);
  try{
    const resp=await fetch('/analyze',{method:'POST',body:fd});
    if(!resp.ok) throw new Error(await resp.text());
    const data=await resp.json();
    lastResult=data;
    lastResult._patient=patient;
    renderResults(data,patient);
    toast('Analysis complete — '+data.positives.length+' positive finding(s).');
  }catch(err){
    toast('Error: '+err.message,true);
  }finally{
    btn.disabled=false; btn.classList.remove('loading');
  }
}

function renderResults(data,patient){
  document.getElementById('emptyState').style.display='none';
  document.getElementById('resultsContent').style.display='block';
  document.getElementById('rDetections').textContent=data.boxes.length;
  document.getElementById('rPositives').textContent=data.positives.length;
  document.getElementById('rTime').textContent=data.elapsed.toFixed(1)+'s';
  // Patient block in results
  document.getElementById('resultPatientBlock').innerHTML=patientBlockHTML(patient);
  const dw=document.getElementById('detectionImgWrap');
  if(data.det_img_b64) dw.innerHTML=`<img src="${data.det_img_b64}" alt="Detection"><div class="hm-label">Detection View — YOLOv8m</div>`;
  const hw=document.getElementById('heatmapImgWrap');
  if(data.gcam_img_b64) hw.innerHTML=`<img src="${data.gcam_img_b64}" alt="Grad-CAM"><div class="hm-label">Grad-CAM — ${data.top_class}</div>`;
  buildFindings(data.predictions);
  buildReportPage(data,patient);
  saveToHistory(data, uploadedFile?uploadedFile.name:'xray.png', data.det_img_b64||null, patient);
}

function buildFindings(preds){
  const fl=document.getElementById('findingsList'); fl.innerHTML='';
  preds.forEach(([n,v])=>{
    const pos=v>=0.5;
    const clr=v>=0.7?'var(--red)':v>=0.5?'var(--amber)':'var(--teal)';
    fl.innerHTML+=`<div class="finding-row">
      <div class="finding-name">${n}</div>
      <div class="finding-bar-wrap"><div class="finding-bar" style="width:${Math.round(v*100)}%;background:${clr}"></div></div>
      <div class="finding-pct" style="color:${clr}">${(v*100).toFixed(1)}%</div>
      <span class="finding-status ${pos?'pos':'neg'}">${pos?'POS':'neg'}</span>
    </div>`;
  });
}

function switchTab(t,el){
  ['detection','heatmap','findings'].forEach(id=>document.getElementById('tab-'+id).style.display='none');
  document.querySelectorAll('.rtab').forEach(x=>x.classList.remove('active'));
  document.getElementById('tab-'+t).style.display='block';
  el.classList.add('active');
}

async function downloadReport(){
  if(!uploadedFile){toast('Run an analysis first.');return;}
  toast('Generating PDF report…');
  const conf=document.getElementById('confVal').textContent;
  const iou=document.getElementById('iouVal').textContent;
  const patient=lastResult&&lastResult._patient?lastResult._patient:getPatientInfo();
  const fd=new FormData();
  fd.append('file',uploadedFile);
  fd.append('conf_thresh',conf);
  fd.append('iou_thresh',iou);
  fd.append('patient_name',patient.name||'');
  fd.append('patient_age',patient.age||'');
  fd.append('patient_phone',patient.phone||'');
  try{
    const resp=await fetch('/report/pdf',{method:'POST',body:fd});
    if(!resp.ok) throw new Error(await resp.text());
    const blob=await resp.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url; a.download=`CliniScan_Report_${Date.now()}.pdf`; a.click();
    URL.revokeObjectURL(url);
    toast('PDF downloaded successfully.');
  }catch(err){
    toast('PDF error: '+err.message,true);
  }
}

function buildReportPage(data,patient){
  const dt=new Date();
  const dateStr=dt.toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'})+' '+dt.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'});
  const positives=data.predictions.filter(([,v])=>v>=0.5);
  const [topN,topV]=data.predictions[0]||['N/A',0];

  const patientMetaItems=[];
  if(patient&&patient.name)  patientMetaItems.push(`<div class="meta-item"><label>Patient</label><strong>${escHtml(patient.name)}</strong></div>`);
  if(patient&&patient.age)   patientMetaItems.push(`<div class="meta-item"><label>Age</label><strong>${escHtml(patient.age)} yrs</strong></div>`);
  if(patient&&patient.phone) patientMetaItems.push(`<div class="meta-item"><label>Phone</label><strong>${escHtml(patient.phone)}</strong></div>`);

  let tableRows=data.predictions.map(([n,v])=>{
    const pos=v>=0.5;
    const aucPct=AUC_DATA[n]!==undefined?AUC_DATA[n].toFixed(2)+'%':'—';
    return `<tr>
      <td>${n}</td>
      <td style="font-family:'DM Mono',monospace">${(v*100).toFixed(2)}%</td>
      <td><span class="badge ${pos?'positive':'negative'}">${pos?'POSITIVE':'Negative'}</span></td>
      <td style="font-family:'DM Mono',monospace">${aucPct}</td>
    </tr>`;
  }).join('');

  let boxRows=data.boxes.length?data.boxes.map(b=>`<tr>
    <td>${b.class}</td>
    <td><span class="badge positive">${(b.conf*100).toFixed(1)}%</span></td>
    <td style="font-family:'DM Mono',monospace;font-size:11px">[${b.x1},${b.y1},${b.x2},${b.y2}]</td>
  </tr>`).join(''):`<tr><td colspan="3" style="text-align:center;color:var(--text3)">No detections at current threshold.</td></tr>`;

  let aucRows=Object.entries(AUC_DATA).map(([n,v])=>{
    const g=v>=95?'Excellent':v>=92?'Very Good':v>=90?'Good':'Moderate';
    const gc=v>=95?'high':v>=90?'negative':'';
    return `<tr><td>${n}</td><td style="font-family:'DM Mono',monospace">${v.toFixed(2)}%</td><td><span class="badge ${gc}">${g}</span></td></tr>`;
  }).join('');

  const detImg=data.det_img_b64?`<div class="hm-img"><img src="${data.det_img_b64}" alt="Detection"><div class="hm-label">YOLOv8m Detection</div></div>`:`<div class="hm-img" style="min-height:160px"><p style="font-size:11px;color:var(--text3)">No detection image</p></div>`;
  const gcamImg=data.gcam_img_b64?`<div class="hm-img"><img src="${data.gcam_img_b64}" alt="Grad-CAM"><div class="hm-label">Grad-CAM — ${data.top_class}</div></div>`:`<div class="hm-img" style="min-height:160px"><p style="font-size:11px;color:var(--text3)">No heatmap</p></div>`;

  const alertCls=positives.length>0?'warn':'ok';
  const alertMsg=positives.length>0
    ?`<strong>${positives.length} positive finding(s) detected above 50% threshold.</strong> ${positives.map(([n,v])=>`${n} (${(v*100).toFixed(1)}%)`).join(', ')} — please consult a qualified radiologist.`
    :`<strong>No positive findings detected above 50% threshold.</strong> All 14 classes below threshold. Always confirm with a radiologist.`;

  const patientSection=patient&&(patient.name||patient.age||patient.phone)?`
    <div class="report-section">
      <div class="report-section-title">Patient Information</div>
      <div style="display:flex;gap:24px;flex-wrap:wrap;background:var(--patient-block-bg);border:1px solid var(--patient-block-border);border-radius:10px;padding:16px 20px">
        ${patient.name?`<div><div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--teal2);margin-bottom:4px">Full Name</div><div style="font-size:15px;font-weight:700;color:var(--text)">${escHtml(patient.name)}</div></div>`:''}
        ${patient.age?`<div><div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--teal2);margin-bottom:4px">Age</div><div style="font-size:15px;font-weight:700;color:var(--text)">${escHtml(patient.age)} years</div></div>`:''}
        ${patient.phone?`<div><div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--teal2);margin-bottom:4px">Phone</div><div style="font-size:15px;font-weight:700;color:var(--text)">${escHtml(patient.phone)}</div></div>`:''}
      </div>
    </div>`:'';

  document.getElementById('reportDoc').innerHTML=`
    <div class="report-header">
      <div><div style="font-size:10px;font-weight:600;color:rgba(167,243,208,.8);letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">Radiology AI Report</div>
      <h2>Chest X-Ray Analysis</h2><p>Automated Abnormality Detection — Research Grade</p></div>
      <div class="report-logo"><span>Clini<em style="font-style:italic;color:#99F6E4">Scan</em></span>EfficientNet-B3 + YOLOv8m</div>
    </div>
    <div class="report-meta">
      <div class="meta-item"><label>Patient File</label><strong>${uploadedFile?escHtml(uploadedFile.name):'xray.png'}</strong></div>
      ${patientMetaItems.join('')}
      <div class="meta-item"><label>Date</label><strong>${dateStr}</strong></div>
      <div class="meta-item"><label>Analysis Time</label><strong>${data.elapsed.toFixed(2)} s</strong></div>
      <div class="meta-item"><label>Device</label><strong>${data.device.toUpperCase()}</strong></div>
      <div class="meta-item"><label>Mean AUC</label><strong>91.93%</strong></div>
    </div>
    <div class="report-body">
      ${patientSection}
      <div class="report-section">
        <div class="report-section-title">Executive Summary</div>
        <div class="report-summary-grid">
          <div class="rsm ${data.boxes.length?'alert':''}"><div class="rv">${data.boxes.length}</div><div class="rl">Detections</div></div>
          <div class="rsm ${positives.length?'alert':''}"><div class="rv">${positives.length}</div><div class="rl">Positive Classes</div></div>
          <div class="rsm"><div class="rv" style="font-size:12px">${topN}</div><div class="rl">Top Finding</div></div>
          <div class="rsm"><div class="rv">${(topV*100).toFixed(1)}%</div><div class="rl">Top Confidence</div></div>
        </div>
        <div class="alert-box ${alertCls}">${alertMsg}</div>
      </div>
      <div class="report-section">
        <div class="report-section-title">Visual Analysis</div>
        <div class="heatmap-demo">${detImg}${gcamImg}</div>
      </div>
      <div class="report-section">
        <div class="report-section-title">Classification Results</div>
        <table><thead><tr><th>Disease Class</th><th>Probability</th><th>Status</th><th>Model AUC</th></tr></thead><tbody>${tableRows}</tbody></table>
      </div>
      <div class="report-section">
        <div class="report-section-title">Detection Results (YOLOv8m)</div>
        <table><thead><tr><th>Class</th><th>Confidence</th><th>Bounding Box [x1,y1,x2,y2]</th></tr></thead><tbody>${boxRows}</tbody></table>
      </div>
      <div class="report-section">
        <div class="report-section-title">Per-Class AUC (Training Results)</div>
        <table><thead><tr><th>Disease Class</th><th>AUC Score</th><th>Grade</th></tr></thead><tbody>${aucRows}</tbody></table>
      </div>
    </div>
    <div class="report-footer">
      <p><strong>Disclaimer:</strong> CliniScan AI is for research purposes only. This report is NOT a clinical diagnosis. Always consult a qualified healthcare professional. EfficientNet-B3 + YOLOv8m · VinDr-CXR · Mean AUC 91.93%</p>
    </div>`;
}

function toast(msg,isErr=false){
  const t=document.getElementById('toast');
  t.textContent=msg;
  t.style.borderColor=isErr?'var(--red)':'var(--teal)';
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),3500);
}

// ── Health check
(async()=>{
  try{
    const h=await fetch('/health');
    const d=await h.json();
    const el=document.getElementById('footerDeviceText');
    if(el) el.textContent=`${d.device.toUpperCase()} · ${d.classifier==='loaded'?'Model Ready':'Demo Mode'}`;
  }catch{
    const el=document.getElementById('footerDeviceText');
    if(el) el.textContent='System Ready';
  }
})();

updateHistoryBadge();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)

@app.post("/analyze")
async def analyze_endpoint(
    file:          UploadFile = File(...),
    conf_thresh:   float = 0.15,
    iou_thresh:    float = 0.45,
    show_gradcam:  bool  = True,
    show_bbox:     bool  = True,
    patient_name:  str   = Form(""),
    patient_age:   str   = Form(""),
    patient_phone: str   = Form(""),
) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file uploaded.")
    try:
        rgb = to_rgb(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    t0             = time.time()
    preds          = classify(rgb)
    det_img, boxes = detect(rgb, conf_thresh, iou_thresh) if show_bbox else (rgb.copy(), [])
    gcam_img, top_cls = compute_gradcam(rgb, preds) if show_gradcam else (None, None)
    elapsed        = time.time() - t0
    positives      = [(n, c) for n, c in preds if c >= 0.5]
    det_b64        = ndarray_to_b64(det_img)  if det_img  is not None else None
    gcam_b64       = ndarray_to_b64(gcam_img) if gcam_img is not None else None
    gc.collect()

    return JSONResponse({
        "predictions":    preds,
        "positives":      positives,
        "boxes":          boxes,
        "top_class":      top_cls or (preds[0][0] if preds else "N/A"),
        "elapsed":        round(elapsed, 3),
        "device":         str(DEVICE),
        "det_img_b64":    det_b64,
        "gcam_img_b64":   gcam_b64,
        "patient_name":   patient_name,
        "patient_age":    patient_age,
        "patient_phone":  patient_phone,
    })

@app.post("/report/pdf")
async def report_pdf_endpoint(
    file:          UploadFile = File(...),
    conf_thresh:   float = 0.15,
    iou_thresh:    float = 0.45,
    patient_name:  str   = Form(""),
    patient_age:   str   = Form(""),
    patient_phone: str   = Form(""),
) -> StreamingResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file uploaded.")
    try:
        rgb = to_rgb(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    t0             = time.time()
    preds          = classify(rgb)
    det_img, boxes = detect(rgb, conf_thresh, iou_thresh)
    gcam_img, _    = compute_gradcam(rgb, preds)
    elapsed        = time.time() - t0

    pdf_bytes = build_pdf(
        preds         = preds,
        filename      = file.filename or "xray_upload.png",
        boxes         = boxes,
        elapsed       = elapsed,
        det_img       = det_img,
        gcam_img      = gcam_img,
        patient_name  = patient_name,
        patient_age   = patient_age,
        patient_phone = patient_phone,
    )
    gc.collect()

    fname = f"CliniScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )

@app.get("/health")
async def health() -> dict:
    return {
        "status":     "ok",
        "device":     str(DEVICE),
        "classifier": "loaded" if clf_model  is not None else "demo_mode",
        "detector":   "loaded" if yolo_model is not None else "unavailable",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), reload=False, workers=1)