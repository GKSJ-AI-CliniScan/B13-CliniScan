"""
CliniScan Local UI — Backend
Run:  pip install fastapi uvicorn python-multipart torch torchvision albumentations opencv-python-headless ultralytics grad-cam reportlab
Then: uvicorn app:app --reload --port 8000
"""

import os, io, time, warnings, base64, json
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS_MODEL_PATH  = os.path.join(BASE, "milestone3_cls", "ckpt", "Exp2_AdamW.pt")
YOLO_MODEL_PATH = os.path.join(BASE, "milestone3_det", "D2_LowLR", "weights", "last.pt")

CLASS_NAMES = [
    "Aortic enlargement","Atelectasis","Calcification","Cardiomegaly",
    "Consolidation","ILD","Infiltration","Lung Opacity",
    "Nodule/Mass","Other lesion","Pleural effusion",
    "Pleural thickening","Pneumothorax","Pulmonary fibrosis",
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_tfm = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])

# ── Load models ──────────────────────────────────────────────
cls_model  = None
yolo_model = None

def load_models():
    global cls_model, yolo_model
    try:
        m = efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
        if os.path.exists(CLS_MODEL_PATH):
            ckpt = torch.load(CLS_MODEL_PATH, map_location=DEVICE, weights_only=False)
            state = ckpt['model'] if 'model' in ckpt else ckpt
            m.load_state_dict(state)
            print(f"✅ Classification model loaded")
        else:
            print(f"⚠️  Model not found: {CLS_MODEL_PATH}")
            print("   Running in demo mode with random weights")
        cls_model = m.to(DEVICE)
        cls_model.eval()
    except Exception as e:
        print(f"❌ Cls error: {e}")
    try:
        from ultralytics import YOLO
        if os.path.exists(YOLO_MODEL_PATH):
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"✅ YOLO detection model loaded")
        else:
            print(f"⚠️  YOLO not found: {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"❌ YOLO error: {e}")

load_models()

# ── Helpers ──────────────────────────────────────────────────
def preprocess(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, "Invalid image")
    h, w = img.shape[:2]
    img  = cv2.resize(img[int(h*.02):int(h*.98), int(w*.02):int(w*.98)], (w, h))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def classify(rgb):
    if cls_model is None:
        import random; return [(c, round(random.uniform(.05,.90),4)) for c in CLASS_NAMES]
    t = eval_tfm(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.sigmoid(cls_model(t)).cpu().numpy()[0]
    return [(CLASS_NAMES[i], float(round(p[i],4))) for i in range(NUM_CLASSES)]

def detect(rgb):
    if yolo_model is None: return [], None
    try:
        tmp = os.path.join(tempfile.gettempdir(), "_cs.png")
        cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        res   = yolo_model.predict(tmp, conf=0.25, verbose=False)[0]
        boxes = []
        if res.boxes:
            for b in res.boxes:
                x1,y1,x2,y2 = b.xyxy[0].tolist(); cls = int(b.cls[0])
                boxes.append({"x1":round(x1),"y1":round(y1),"x2":round(x2),"y2":round(y2),
                               "conf":round(float(b.conf[0]),3),
                               "class":CLASS_NAMES[cls] if cls<NUM_CLASSES else f"cls{cls}"})
        vis = res.plot()
        _, buf = cv2.imencode(".png", vis)
        return boxes, base64.b64encode(buf).decode()
    except Exception as e:
        print(f"Det error: {e}"); return [], None

def gradcam(rgb, target=None):
    if cls_model is None: return None
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        cam  = GradCAM(model=cls_model, target_layers=[cls_model.features[-1]])
        t    = eval_tfm(image=rgb)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p = torch.sigmoid(cls_model(t)).cpu().numpy()[0]
        top  = int(np.argmax(p)) if target is None else target
        gc   = cam(t, [ClassifierOutputTarget(top)])[0]
        r224 = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        ov   = show_cam_on_image(r224.astype(np.float32)/255.0, gc, use_rgb=True)
        _, buf = cv2.imencode(".png", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode()
    except Exception as e:
        print(f"CAM error: {e}"); return None

def make_pdf(preds, fname, boxes):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import inch
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             rightMargin=inch, leftMargin=inch,
                             topMargin=inch, bottomMargin=inch)
    st = getSampleStyleSheet(); story = []
    story.append(Paragraph("CliniScan — AI Chest X-Ray Report", st["Title"]))
    story.append(Spacer(1,.15*inch))
    story.append(Paragraph(f"File: {fname}  |  {time.strftime('%Y-%m-%d %H:%M')}", st["Normal"]))
    story.append(Paragraph("Research use only — not a clinical diagnosis.", st["Normal"]))
    story.append(Spacer(1,.25*inch))
    story.append(Paragraph("Classification Results", st["Heading2"]))
    data = [["Disease","Confidence","Status"]]
    for n,c in sorted(preds, key=lambda x:-x[1]):
        data.append([n, f"{c:.1%}", "DETECTED" if c>=.5 else "Normal"])
    t = Table(data, colWidths=[3*inch,1.5*inch,1.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1F4E79")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),10),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F2F2F2")]),
        ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(t)
    if boxes:
        story.append(Spacer(1,.2*inch))
        story.append(Paragraph("Detection Bounding Boxes", st["Heading2"]))
        dd = [["Finding","Confidence","Location"]]
        for b in boxes:
            dd.append([b["class"],f"{b['conf']:.1%}",
                       f"({b['x1']},{b['y1']}) to ({b['x2']},{b['y2']})"])
        dt = Table(dd, colWidths=[2.5*inch,1.5*inch,2*inch])
        dt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2E75B6")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTSIZE",(0,0),(-1,-1),10),
            ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F2F2F2")]),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ]))
        story.append(dt)
    doc.build(story); buf.seek(0); return buf

# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(title="CliniScan")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve the HTML frontend
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html not found — place it in the same folder as app.py</h1>"

@app.get("/health")
def health():
    return {"status":"ok","cls_model":cls_model is not None,
            "yolo_model":yolo_model is not None,"device":str(DEVICE)}

@app.get("/model-info")
def model_info():
    return {
        "classification":{"name":"EfficientNet-B0 (Exp2_AdamW)","backbone":"EfficientNet-B0",
                          "optimizer":"AdamW","lr":"5e-4","epochs":12,
                          "val_auc":0.9213,"baseline_auc":0.8858,"improvement":"+3.55%"},
        "detection":{"name":"YOLOv8s (D2_LowLR)","model":"YOLOv8s",
                     "optimizer":"AdamW","lr":"5e-4","epochs":15,
                     "map50":0.0592,"map50_95":0.0251,"baseline_map50":0.0658},
        "cls_experiments":[
            {"name":"Exp1_Adam",    "auc":0.9104,"optimizer":"Adam",  "lr":"1e-3"},
            {"name":"Exp2_AdamW",   "auc":0.9213,"optimizer":"AdamW", "lr":"5e-4"},
            {"name":"Exp3_SGD",     "auc":0.9094,"optimizer":"SGD",   "lr":"1e-2"},
            {"name":"Exp4_DenseNet","auc":0.8304,"optimizer":"Adam",  "lr":"1e-3"},
            {"name":"Baseline M2",  "auc":0.8858,"optimizer":"Adam",  "lr":"1e-3"},
        ],
        "det_experiments":[
            {"name":"D1_YOLOv8m",  "map50":0.0463,"prec":0.6951,"rec":0.0527},
            {"name":"D2_LowLR",    "map50":0.0592,"prec":0.3683,"rec":0.0725},
            {"name":"D3_LowThresh","map50":0.0574,"prec":0.1749,"rec":0.0624},
            {"name":"D4_SGD",      "map50":0.0446,"prec":0.5503,"rec":0.0436},
            {"name":"Baseline M2", "map50":0.0658,"prec":0.3793,"rec":0.0821},
        ],
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    run_detection: bool = True,
    run_gradcam: bool = True,
):
    t0    = time.time()
    data  = await file.read()
    rgb   = preprocess(data)
    preds = classify(rgb)

    boxes, det_vis = [], None
    if run_detection: boxes, det_vis = detect(rgb)

    cam_b64 = None
    if run_gradcam: cam_b64 = gradcam(rgb)

    _, buf   = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    orig_b64 = base64.b64encode(buf).decode()
    detected = sorted([{"disease":n,"confidence":c} for n,c in preds if c>=threshold],
                      key=lambda x:-x["confidence"])
    return {
        "filename":    file.filename,
        "predictions": [{"disease":n,"confidence":c} for n,c in preds],
        "detected":    detected,
        "boxes":       boxes,
        "images":{"original":orig_b64,"detection":det_vis,"gradcam":cam_b64},
        "threshold":   threshold,
        "time_ms":     round((time.time()-t0)*1000),
    }

@app.post("/batch")
async def batch_predict(files: List[UploadFile] = File(...), threshold: float = 0.5):
    results = []
    for f in files:
        try:
            data  = await f.read()
            rgb   = preprocess(data)
            preds = classify(rgb)
            det   = [n for n,c in preds if c>=threshold]
            results.append({"filename":f.filename,"status":"success",
                             "detected":det,"findings":len(det),
                             "top_conf":round(max(c for _,c in preds),4),
                             "predictions":[{"disease":n,"confidence":c} for n,c in preds]})
        except Exception as e:
            results.append({"filename":f.filename,"status":"error","error":str(e)})
    return {"total":len(results),
            "processed":sum(1 for r in results if r["status"]=="success"),
            "errors":sum(1 for r in results if r["status"]=="error"),
            "results":results}

@app.post("/report")
async def generate_report(file: UploadFile = File(...)):
    data  = await file.read()
    rgb   = preprocess(data)
    preds = classify(rgb)
    boxes, _ = detect(rgb)
    pdf   = make_pdf(preds, file.filename, boxes)
    return StreamingResponse(pdf, media_type="application/pdf",
        headers={"Content-Disposition":"attachment; filename=cliniscan_report.pdf"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run("app:app", host=host, port=port, reload=False)