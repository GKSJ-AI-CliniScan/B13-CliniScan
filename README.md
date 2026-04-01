# MediScanAI: Clinical Diagnostic Platform

![MediScanAI Workspace](https://raw.githubusercontent.com/mittalyash/AI-CliniScan/main/assets/ui.png) <!-- Replace with actual screenshot later -->

MediScanAI is an end-to-end full-stack medical machine learning platform. It automates the screening and precise geometric localization of 14 common chest abnormalities utilizing deep CNN architectures, wrapped in a secure, high-contrast clinical web interface.

> *Developed during the AI/ML Internship @ Infosys Springboard.*

---

## 🚀 Live Production Links
- **Vercel Edge Frontend (React/Vite):** [https://mediscanai-tawny.vercel.app](https://mediscanai-tawny.vercel.app)
- **Hugging Face Backend (FastAPI/Docker):** [https://mittalyash-mediscan-api.hf.space](https://mittalyash-mediscan-api.hf.space)

---

## 📁 Monorepo Structure

### `/frontend`
The sleek, high-contrast, React 18 interface designed to augment radiological workflows. Built with Vite and Tailwind CSS.
- **Features:** Dual-mode Analysis, Dynamic Bounding Box SVG overlays, Session persistence, Secure Auth (SHA-256), Clinical Printable Diagnostic Reports.

### `/backend`
The dual-architecture PyTorch intelligence engine, containerized for Hugging Face Spaces.
- **Classification:** `ResNet-50` backbone tailored for 14-class multi-label diagnosis (Peak 0.9449 AUC).
- **Detection:** `Faster R-CNN` (ResNet-50-FPN) for precise diagnostic bounding box generation.
- **Explainability:** Grad-CAM hooks inject accountability by dynamically rendering activation heatmaps.

### `/docs`
Contains the structured internship evaluation milestones outlining the entire software and machine learning development lifecycle.
- **M1:** Data Preparation, EDA & Setup
- **M2:** Baseline Training & Architectural Implementation
- **M3:** Advanced Optimizations & Explainable AI (Grad-CAM)
- **M4:** Full-Stack Integration & Cloud Deployment

---

## ⚙️ Running Locally
To run the full stack on your local machine:

1. **Boot Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

2. **Boot Frontend:**
```bash
cd frontend
npm install
VITE_API_URL=http://localhost:7860 npm run dev
```
