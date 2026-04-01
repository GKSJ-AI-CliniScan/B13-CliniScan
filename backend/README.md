---
title: MediScan API
emoji: 🏥
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# AI-CliniScan: Clinical Chest X-Ray Diagnosis AI

AI-CliniScan is a dual-architecture PyTorch machine learning pipeline designed to automate the screening and localization of 14 common chest abnormalities. It serves as the advanced intelligence backend for the **MediScanAI** clinical platform.

## 🚀 Live Production API
**Hugging Face Spaces Engine:** [https://mittalyash-mediscan-api.hf.space](https://mittalyash-mediscan-api.hf.space)

*(The frontend interface is hosted independently on Vercel at [https://mediscanai-tawny.vercel.app](https://mediscanai-tawny.vercel.app))*

## 🔬 Core Architectures
- **High-Accuracy Classification:** Powered by a `ResNet-50` backbone tailored for 14-class multi-label diagnosis (Peak **0.9449 AUC**).
- **Stable Object Detection:** Powered by a `Faster R-CNN` (ResNet-50-FPN) for precise diagnostic bounding box generation.
- **Explainable AI:** Grad-CAM hooks inject accountability by dynamically rendering activation heatmaps over the classified pathologies.

## 📁 Project Structure
- `app.py`: The live FastAPI application serving endpoints (`/predict`) in our Docker container.
- `classification/`: Model architecture, training, and standalone inference for multi-label diagnosis.
- `detection/`: Faster R-CNN implementation for abnormality localization.
- `data_prep/`: DICOM-to-PNG extraction pipelines and YOLO bounding box engineering.
- `docs/`: Professional Milestone generated reports (M1-M4) documenting the entire ML lifecycle.
- `Dockerfile`: Configuration for our Hugging Face Spaces deployment.

## 📦 Deployment Container
This repository is configured to deploy instantly as a Docker Space on Hugging Face. The heavy inference payloads operations are containerized, leveraging PyTorch and OpenCV dynamically under the `/predict` POST endpoint.

---
*Developed during the AI/ML Internship @ Infosys Springboard.*
