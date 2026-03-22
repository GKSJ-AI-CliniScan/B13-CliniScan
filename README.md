# AI-CliniScan: Clinical Chest X-Ray Diagnosis & Localization

AI-CliniScan is an advanced medical imaging project built to automate the screening and localization of 14 common chest abnormalities using deep learning.

## 🚀 Accomplishments (Milestone 1-3)
- **High-Accuracy Classification:** Achieved a peak **0.9449 AUC** using a custom ResNet-50 architecture.
- **Stable Object Detection:** Implemented and trained a Faster R-CNN (ResNet-50-FPN) for precise localization.
- **Deployment-Ready Data:** Preprocessed 15,000 images into localized PNG formats with clinical bounding boxes.
- **Local Acceleration:** Fully optimized for Apple Silicon (MPS) and CPU training on local hardware.

## 📂 Project Structure
- `classification/`: Model architecture, training, and inference for multi-label diagnosis.
- `detection/`: Faster R-CNN implementation for abnormality localization.
- `data_prep/`: DICOM-to-PNG conversion and YOLO-to-Torchbox label engineering.
- `results/`: Visual evidence of the AI identifying and localizing abnormalities.
- `docs/`: Professional Milestone 1, 2, and 3 reports generated for internship evaluation.

## 🔬 How to Run Inference
### 1. Classification Test
```bash
python3 classification/inference.py
```
This will pick a random X-ray using the weights in `models/best_resnet_classification.pth` and output the top-3 abnormalities.

### 2. Detection Test
```bash
python3 detection/inference.py
```
This will use `models/best_faster_rcnn_detection.pth` to generate bounding boxes around detected findings and save them to the `results/` folder.

## 📦 Requirements
Install dependencies via:
```bash
pip install -r requirements.txt
```

---
*Developed during the AI/ML Internship @ Infosys Springboard.*
