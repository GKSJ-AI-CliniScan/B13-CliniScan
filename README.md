<div align="center">

# 🫁 CliniScan
### Lung Abnormality Detection on Chest X-rays using AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Batch](https://img.shields.io/badge/Batch-B13-orange?style=flat-square)](https://github.com/GKSJ-AI-CliniScan)

<br/>

> An AI-powered system that automatically detects and localizes lung abnormalities from chest X-ray images using deep learning — built to assist radiologists by identifying pathological findings such as opacities, consolidations, fibrosis, masses, and nodules.

<br/>

**Dataset:** VinBigData Chest X-ray (5,000 images) &nbsp;|&nbsp; **Author:** Lucky Soni &nbsp;|&nbsp; **Batch:** B13

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Milestones](#-milestones)
- [Results](#-results)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Outputs & Visualizations](#-outputs--visualizations)

---

## 🔬 Project Overview

CliniScan is a deep learning pipeline for automated chest X-ray analysis. The system addresses one of the most critical challenges in modern radiology — the high volume of chest X-rays requiring expert interpretation. By leveraging state-of-the-art CNN architectures and object detection models, CliniScan:

- **Classifies** which abnormalities are present in a chest X-ray (multi-label classification)
- **Localizes** where abnormalities are located using bounding boxes (object detection)
- **Interprets** model predictions visually using Grad-CAM heatmaps

The system is trained on the **VinBigData Chest X-ray dataset** — 18,000 DICOM images annotated by professional radiologists with 14 thoracic pathology classes.

---

## 🏗 Architecture

```
Chest X-ray (DICOM)
        │
        ▼
┌─────────────────────────────────────────────┐
│           PREPROCESSING PIPELINE            │
│  DICOM → PNG │ Resize 224×224 │ CLAHE │     │
│  Normalize │ Denoise │ Augmentations        │
└─────────────────────┬───────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐   ┌──────────────────────┐
│  CLASSIFICATION  │   │     DETECTION        │
│                  │   │                      │
│  EfficientNet-B0 │   │     YOLOv8s          │
│  (Multi-label)   │   │  (Bounding Boxes)    │
│                  │   │                      │
│  14 classes      │   │  14 classes          │
│  Sigmoid output  │   │  mAP50: 0.176        │
│  AUC: 0.905      │   │                      │
└────────┬─────────┘   └──────────┬───────────┘
         │                        │
         ▼                        ▼
┌──────────────────────────────────────────────┐
│           VISUALIZATION & INTERPRETATION     │
│   Grad-CAM Heatmaps │ Bounding Box Overlays  │
│   Precision-Recall Curves │ Error Analysis   │
└──────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch 2.x, torchvision, Ultralytics YOLOv8 |
| **Models** | EfficientNet-B0, ResNet50, YOLOv8s |
| **Data Processing** | pydicom, OpenCV, Pillow, NumPy, Pandas |
| **Augmentation** | Albumentations 2.x |
| **Evaluation** | scikit-learn (AUC, F1), torchmetrics |
| **Visualization** | Grad-CAM, matplotlib, seaborn |
| **Deployment (Optional)** | Streamlit, Gradio |
| **Environment** | Python 3.10+, Anaconda, Jupyter Notebook |

---

## 📊 Dataset

**VinBigData Chest X-ray Abnormalities Detection**
- Source: [Kaggle Competition](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
- Full dataset: 18,000 DICOM chest X-ray images
- Project subset: **5,000 images** (random seed=42)
- Annotation: CSV bounding boxes — 3 radiologists per image
- Format: DICOM → PNG (224×224) with YOLO label conversion

### Pathology Classes (14 + Normal)

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Aortic enlargement | 7 | Lung Opacity |
| 1 | Atelectasis | 8 | Nodule/Mass |
| 2 | Calcification | 9 | Other lesion |
| 3 | Cardiomegaly | 10 | Pleural effusion |
| 4 | Consolidation | 11 | Pleural thickening |
| 5 | ILD | 12 | Pneumothorax |
| 6 | Infiltration | 13 | Pulmonary fibrosis |
| 14 | No finding (Normal) | | |

---

## 📁 Project Structure

```
B13-CliniScan/
│
├── data/
│   ├── raw/
│   │   ├── train.csv                   # Full VinBigData annotations
│   │   ├── my_5000_labels.csv          # Filtered 5,000 image labels
│   │   ├── multilabel_matrix.csv       # 5000×14 binary label matrix
│   │   ├── train_split.csv             # 4,000 training image IDs
│   │   ├── val_split.csv               # 500 validation image IDs
│   │   └── test_split.csv              # 500 test image IDs
│   │
│   ├── processed/
│   │   ├── labels_yolo/                # 5,000 YOLO format .txt files
│   │   ├── sample_images.png           # EDA visualization
│   │   ├── class_distribution.png      # Class imbalance chart
│   │   └── training_curves.png         # M2 baseline training curves
│   │
│   ├── yolo_dataset/
│   │   ├── train/images/               # 4,001 training images
│   │   ├── train/labels/               # YOLO labels — train
│   │   ├── val/images/                 # 1,001 validation images
│   │   ├── val/labels/                 # YOLO labels — val
│   │   └── dataset.yaml                # YOLOv8 config (14 classes)
│   │
│   ├── detection_results/              # M2 detection outputs
│   │   ├── map_results_table.csv
│   │   ├── map50_per_class.png
│   │   └── precision_recall_plot.png
│   │
│   └── milestone3_results/             # M3 optimization outputs
│       ├── experiment_comparison.png
│       ├── detection_comparison.csv
│       ├── gradcam/                    # Grad-CAM heatmap images
│       ├── detection_viz/              # Sample detection visuals
│       └── error_analysis/             # FP/FN analysis
│
├── src/
│   ├── milestone2_classification.ipynb # M2 — ResNet50 baseline training
│   ├── milestone2_detection.ipynb      # M2 — YOLOv8s baseline training
│   ├── milestone3_classification.ipynb # M3 — EfficientNet experiments
│   ├── milestone3_detection.ipynb      # M3 — YOLOv8s optimization
│   ├── annotation.ipynb                # YOLO annotation conversion
│   ├── dicom_to_png.py                 # M1 — DICOM conversion script
│   ├── preprocessing.py                # M1 — image preprocessing
│   └── annotation_conversion.py        # M1 — annotation conversion
│
├── docs/
│   ├── Dataset_Description.pdf
│   ├── Milestone1_Report.pdf
│   ├── Milestone2_Report.pdf
│   └── Milestone3_Report.pdf
│
├── runs/                               # YOLOv8 training outputs (auto-generated)
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🚀 Milestones

### ✅ Milestone 1 — Data Preparation & Preprocessing
- Downloaded VinBigData dataset (5,000 image subset via Kaggle)
- DICOM → PNG conversion with MONOCHROME1 inversion handling
- Image preprocessing: resize 224×224, CLAHE, Gaussian blur, normalization
- Bounding box annotation conversion: CSV → YOLO format
- Per-image coordinate scaling from original DICOM resolution

### ✅ Milestone 2 — Model Development & Baseline Training
- Multi-label binary matrix construction (5,000 × 14)
- Train/Val/Test split: 80% / 10% / 10%
- **Classification:** ResNet50 pretrained baseline — Test AUC: 0.884
- **Detection:** YOLOv8s baseline — mAP50: 0.162
- Training curves, per-class evaluation, detection visualizations

### ✅ Milestone 3 — Model Optimization & Refinement
- 3 classification experiments: ResNet50+Adam, ResNet50+AdamW, EfficientNet-B0+AdamW
- Advanced Albumentations augmentation pipeline (medically safe)
- **Best classifier:** EfficientNet-B0 + AdamW — Test AUC: **0.905**
- **Best detection:** YOLOv8s + AdamW + 20 epochs — mAP50: **0.176**
- Grad-CAM heatmap generation for model interpretability
- Error analysis — FP/FN per class with miss rate calculation

### 🔄 Milestone 4 — Final Evaluation & Deployment (Upcoming)
- Class-weighted loss for rare pathology improvement
- Web interface prototype (Streamlit/Gradio)
- Final comprehensive evaluation and documentation

---

## 📈 Results

### Classification — EfficientNet-B0 (Best Model)

| Metric | M2 Baseline | M3 Best | Change |
|--------|-------------|---------|--------|
| Test AUC | 0.8844 | **0.9046** | +0.020 ✅ |
| Test Loss | 0.1803 | **0.1585** | -0.022 ✅ |
| Test F1 (macro) | 0.2967 | 0.2429 | Class imbalance |

**Top Performing Classes:**

| Class | F1 Score | Class | F1 Score |
|-------|----------|-------|----------|
| Aortic enlargement | **0.847** | Pleural thickening | 0.479 |
| Cardiomegaly | **0.778** | Pleural effusion | 0.375 |
| Lung Opacity | 0.457 | Pulmonary fibrosis | 0.333 |

### Detection — YOLOv8s (Best Model)

| Metric | M2 Baseline | M3 Best | Change |
|--------|-------------|---------|--------|
| mAP50 | 0.162 | **0.176** | +8.6% ✅ |
| mAP50-95 | 0.074 | **0.079** | +6.6% ✅ |
| Recall | 0.184 | **0.196** | +6.5% ✅ |

---

## ⚙ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/GKSJ-AI-CliniScan/B13-CliniScan.git
cd B13-CliniScan
```

### 2. Create Environment
```bash
conda create -n cliniscan python=3.10 -y
conda activate cliniscan
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Fix OMP Issue (Windows)
Add this as the **first line** in every notebook before imports:
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

### 5. Set Working Directory
```python
import os
os.chdir(r"path/to/B13-CliniScan")
```

---

## 💻 Usage

### Classification Training
Open `src/milestone3_classification.ipynb` and run all cells.
The notebook covers:
- Dataset loading with Albumentations augmentation
- EfficientNet-B0 / ResNet50 training
- Validation tracking (AUC, F1, Loss)
- Test set evaluation and per-class F1

### Detection Training
Open `src/milestone3_detection.ipynb` and run all cells.
The notebook covers:
- YOLOv8s training with AdamW optimizer
- mAP50 / mAP50-95 evaluation
- Bounding box visualization generation

### Grad-CAM Visualization
Grad-CAM heatmaps are generated inside `milestone3_classification.ipynb`.
Output saved to: `data/milestone3_results/gradcam/`

---

## 🖼 Outputs & Visualizations

| Output | Location | Description |
|--------|----------|-------------|
| `training_curves.png` | `data/processed/` | Loss + F1 curves |
| `experiment_comparison.png` | `data/milestone3_results/` | M3 experiment bar charts |
| `gradcam_grid.png` | `data/milestone3_results/gradcam/` | 10-image Grad-CAM grid |
| `map50_per_class.png` | `data/detection_results/` | Detection mAP per class |
| `error_analysis_plot.png` | `data/milestone3_results/error_analysis/` | FP/FN charts |
| Detection visuals (×20) | `data/milestone3_results/detection_viz/` | Predicted bounding boxes |

---

## 📄 Requirements

See `requirements.txt` for the full list. Key packages:

```
torch torchvision
ultralytics
albumentations
pydicom
opencv-python
scikit-learn
pandas numpy matplotlib seaborn
Pillow tqdm pyyaml
streamlit gradio
```

---

## 📚 References

- VinBigData Dataset: [physionet.org](https://physionet.org/content/vindr-cxr/1.0.0/)
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- Grad-CAM: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)

---

<div align="center">

**CliniScan** — Built by LUCKY SONI

</div>
