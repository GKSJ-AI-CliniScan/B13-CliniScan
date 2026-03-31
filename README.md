# 🫁 CliniScan — Chest X-ray Lung Abnormality Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=pytorch&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-red?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-VinBigData-blueviolet)

**AI-powered chest X-ray analysis system that detects and localises 14 types of lung abnormalities using YOLO object detection.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Models](#-models)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Scripts Reference](#-scripts-reference)
- [Results](#-results)
- [Per-Class Performance](#-per-class-performance)
- [Report](#-report)
- [Troubleshooting](#-troubleshooting)

---

## 🔍 Overview

CliniScan is a deep-learning pipeline built on **Ultralytics YOLOv8** to automatically detect and localise radiological abnormalities in frontal chest X-ray images. The system is trained on the **VinBigData Chest X-ray Abnormality Detection** dataset and supports 14 pathology classes.

The pipeline covers the full ML lifecycle:

```
DICOM Images → Preprocessing → YOLO Dataset → Training → Evaluation → PDF Report
```

---

## ✨ Features

- 🏥 **14-class chest abnormality detection** using bounding box localisation
- ⚡ **Two model variants** — YOLOv8s (fast, lightweight) and YOLOv8x (high accuracy)
- 🔧 **Full preprocessing pipeline** — DICOM → PNG, CLAHE enhancement, label validation
- 📊 **Automated evaluation** — mAP, Precision, Recall, per-class breakdown
- 🖼️ **Visual comparison** — Ground truth vs. prediction side-by-side images
- 📈 **Training dashboard** — Loss curves, mAP graphs, F1/PR curves
- 📄 **PDF report generator** — Professional comparative model report

---

## 🤖 Models

### YOLOv8s — Small (Deployed)

| Property | Value |
|---|---|
| Parameters | 11.2 M |
| Model size | 22.6 MB |
| Input size | 512 × 512 |
| Training epochs | 25 + 17 (resumed) |
| **Test mAP@50** | **0.3254** |
| **Test Precision** | **0.5568** |
| **Test Recall** | **0.0979** |
| Inference speed | ~5 ms/img (GPU) |

### YOLOv8x — Extra-Large (Target)

| Property | Value |
|---|---|
| Parameters | 68.2 M |
| Model size | 136.9 MB |
| Input size | 512 × 512 |
| Architecture | Deeper C2f backbone, wider PANet |
| GFLOPs (512px) | 257.8 |
| Inference speed | ~12 ms/img (GPU) |
| Status | Pre-trained weights available — full fine-tuning recommended |

---

## 📁 Dataset

**Source:** [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)

| Split | Images | Annotations |
|---|---|---|
| Train | ~3,000 | ~18,500 |
| Validation | ~1,000 | ~6,000 |
| Test | 260 | 2,143 |
| **Total** | **~5,000** | **~26,000+** |

### 14 Abnormality Classes

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | Aortic enlargement | 7 | Lung Opacity |
| 1 | Atelectasis | 8 | Nodule/Mass |
| 2 | Calcification | 9 | Other lesion |
| 3 | Cardiomegaly | 10 | Pleural effusion |
| 4 | Consolidation | 11 | Pleural thickening |
| 5 | ILD | 12 | Pneumothorax |
| 6 | Infiltration | 13 | Pulmonary fibrosis |

---

## 📂 Project Structure

```
cliniscan/
│
├── 📄 README.md                        ← You are here
├── 📄 dataset.yaml                     ← YOLO dataset configuration
│
├── 🔧 PREPROCESSING
│   ├── preprocess_dicom.py             ← DICOM → PNG conversion + augmentation
│   └── fix_labels.py                   ← Fix & validate YOLO bounding box labels
│
├── 🏋️ TRAINING
│   ├── train_yolov8s.py                ← Train YOLOv8s from scratch (25 epochs)
│   ├── train_yolo.py                   ← Resume training from latest best.pt
│   └── remove_unused.py               ← Clean up unused label/image files
│
├── 🧪 TESTING & EVALUATION
│   ├── test_yolo.py                    ← Run inference, save annotated images
│   ├── test_yolov8x.py                 ← Test YOLOv8x model (visual + metrics)
│   ├── test_with_labels.py             ← Run inference + save prediction .txt files
│   ├── evaluate_test_set.py            ← Quantitative evaluation on test split
│   └── get_metrics.py                  ← Quick print of latest training metrics
│
├── 📊 ANALYSIS & VISUALISATION
│   ├── compare_results.py              ← Ground truth vs prediction comparison images
│   ├── compare_original_vs_pred.py     ← Alternative comparison visualiser
│   ├── plot_yolov8s_graphs.py          ← Training dashboard charts (dark theme)
│   └── plot_loss_graph.py              ← Loss curve plotter
│
├── 📄 REPORTING
│   └── generate_model_report.py        ← Generate PDF comparative model report
│
├── 📦 DATASET
│   ├── processed_5000/                 ← Processed dataset (YOLO format)
│   │   ├── train/  images/ + labels/
│   │   ├── val/    images/ + labels/
│   │   └── test/   images/ + labels/
│   ├── train.csv                       ← VinBigData annotations
│   └── images.csv                      ← Image metadata (dimensions etc.)
│
├── 🏆 WEIGHTS
│   ├── yolov8n.pt                      ← YOLOv8 Nano  (pretrained)
│   ├── yolov8s.pt                      ← YOLOv8 Small (pretrained)
│   ├── yolov8m.pt                      ← YOLOv8 Medium (pretrained)
│   └── yolov8x.pt                      ← YOLOv8 Extra-Large (pretrained)
│
└── runs/                               ← All training & test outputs
    └── detect/
        ├── yolov8s_training/           ← Run 1 weights, curves, confusion matrix
        ├── yolov8s_training2/          ← Run 2 (resumed) — best.pt lives here
        ├── yolov8x_test_predictions/   ← Annotated test images
        └── yolov8x_test_metrics/       ← PR curves, F1 curves, confusion matrix
```

---

## ⚙️ Installation

### 1. Clone / open the project

```bash
cd "C:\Users\Asus\OneDrive\Pictures\Desktop\cliniscan"
```

### 2. Install dependencies

```bash
pip install ultralytics opencv-python numpy pandas matplotlib reportlab pydicom albumentations tqdm
```

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 training + inference |
| `opencv-python` | Image I/O and drawing |
| `numpy` | Array operations |
| `pandas` | CSV / annotation handling |
| `matplotlib` | Training graphs |
| `reportlab` | PDF report generation |
| `pydicom` | DICOM file reading |
| `albumentations` | Image augmentation pipeline |
| `tqdm` | Progress bars |

### 3. Verify GPU

```python
import torch
print(torch.cuda.is_available())   # Should print True
print(torch.cuda.get_device_name(0))
```

---

## 🚀 Quick Start

### Option A — Test the trained model immediately

```bash
# Run inference on test images + quantitative evaluation
py test_yolov8x.py
```

Results saved to:
- `runs/detect/yolov8x_test_predictions/` — annotated images
- `runs/detect/yolov8x_test_metrics/` — mAP, PR curve, confusion matrix

---

### Option B — Full pipeline from scratch

**Step 1: Preprocess DICOM images**
```bash
# Edit DICOM_DIR and ANNOTATION_CSV paths in the script first
py preprocess_dicom.py
```

**Step 2: Fix any corrupted labels**
```bash
py fix_labels.py
```

**Step 3: Train YOLOv8s**
```bash
py train_yolov8s.py
```

**Step 4: Resume training (optional)**
```bash
py train_yolo.py
```

**Step 5: Evaluate on test set**
```bash
py evaluate_test_set.py
```

**Step 6: Generate comparison images**
```bash
py compare_results.py
```

**Step 7: Plot training graphs**
```bash
py plot_yolov8s_graphs.py
```

**Step 8: Generate PDF report**
```bash
py generate_model_report.py
```

---

## 📜 Scripts Reference

### 🔧 Preprocessing

#### `preprocess_dicom.py`
Converts raw DICOM chest X-rays into YOLO-format PNG images.

- Reads `.dcm` files using `pydicom`
- Applies MONOCHROME1 inversion correction
- MinMax intensity normalisation → uint8
- Border cropping, Gaussian + Median denoising
- CLAHE contrast enhancement
- Train/Val/Test split (70/20/10)
- Albumentations augmentation (flip, shift-scale-rotate, brightness, elastic)
- Outputs YOLO labels: `class cx cy w h`

```bash
py preprocess_dicom.py
# Edit DICOM_DIR and ANNOTATION_CSV before running
```

#### `fix_labels.py`
Re-generates and validates YOLO label files from the original CSV annotations.

- Loads `train.csv` and `images.csv`
- Accounts for image scaling and letterbox padding to 512×512
- Clips all coordinates to `[0.0, 1.0]`
- Writes corrected `.txt` files; creates empty files for background images

```bash
py fix_labels.py
```

---

### 🏋️ Training

#### `train_yolov8s.py`
Trains YOLOv8s from COCO pretrained weights.

```python
# Key config inside the script:
model = YOLO('yolov8s.pt')
model.train(data='dataset.yaml', epochs=25, imgsz=512, batch=2, device=0)
```

```bash
py train_yolov8s.py
```

#### `train_yolo.py`
Resumes training from the latest `best.pt` checkpoint with advanced settings.

```python
# Auto-picks newest best.pt, trains for 25 more epochs
model.train(epochs=25, batch=4, imgsz=512, device=0, name='yolo_medical_epoch25')
```

```bash
py train_yolo.py
```

---

### 🧪 Testing & Evaluation

#### `test_yolov8x.py` ⭐ _Main test script_
Runs both visual predictions and quantitative evaluation on the test split.

```bash
py test_yolov8x.py
```

**Outputs:**
| Folder | Contents |
|---|---|
| `runs/detect/yolov8x_test_predictions/` | Annotated test images with bounding boxes |
| `runs/detect/yolov8x_test_metrics/` | mAP, PR curve, F1 curve, confusion matrix |

**Config (edit inside script):**
```python
SPECIFIC_MODEL = None    # None = auto-pick latest best.pt
CONF_THRESH    = 0.25   # Confidence threshold
IOU_THRESH     = 0.45   # NMS IoU threshold
IMAGE_SIZE     = 512
BATCH_SIZE     = 8
```

#### `evaluate_test_set.py`
Runs `model.val(split='test')` and prints mAP / Precision / Recall.

```bash
py evaluate_test_set.py
```

#### `test_yolo.py`
Runs inference only (no ground truth comparison). Saves annotated images.

```bash
py test_yolo.py
```

#### `test_with_labels.py`
Runs inference and saves both annotated images and prediction `.txt` label files.

```bash
py test_with_labels.py
# Labels saved alongside images for downstream comparison
```

#### `get_metrics.py`
Quick-print the last epoch metrics from the most recently modified `results.csv`.

```bash
py get_metrics.py
```

---

### 📊 Analysis & Visualisation

#### `compare_results.py`
Generates side-by-side **Ground Truth vs Prediction** comparison images.

```bash
py compare_results.py
# Output: runs/detect/comparison_results/
```

Each image shows:
- 🟩 **Left panel** — Ground truth boxes (green header)
- 🟦 **Right panel** — Model predictions (blue header)
- 14 distinct colours per class

#### `plot_yolov8s_graphs.py`
Creates a full dark-themed training dashboard with 3 figures.

```bash
py plot_yolov8s_graphs.py
# Output: runs/detect/yolov8s_graphs/
```

| Figure | Content |
|---|---|
| `yolov8s_loss_curves.png` | Box Loss, Class Loss, DFL Loss (train vs val) |
| `yolov8s_map_metrics.png` | Precision, Recall, mAP@50 / mAP@50-95 over epochs |
| `yolov8s_dashboard.png` | Combined 2×3 grid summary dashboard |

#### `plot_loss_graph.py`
Lightweight single-chart loss plotter.

```bash
py plot_loss_graph.py
```

---

### 📄 Reporting

#### `generate_model_report.py`
Generates a professional multi-page PDF comparing YOLOv8s and YOLOv8x.

```bash
py generate_model_report.py
# Output: CliniScan_Model_Report.pdf (~2.7 MB)
```

**Report sections:**
1. Executive Summary
2. Dataset Overview
3. Model Architectures (YOLOv8s vs YOLOv8x)
4. Training Configuration
5. Training Results (Run 1 & Run 2)
6. Test Set Evaluation + Per-class table
7. Visual Outputs (confusion matrix, PR/F1 curves)
8. Challenges & Recommendations
9. Conclusion

---

## 📊 Results

### Test Set — Overall Metrics

| Metric | Score |
|---|---|
| **mAP@0.50** | **0.3254** |
| **mAP@0.50–0.95** | **0.1846** |
| **Precision** | **0.5568** |
| **Recall** | **0.0979** |
| Test images | 260 |
| Total instances | 2,143 |

> Evaluated with `conf=0.25`, `iou=0.45`, `imgsz=512`

---

## 🔬 Per-Class Performance

| Class | mAP@50 | Precision | Recall | Notes |
|---|---|---|---|---|
| 🟢 Aortic enlargement | **0.647** | 0.862 | 0.389 | Best class — most training data |
| 🟢 Cardiomegaly | **0.633** | 0.849 | 0.378 | Strong performance |
| ILD | 0.513 | 1.000 | 0.026 | Perfect precision, very low recall |
| Nodule/Mass | 0.512 | 1.000 | 0.025 | Perfect precision, very low recall |
| Pulmonary fibrosis | 0.426 | 0.818 | 0.031 | — |
| Infiltration | 0.395 | 0.750 | 0.042 | — |
| Pleural thickening | 0.388 | 0.750 | 0.026 | — |
| Pleural effusion | 0.344 | 0.486 | 0.224 | Best recall after top-2 |
| Lung Opacity | 0.300 | 0.571 | 0.027 | — |
| Other lesion | 0.270 | 0.500 | 0.030 | — |
| Consolidation | 0.075 | 0.125 | 0.048 | Rare class |
| Atelectasis | 0.052 | 0.083 | 0.125 | Very few samples |
| 🔴 Calcification | **0.000** | 0.000 | 0.000 | Insufficient data |
| 🔴 Pneumothorax | **0.000** | 0.000 | 0.000 | Only 11 test instances |

---

## 📄 Report

A full **PDF comparative report** is available:

📁 `CliniScan_Model_Report.pdf`

Regenerate it anytime:
```bash
py generate_model_report.py
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `No best.pt found` | Train a model first with `py train_yolov8s.py` |
| `CUDA out of memory` | Reduce `batch` size in training script (try `batch=2`) |
| `ModuleNotFoundError: reportlab` | Run `pip install reportlab` |
| `Python not found` | Use `py` instead of `python` on Windows |
| Low recall (high precision) | Lower `CONF_THRESH` from 0.25 → 0.10 |
| Many duplicate boxes | Lower `IOU_THRESH` from 0.45 → 0.35, or use WBF post-processing |
| Corrupted labels error | Run `py fix_labels.py` to repair annotation files |
| Training stops early | Increase `patience` parameter in the training script |

---

## 📌 Configuration — `dataset.yaml`

```yaml
path: C:/Users/Asus/OneDrive/Pictures/Desktop/cliniscan/processed_5000
train: train/images
val:   val/images
test:  test/images

nc: 14
names: ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
        'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
        'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
        'Pulmonary fibrosis']
```

> ⚠️ Update `path` if you move the project to a different location.

---

## 🗺️ Roadmap

- [ ] Train a dedicated **YOLOv8x** model end-to-end on the full dataset
- [ ] Use the **full 18k VinBigData** dataset (currently 5k subset)
- [ ] Add **Weighted Box Fusion (WBF)** post-processing
- [ ] Implement **SAHI** (Slicing Aided Hyper Inference) for small lesions
- [ ] Add a **Streamlit/Gradio web demo** for radiologist testing
- [ ] Experiment with **YOLOv8-seg** for instance segmentation masks

---

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [VinBigData Chest X-ray Dataset — Kaggle](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
- [Albumentations Documentation](https://albumentations.ai)
- [pydicom Documentation](https://pydicom.github.io)

---

<div align="center">
<b>CliniScan</b> · Built with ❤️ using YOLOv8 + PyTorch · 2026
</div>
