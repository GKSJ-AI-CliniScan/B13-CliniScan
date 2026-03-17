# CliniScan – Chest X-ray Abnormality Detection

## Overview
Two-stage deep learning system:
1. ResNet18 classifier (Normal vs Abnormal)
2. YOLOv8 detector (localization)

## Dataset
VinDr-CXR dataset

## Experiments (Milestone-3)
- Learning rate tuning
- Data augmentation (Albumentations)
- Transfer learning (DenseNet121)
- YOLO threshold tuning
- Grad-CAM visualization

## Results

### Classifier
| Model | Accuracy |
|------|------|
ResNet18 | 0.95 |
Augmented | 0.43 |
DenseNet121 | 0.63 |

### Detector
| Metric | Value |
|------|------|
Precision | 0.55 |
Recall | 0.37 |
mAP50 | 0.41 |

## Outputs
- Detection examples in `/results`
- Grad-CAM heatmaps
