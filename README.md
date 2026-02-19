# CliniScan – AI-based Chest X-ray Abnormality Detection

## Project Overview
This project builds a YOLO-based object detection model for detecting abnormalities in chest X-ray images using the VinBigData dataset.

## Dataset
- 15,000 chest X-ray images
- 67,914 annotations
- 14 abnormality classes

## Preprocessing Pipeline
- DICOM → PNG conversion
- Grayscale enforcement
- Min-max normalization
- CLAHE contrast enhancement
- Gaussian denoising
- Resize to 640x640
- Bounding box scaling
- YOLO label conversion
- Train/validation split (80/20)

## Model
YOLOv8 detection model.

## Status
Preprocessing complete.
Training phase in progress.
