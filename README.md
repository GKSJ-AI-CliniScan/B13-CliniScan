# [cite_start]AI-CliniScan: Lung-Abnormality Detection on Chest X-rays using AI [cite: 1]

**Author:** Tirumani Surya

## Overview
[cite_start]AI-CliniScan is an AI-powered diagnostic system designed to automatically detect and localize lung abnormalities from chest X-ray images[cite: 3]. [cite_start]This repository contains the Milestone 1 data engineering and preprocessing pipeline, which prepares the raw **VinDr-CXR dataset** [cite: 5, 39] [cite_start]for downstream deep learning models (specifically YOLOv8 and ResNet/EfficientNet)[cite: 14, 15]. 

[cite_start]The raw dataset consists of 18,000 high-resolution DICOM images and a single CSV file containing bounding box coordinates[cite: 9]. This pipeline standardizes the imagery and normalizes the annotations for object detection training.



## Repository Structure
* [cite_start]`1_dicom_to_png.py`: Extracts pixel arrays from DICOM files, handles MONOCHROME inversion, normalizes intensities, resizes to 512x512, and applies CLAHE contrast enhancement[cite: 12, 111, 112, 113].
* [cite_start]`2_data_preprocessing_yolo.py`: Converts absolute bounding box coordinates from the Kaggle CSV into mathematically normalized `.txt` files required for YOLO architecture[cite: 117, 130].
* `Dataset_Description.pdf`: Detailed overview of the VinDr-CXR clinical data.
* `Milestone_1_Report.pdf`: Theoretical breakdown of the data transformation logic.

## Environment & Execution Setup
Due to the massive size of the raw medical dataset (200GB+), this preprocessing pipeline is specifically optimized to run within a **Kaggle Notebook** environment using the Kaggle API.

### Prerequisites
1. Create a Kaggle Notebook.
2. Click **+ Add Data** and attach the `vinbigdata-chest-xray-abnormalities-detection` competition dataset to your environment.
3. Ensure your notebook has internet access enabled.

### Running the Pipeline

**Step 1: Image Transformation**
Run the first script to convert the raw `.dicom` files into enhanced `.png` images.
```bash
python 1_dicom_to_png.py
