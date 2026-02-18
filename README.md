🩺 CliniScan
Chest X-ray Abnormality Detection Pipeline

CliniScan is a structured medical imaging project focused on preparing chest X-ray data for automated abnormality detection and localization.

The project uses the VinBigData Chest X-ray Abnormalities Detection dataset released by VinBigData and hosted on Kaggle.

🎯 Milestone 1 — Data Understanding & Preprocessing

Milestone 1 establishes a clean and standardized preprocessing pipeline to make raw medical images training-ready.

✔ Completed Tasks

DICOM → PNG conversion

Photometric inversion handling (MONOCHROME1 correction)

Image resizing (512 × 512)

Intensity normalization

CLAHE-based contrast enhancement

Gaussian denoising

Z-score standardization

Annotation conversion (CSV → YOLO format)

A subset of 50–60 training images was used to validate the pipeline during development.

**b13-cliniscan/**

b13-cliniscan/
│
├── 📁 data/
│ ├── 📁 raw/
│ │ └── 📁 train/ # Original DICOM images
│ ├── 📁 processed/
│ │ └── 📁 png-images/
│ └── 📄 train.csv
│
├── 📁 docs/
│ ├── 📄 Dataset_Description.pdf
│ └── 📄 Milestone1_Report.pdf
│
├── 📁 src/
│ ├── 🐍 dicom_to_png.py
│ ├── 🐍 preprocessing.py
│ └── 🐍 annotation_conversion.py
│
├── 📄 README.md
├── 📦 requirements.txt
└── 📜 LICENSE

The repository follows a modular structure separating raw data, processed outputs, source code, and documentation.

🔬 Preprocessing Overview

The pipeline is designed to:

Standardize image dimensions

Improve local contrast for subtle abnormalities

Reduce scanner-induced noise

Preserve anatomical structures

Align bounding boxes with resized images

Output images are detection-ready and compatible with YOLO-based object detection models.

⚙️ How to Run

Install dependencies:

pip install -r requirements.txt


Run pipeline:

python src/dicom_to_png.py
python src/preprocessing.py
python src/annotation_conversion.py

*Dataset not included in repository.
*Download from Kaggle and place inside data/raw/


📌 Outcome of Milestone 1

Cleaned and standardized dataset

YOLO-formatted annotations

Reproducible preprocessing pipeline

Detection-ready training dataset

CliniScan is now prepared for model training and evaluation in the next milestone.