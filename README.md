🩺 CliniScan – Chest X-ray Abnormality Detection System

CliniScan is a deep learning-based web application designed to assist in the analysis of chest X-ray images. It integrates classification, object detection, and explainability into a unified pipeline, providing both predictions and visual insights.

🚀 Live Demo

🔗 Deployed Application:
👉 https://b13-cliniscan-adithya-jayaram-ppgjlzg9mpndfqyschjatz.streamlit.app/

🧠 Project Overview

CliniScan follows a two-stage pipeline:

Classification (ResNet18)
Classifies X-ray as Normal or Abnormal
Detection (YOLOv8)
Localizes abnormalities using bounding boxes
Explainability (Grad-CAM)
Highlights regions influencing model decisions

🏗️ System Architecture
Input X-ray Image
        ↓
ResNet18 Classifier
        ↓
If Abnormal → YOLOv8 Detector
        ↓
Grad-CAM Visualization
        ↓
Streamlit Interface (UI)

📊 Dataset
Dataset: VinDr-CXR
Total Images Used: ~15,000
Train/Test Split: ~13,200 / 1,500

📈 Performance
🔹 Classification
Accuracy: ~70%
ROC-AUC: ~0.96
🔹 Detection
Precision: ~0.55
Recall: ~0.37
mAP@50: ~0.41

✨ Features
Upload chest X-ray images
Binary classification (Normal / Abnormal)
Abnormality detection with bounding boxes
Grad-CAM visualization for explainability
Confidence score display
PDF report generation
Usage history tracking
Web-based deployment

🖥️ Tech Stack
Frontend: Streamlit
Backend: Python, PyTorch
Models: ResNet18, YOLOv8
Libraries: OpenCV, Ultralytics, Albumentations

📂 Project Structure
App_Deploy/
│── app.py
│── requirements.txt
│── utils/
│     ├── inference.py
│     ├── gradcam.py
│     ├── report.py
│── Assets/
│── history.json

⚙️ Setup Instructions (Local)
git clone https://github.com/GKSJ-AI-CliniScan/B13-CliniScan/edit/Adithya-Jayaram
cd App_Deploy

pip install -r requirements.txt
streamlit run app.py
📦 Model Handling

Due to file size constraints, trained models are not stored in the repository.

They are downloaded dynamically at runtime using Google Drive links.

⚠️ Limitations
Moderate detection performance (low recall)
Limited dataset size
Binary classification only
Not clinically validated

🔮 Future Work
Multi-class classification
Improved detection accuracy
Larger and more diverse datasets
Clinical validation
Integration into healthcare systems

👨‍💻 Author: Adithya Jayaram
MSc Artificial Intelligence
MG University

📄 License

This project is developed for academic purposes.
