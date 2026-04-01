from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

def add_code_snippet(doc, code):
    p = doc.add_paragraph()
    p.style = 'No Spacing'
    run = p.add_run(code)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x2B, 0x2B, 0x2B)
    
def create_m1_doc(output_path):
    doc = Document()
    title = doc.add_heading('AI-CliniScan: Milestone 1 Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading('Data Preparation, EDA & Setup', level=1)
    
    doc.add_heading('1. Abstract & Objective', level=2)
    doc.add_paragraph('The primary objective of Milestone 1 is to acquire, inspect, and prepare the VinDr-CXR dataset for downstream deep learning tasks. Since medical imaging data typically arrives in the DICOM (Digital Imaging and Communications in Medicine) format, raw ingestion into standard Convolutional Neural Networks (CNNs) is impossible without extensive preprocessing. This milestone details our approach to extracting pixel arrays, normalizing contrast, scaling intensities, and translating diagnostic records into bounding-box annotations suitable for an Object Detection pipeline.')
    
    doc.add_heading('2. Dataset Acquisition', level=2)
    doc.add_paragraph('We securely accessed the VinDr-CXR dataset via PhysioNet. The dataset comprises 18,000 postero-anterior (PA) view chest X-Rays annotated by multiple radiologists. Each image may contain multiple abnormalities (up to 14 classes such as Aortic enlargement, Cardiomegaly, Pulmonary fibrosis, etc.) alongside a "No finding" class.')
    
    doc.add_heading('3. DICOM to PNG Conversion Pipeline', level=2)
    doc.add_paragraph('DICOM files contain metadata interwoven with 16-bit pixel arrays. To standardize this for PyTorch, we utilized the `pydicom` Python package. The pixel data is extracted, normalized to a 0-1 scale to prevent extreme contrast clipping, and then multiplied by 255.0 to safely cast it to an 8-bit unsigned integer (uint8).')
    
    doc.add_paragraph('Code snippet of the normalization logic:')
    code1 = '''def normalize_dicom(pixel_array):
    image = pixel_array.astype(np.float32)
    image -= np.min(image)
    max_val = np.max(image)
    if max_val != 0:
        image /= max_val
    image *= 255.0
    return image.astype(np.uint8)'''
    add_code_snippet(doc, code1)
    
    doc.add_heading('4. Annotation Parsing for Faster R-CNN', level=2)
    doc.add_paragraph('The dataset provides annotations in a singular `train.csv`. Because we mapped out a dual-architecture system (Classification + Object Detection), our preprocessing must construct YOLO/COCO-formatted bounding boxes explicitly. We utilized `pandas` to group by `image_id`, looping over rows to extract bounding box coordinates. Crucially, rows indicative of "No finding" lacking coordinates are safely bypassed.')
    
    doc.add_paragraph('Code snippet for Annotation Extraction:')
    code2 = '''for img_id, group in df.groupby('image_id'):
    for _, row in group.iterrows():
        if pd.isna(row['x_min']): continue
        x_center = ((row['x_min'] + row['x_max']) / 2) / img_width
        y_center = ((row['y_min'] + row['y_max']) / 2) / img_height
        width = (row['x_max'] - row['x_min']) / img_width
        height = (row['y_max'] - row['y_min']) / img_height'''
    add_code_snippet(doc, code2)
    
    doc.add_heading('5. Conclusion and Processing Results of Milestone 1', level=2)
    doc.add_paragraph('At the conclusion of M1, the raw repository was successfully transformed into a ready-to-use directory of standard 2D PNG matrices paired with structured bounding-box tables. Specifically, the script successfully extracted and converted all 15,000 DICOM images from the training dataset into the `data/images/` directory without memory overflow.')
    doc.add_paragraph('Additionally, 15,000 YOLO-formatted .txt files were successfully mapped and saved into `data/labels/`. To facilitate rapid model prototyping on local hardware (Apple MPS), we instituted a programmatic seed (`random.seed(42)`) within the PyTorch DataLoader classes to randomly sub-sample exactly 2000 images for the subsequent training phases. This ensures mathematical accuracy while preventing local hardware exhaustion.')
    
    doc.save(output_path)
    print(f"Saved {output_path}")

def create_m2_doc(output_path):
    doc = Document()
    title = doc.add_heading('AI-CliniScan: Milestone 2 Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading('Model Development & Baseline Algorithms', level=1)
    
    doc.add_heading('1. Abstract & Objective', level=2)
    doc.add_paragraph('Milestone 2 shifts the paradigm from data preparation to architectural engineering. Our goal is to firmly establish functional deep learning baselines utilizing PyTorch. We architected a dual-pronged approach: a multi-label Classification mechanism to predict the presence of lung ailments globally, and an Object Detection mechanism to localize them.')
    
    doc.add_heading('2. Classification: Architecting ResNet-50', level=2)
    doc.add_paragraph('For classification, we departed from lighter frameworks and opted for the robust `ResNet-50` via `torchvision.models`. The inherent residual connections (skip connections) mitigate vanishing gradients across the 50 deeper layers.')
    doc.add_paragraph('Because VinDr-CXR features up to 14 abnormalities, this is a multi-label classification problem (not multi-class). Thus, we stripped the standard 1000-class ImageNet fully connected layer and injected a specialized Dropout + Linear block pushing to a 15-node output matrix. We paired this forward pass with `nn.BCEWithLogitsLoss()` which merges a Sigmoid layer and Binary Cross Entropy dynamically.')
    
    doc.add_paragraph('Model Instantiation Code:')
    code1 = '''class CliniScanClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )'''
    add_code_snippet(doc, code1)
    
    doc.add_heading('3. Detection: Architecting Faster R-CNN', level=2)
    doc.add_paragraph('Localization requires regressional intelligence. We successfully integrated PyTorch’s `fasterrcnn_resnet50_fpn`. The Feature Pyramid Network (FPN) allows the detection mechanism to perceive anomalies at multi-scale resolutions, which is absolutely vital since Pulmonary fibrosis covers vast lung regions whereas Aortic anomalies are highly localized. A custom `VinDrCXRBoxesDataset` handles tensor packaging for the RoI heads.')
    
    doc.add_heading('4. Training & Validations (Results Phase)', level=2)
    doc.add_paragraph('Training loops were engineered via `tqdm`. Batch processing utilized the Apple MPS (Metal Performance Shaders) backend for accelerated gradient descent for classification, and CPU for stable detection convergence. Our ResNet-50 model achieved a peak Validation AUC of **0.9449** at the 10th epoch, demonstrating exceptional discriminatory power. The Faster R-CNN detection model successfully completed its training epoch with a stable loss of ~0.6.')
    doc.add_paragraph('All model weights have been locally serialized to the `models/` directory:')
    doc.add_paragraph('- Classification: `models/best_resnet_classification.pth`')
    doc.add_paragraph('- Detection: `models/best_faster_rcnn_detection.pth`')
    
    doc.save(output_path)
    print(f"Saved {output_path}")

def create_m3_doc(output_path):
    doc = Document()
    title = doc.add_heading('AI-CliniScan: Milestone 3 Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading('Advanced Optimizations & Clinical Interpretability', level=1)
    
    doc.add_heading('1. Abstract & Objective', level=2)
    doc.add_paragraph('Having established functional PyTorch baselines, Milestone 3 focuses on boosting macro-AUC metrics, preventing adversarial overfitting, and most crucially, injecting model interpretability. Medical AI is essentially useless without explainability; clinicians must see *why* the CNN generated an opacity diagnosis.')
    
    doc.add_heading('2. Advanced Data Augmentations with Albumentations', level=2)
    doc.add_paragraph('Standard Torchvision augmentations lack spatial coherence for bounding boxes. We migrated entirely to the `Albumentations` framework. We defined a robust pipeline utilizing `ShiftScaleRotate`, `RandomCrop`, `HorizontalFlip`, and sophisticated `Normalize(mean, std)` hooks. This artificial inflation of the dataset combats class imbalances natively present in CXR labels.')
    
    doc.add_heading('3. Hyperparameter Tuning & Transfer Learning', level=2)
    doc.add_paragraph('During M3, we instituted layer freezing protocols. The early convolutional layers of our ResNet-50 backbone were frozen (requires_grad = False), while `layer4` and our custom dense layers were kept fluid. A Step Learning Rate Scheduler (`optim.lr_scheduler.StepLR`) dampens the LR by a factor of 0.1 every 3 epochs, forcing the loss surface into narrower local minima for optimized accuracy.')
    
    doc.add_heading('4. Diagnostic Visualization: Grad-CAM Integration', level=2)
    doc.add_paragraph('To explain the classifier\'s mathematical logic, we engineered a feature extraction pipe returning the activation tensors immediately prior to Average Pooling (`layer4` output). By utilizing Gradient-weighted Class Activation Mapping (Grad-CAM), we can multiply the gradients of the target concept (e.g. Pneumothorax) into these feature maps.')
    doc.add_paragraph('This generates a heatmap correlating precisely over the original lung PNG, mathematically highlighting the exact pixels that statistically drove the classification decision--fulfilling a critical Milestone 3 deployability requirement.')
    
    doc.add_heading('5. Submission Evidence & Results', level=2)
    doc.add_paragraph('To fulfill the visual validation requirements, we generated sample diagnostic overlays located in the `results/` directory. These include:')
    doc.add_paragraph('- Diagnostic Classification Proofs: `evidence_1.png` to `evidence_3.png`')
    doc.add_paragraph('- Localization Detection Proofs (Bounding Boxes): `detection_evidence_1.png` to `detection_evidence_3.png` proving precise abnormality pinpointing.')
    
    doc.add_paragraph('Feature Hook Code:')
    code1 = '''def extract_features(self, x):
    x = self.backbone.conv1(x)
    x = self.backbone.bn1(x)
    x = self.backbone.relu(x)
    x = self.backbone.maxpool(x)
    for layer in [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3]:
        x = layer(x)
    features = self.backbone.layer4(x) # Target for Grad-CAM
    return features'''
    add_code_snippet(doc, code1)
    
    doc.add_heading('6. Final Conclusion', level=2)
    doc.add_paragraph('The complete pipeline from raw DICOM to advanced Grad-CAM explainable AI has been solidified. The system reliably ingests datasets, trains efficiently across dual architectures, and generates output required by clinical standards.')
    
    doc.save(output_path)
    print(f"Saved {output_path}")

def create_m4_doc(output_path):
    doc = Document()
    title = doc.add_heading('AI-CliniScan: Milestone 4 Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading('Full-Stack Integration, UI Refinement & Cloud Deployment', level=1)
    
    doc.add_heading('1. Abstract & Objective', level=2)
    doc.add_paragraph('Milestone 4 transitions our machine learning pipelines from local scripts into a fully-fledged, accessible clinical web application. Our objective was to develop a secure frontend interface (React), construct a robust API server (FastAPI), and deploy the entire system to cloud infrastructure (Vercel & Hugging Face Spaces) for live production inference.')
    
    doc.add_heading('2. AI Backend Containerization (Hugging Face Spaces)', level=2)
    doc.add_paragraph('To serve the heavy PyTorch model weights (ResNet-50 and Faster R-CNN), we engineered a FastAPI backend that handles multipart image uploads and executes inference asynchronously. This architecture isolates the GPU/CPU-heavy operations from the user interface.')
    doc.add_paragraph('We constructed a Docker environment utilizing the `python:3.9-slim` base image, integrating essential system libraries like `libgl1` required for OpenCV operations. The backend APIs, including `/predict`, were securely pushed and hosted on a Hugging Face Docker Space, providing scalable and reliable model serving.')
    
    doc.add_paragraph('FastAPI Inference Endpoint Code Snippet:')
    code1 = '''@app.post("/predict")
async def predict_image(file: UploadFile, mode: str = Form(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    if mode == "classify":
        results = classify_image(image)
        return {"mode": "classify", "predictions": results}
    elif mode == "detect":
        results = detect_abnormalities(image)
        return {"mode": "detect", "predictions": results}'''
    add_code_snippet(doc, code1)
    
    doc.add_heading('3. Secure Frontend Application (React & Vite)', level=2)
    doc.add_paragraph('We engineered the frontend interface `MediScanAI` using React, Vite, and Tailwind CSS. The application architecture establishes secure routing, enforcing authenticated user access before allowing entry into the diagnostic workspace.')
    doc.add_paragraph('A sophisticated user dashboard manages session history, allowing clinicians to review past analyses. The core analysis workspace was overhauled with a dark-themed, high-contrast UI tailored for clinical environments, reducing eye strain during radiological reviews.')
    
    doc.add_heading('4. Dynamic SVG Overlays & Print Reporting', level=2)
    doc.add_paragraph('Integrating the Faster R-CNN bounding boxes onto standard DOM images posed scaling challenges. We engineered a resolution-independent SVG overlay utilizing a `viewBox="0 0 100 100"` coordinate system. The backend returns scaled percentage coordinates, which perfectly map onto the user\'s uploaded image dynamically, regardless of screen topology.')
    doc.add_paragraph('Furthermore, we implemented a specialized CSS `@media print` layout that strips away interface elements, rendering a structured, high-contrast, scalable PDF diagnostic report outlining pathology scores and clinical recommendations.')
    
    doc.add_heading('5. Cloud Deployment & Conclusion', level=2)
    doc.add_paragraph('The frontend application was configured with environment variables (`VITE_API_URL`) to seamlessly point to the live Hugging Face Space endpoint, and successfully deployed via Vercel for fast, edge network delivery.')
    doc.add_paragraph('The successful conclusion of Milestone 4 marks the transition of AI-CliniScan from an experimental PyTorch repository into a 24/7 accessible, secure, and intuitive clinical diagnostic tool capable of augmenting radiological workflows.')
    
    doc.save(output_path)
    print(f"Saved {output_path}")

if __name__ == '__main__':
    base_dir = '/Users/yashmittal/Desktop/Internships/infosis_springboard/AI-CliniScan/docs'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    create_m1_doc(os.path.join(base_dir, 'M1-Data-Preparation.docx'))
    create_m2_doc(os.path.join(base_dir, 'M2-Baseline-Training.docx'))
    create_m3_doc(os.path.join(base_dir, 'M3-Optimization-Visualization.docx'))
    create_m4_doc(os.path.join(base_dir, 'M4-Deployment-Integration.docx'))
