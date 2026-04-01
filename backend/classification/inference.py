import torch
import cv2
import numpy as np
import os
import sys
# Add current directory to path
sys.path.append(os.getcwd())
from classification.model import CliniScanClassifier
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_inference_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def predict(image_path, model_path, device):
    # 1. Load Model
    num_classes = 15
    model = CliniScanClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load and Preprocess Image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = get_inference_transforms()
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # 4. Class Names (Standard VinDr-CXR)
    class_names = [
        "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", 
        "Consolidation", "ILD", "Infiltration", "Lung Opacity", 
        "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening", 
        "Pneumothorax", "Pulmonary fibrosis", "No finding"
    ]

    # 5. Get Top Predictions
    results = []
    for i, prob in enumerate(probs):
        if prob > 0.2:  # 20% Threshold for medical findings
            results.append((class_names[i], prob))
    
    # Sort by confidence
    results.sort(key=lambda x: x[1], reverse=True)
    return results

if __name__ == '__main__':
    # Updated path to the models/ folder
    MODEL_PATH = 'models/best_resnet_classification.pth'
    os.makedirs('results', exist_ok=True)
    
    # Pick 3 samples from data/images
    img_list = [f for f in os.listdir('data/images') if f.endswith('.png')]
    if not img_list:
        print("No images found in data/images!")
    else:
        import random
        random.seed(42)
        samples = random.sample(img_list, min(3, len(img_list)))
        
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"--- CliniScan AI Proof-of-Work Generator ---")
        
        for i, img_name in enumerate(samples):
            image_path = os.path.join('data/images', img_name)
            predictions = predict(image_path, MODEL_PATH, device)
            
            # Load original image for drawing
            orig_img = cv2.imread(image_path)
            
            # Drawing overlay
            y_offset = 30
            cv2.putText(orig_img, "AI DIAGNOSIS", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            if not predictions:
                cv2.putText(orig_img, "No abnormalities detected", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                for name, score in predictions[:3]: # Top 3
                    text = f"{name}: {score:.1%}"
                    cv2.putText(orig_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_offset += 25
            
            save_path = f"results/evidence_{i+1}.png"
            cv2.imwrite(save_path, orig_img)
            print(f"Generated Evidence: {save_path}")

