import torch
import cv2
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from detection.model import get_detection_model

def predict_detection(image_path, model_path, device, threshold=0.5):
    # 1. Load Model (16 classes = 14 findings + 1 No Finding + 1 Background)
    num_classes = 16
    model = get_detection_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load and Preprocess Image (Match dataset 256x256)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Simple Resize for inference
    image_resized = cv2.resize(image_rgb, (256, 256))
    image_tensor = torch.as_tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # 4. Class Names (Shifted +1)
    class_names = [
        "Background", "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", 
        "Consolidation", "ILD", "Infiltration", "Lung Opacity", 
        "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening", 
        "Pneumothorax", "Pulmonary fibrosis", "No finding"
    ]

    # 5. Process Results
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    # Scale boxes back to original image size
    orig_h, orig_w = image.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * (orig_w / 256.0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * (orig_h / 256.0)

    # 6. Draw Bounding Boxes
    output_image = image.copy()
    count = 0
    for i in range(len(scores)):
        if scores[i] > threshold:
            count += 1
            box = boxes[i].astype(int)
            label_idx = labels[i]
            label_text = f"{class_names[label_idx]}: {scores[i]:.2f}"
            
            cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(output_image, label_text, (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return output_image, count

if __name__ == '__main__':
    MODEL_PATH = 'models/best_faster_rcnn_detection.pth'
    os.makedirs('results', exist_ok=True)
    
    # Pick 3 random samples from data/images
    img_list = [f for f in os.listdir('data/images') if f.endswith('.png')]
    if not img_list:
        print("No images found in data/images!")
    else:
        import random
        # Optional: set seed for reproducibility, or comment out for variety
        # random.seed(42) 
        samples = random.sample(img_list, min(3, len(img_list)))
        
        device = torch.device('cpu') # Detection is more stable on CPU for inference too
        
        print(f"--- CliniScan AI Detection Proof-of-Work Generator ---")
        
        for i, img_name in enumerate(samples):
            image_path = os.path.join('data/images', img_name)
            output_img, found = predict_detection(image_path, MODEL_PATH, device)
            
            save_path = f'results/detection_evidence_{i+1}.png'
            cv2.imwrite(save_path, output_img)
            print(f"Saved: {save_path} (Detected: {found})")

