#Run this first to prove your local images and labels are perfectly aligned.
import os
import cv2
import random
import matplotlib.pyplot as plt

# Paths based on your folder structure
IMAGE_DIR = 'vinbigdata_yolo_preprocessed/train/images'
LABEL_DIR = 'vinbigdata_yolo_preprocessed/train/labels'

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
samples = random.sample(image_files, 10) # Instructor asked for 10 images

plt.figure(figsize=(20, 10))
for i, img_name in enumerate(samples):
    img = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    h, w, _ = img.shape
    label_path = os.path.join(LABEL_DIR, img_name.replace('.png', '.txt'))
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    plt.subplot(2, 5, i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
plt.tight_layout()
plt.show()