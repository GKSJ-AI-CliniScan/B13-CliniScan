import cv2
import os
import numpy as np

INPUT_DIR = "data/processed/png_images"
OUTPUT_DIR = "data/processed/preprocessed_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize
    image = cv2.resize(image, (512, 512))

    # CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)

    # Denoising
    image = cv2.GaussianBlur(image, (3,3), 0)

    # Normalize (Z-score)
    image = image.astype(np.float32)
    image = (image - np.mean(image)) / (np.std(image) + 1e-8)

    return image

for file in os.listdir(INPUT_DIR):
    if file.endswith(".png"):
        img_path = os.path.join(INPUT_DIR, file)
        processed_img = preprocess_image(img_path)

        save_path = os.path.join(OUTPUT_DIR, file)
        cv2.imwrite(save_path, processed_img * 255)

print("Preprocessing completed.")
