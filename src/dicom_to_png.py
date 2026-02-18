import os
import cv2
import pydicom
import numpy as np

RAW_DIR = "data/raw/train"
OUTPUT_DIR = "data/processed/png_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_dicom_to_png(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.float32)

    # Min-Max Normalization
    image -= np.min(image)
    image /= np.max(image)
    image *= 255.0
    image = image.astype(np.uint8)

    cv2.imwrite(output_path, image)

for file in os.listdir(RAW_DIR):
    if file.endswith(".dicom") or file.endswith(".dcm"):
        input_path = os.path.join(RAW_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file.replace(".dicom", ".png").replace(".dcm", ".png"))
        convert_dicom_to_png(input_path, output_path)

print("DICOM to PNG conversion completed.")
