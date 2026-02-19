import os
import cv2
import numpy as np
import pydicom

INPUT_FOLDER = "CliniScan/dataset/train"
OUTPUT_FOLDER = "CliniScan/dataset/png_images"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".dicom"):
        path = os.path.join(INPUT_FOLDER, file)

        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array.astype(np.float32)

        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        image = (image * 255).astype(np.uint8)

        save_path = os.path.join(
            OUTPUT_FOLDER,
            file.replace(".dicom", ".png")
        )

        cv2.imwrite(save_path, image)

print("Conversion Complete")
