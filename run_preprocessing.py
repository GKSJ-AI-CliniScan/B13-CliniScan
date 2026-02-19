import os
import pandas as pd
from preprocessing_pipeline import preprocess_image

# Paths
IMAGE_FOLDER = "CliniScan/dataset/png_images"
CSV_PATH = "CliniScan/dataset/filtered_annotations.csv"

# Load filtered annotations
df = pd.read_csv(CSV_PATH)

files = os.listdir(IMAGE_FOLDER)

for file in files[:5]:   # test only first 5 images
    image_path = os.path.join(IMAGE_FOLDER, file)

    img, boxes = preprocess_image(image_path, df)

    print(f"Processed: {file}")
    print("Bounding boxes:", boxes)
