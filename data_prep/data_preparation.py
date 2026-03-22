import os
import cv2
import pydicom
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def normalize_dicom(pixel_array):
    """Normalize pixel array to 0-255 uint8 format."""
    image = pixel_array.astype(np.float32)
    image -= np.min(image)
    max_val = np.max(image)
    if max_val != 0:
        image /= max_val
    image *= 255.0
    return image.astype(np.uint8)

def convert_dicom_to_png(input_dir, output_dir):
    """Iterate through DICOMs and convert them to PNG."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dicom_files = list(input_path.glob('*.dicom')) + list(input_path.glob('*.dcm'))
    print(f"Found {len(dicom_files)} DICOM files to convert.")
    
    for dcm_file in tqdm(dicom_files, desc="Converting DICOMs"):
        try:
            ds = pydicom.dcmread(dcm_file)
            img = normalize_dicom(ds.pixel_array)
            png_name = f"{dcm_file.stem}.png"
            cv2.imwrite(str(output_path / png_name), img)
        except Exception as e:
            print(f"Error processing {dcm_file}: {e}")

def prepare_yolo_annotations(csv_path, output_dir, img_width=1024, img_height=1024):
    """Convert CSV annotations to YOLO standard txt files."""
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_id, group in tqdm(df.groupby('image_id'), desc="Parsing Annotations"):
        label_file = output_path / f"{img_id}.txt"
        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                # Skip 'No finding' standard rows if class_id indicates no bounding box
                if pd.isna(row['x_min']):
                    continue
                    
                # Calculate YOLO formatted outputs
                x_center = ((row['x_min'] + row['x_max']) / 2) / img_width
                y_center = ((row['y_min'] + row['y_max']) / 2) / img_height
                width = (row['x_max'] - row['x_min']) / img_width
                height = (row['y_max'] - row['y_min']) / img_height
                
                f.write(f"{row['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == '__main__':
    # Placeholders for actual paths
    RAW_DATA_DIR = './data/raw'
    PROCESSED_IMG_DIR = './data/images'
    CSV_ANNOTATION = './data/train.csv'
    YOLO_LABELS_DIR = './data/labels'
    
    # Run conversion if data is placed
    if os.path.exists(RAW_DATA_DIR):
        convert_dicom_to_png(RAW_DATA_DIR, PROCESSED_IMG_DIR)
        
    if os.path.exists(CSV_ANNOTATION):
        prepare_yolo_annotations(CSV_ANNOTATION, YOLO_LABELS_DIR)
