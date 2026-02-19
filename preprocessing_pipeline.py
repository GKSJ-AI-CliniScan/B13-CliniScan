import os
import cv2
import numpy as np
import pandas as pd

# ------------------------------
# BASIC IMAGE PREPROCESSING
# ------------------------------

def resize_image(img, size=(512, 512)):
    """Resize image to fixed dimensions"""
    return cv2.resize(img, size)

def normalize_image(img):
    """Normalize pixel values to range [0,1]"""
    return img.astype(np.float32) / 255.0

def apply_clahe(img):
    """Light contrast enhancement using CLAHE"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def denoise_image(img):
    """Apply light Gaussian blur for noise reduction"""
    return cv2.GaussianBlur(img, (5,5), 0)

# ------------------------------
# BOUNDING BOX PROCESSING
# ------------------------------

def scale_bboxes(bboxes, original_size, new_size=(512,512)):
    """
    Scale bounding boxes after resizing image.
    bboxes: list of [xmin, ymin, xmax, ymax]
    """
    orig_h, orig_w = original_size
    new_w, new_h = new_size

    x_scale = new_w / orig_w
    y_scale = new_h / orig_h

    scaled_boxes = []

    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        xmin = int(xmin * x_scale)
        xmax = int(xmax * x_scale)
        ymin = int(ymin * y_scale)
        ymax = int(ymax * y_scale)

        scaled_boxes.append([xmin, ymin, xmax, ymax])

    return scaled_boxes

# ------------------------------
# CSV HANDLING
# ------------------------------

def load_annotations(csv_path):
    """
    Load VinBigData annotation CSV
    """
    df = pd.read_csv(csv_path)
    return df

def get_boxes_for_image(df, image_id):
    """
    Extract bounding boxes for single image
    """
    rows = df[df["image_id"] == image_id]

    boxes = []

    for _, row in rows.iterrows():
        if not np.isnan(row["x_min"]):
            boxes.append([
                row["x_min"],
                row["y_min"],
                row["x_max"],
                row["y_max"]
            ])

    return boxes

# ------------------------------
# MAIN PIPELINE
# ------------------------------

def preprocess_image(image_path, annotation_df):
    """
    Full preprocessing pipeline:
    resize → CLAHE → denoise → normalize
    """
    img = cv2.imread(image_path, 0)  # grayscale

    original_size = img.shape

    img = resize_image(img)
    img = apply_clahe(img)
    img = denoise_image(img)
    img = normalize_image(img)

    image_id = os.path.basename(image_path).replace(".png", "")

    boxes = get_boxes_for_image(annotation_df, image_id)
    boxes = scale_bboxes(boxes, original_size)

    return img, boxes
