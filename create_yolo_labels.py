import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# =============================
# PATHS
# =============================
CSV_PATH = "train_folder/train.csv"
IMAGE_FOLDER = r"D:\B13-CliniScan\CliniScan_dataset\images_raw"
OUTPUT_FOLDER = "dataset"

# =============================
# CREATE OUTPUT FOLDERS (VERY IMPORTANT)
# =============================
os.makedirs(os.path.join(OUTPUT_FOLDER, "images/train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "images/val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "labels/val"), exist_ok=True)

# =============================
# LOAD CSV
# =============================
df = pd.read_csv(CSV_PATH)

# Remove "No Finding" class (class_id == 14)
df = df[df["class_id"] != 14]

# Unique images
image_ids = df["image_id"].unique()

# Train / Validation split
train_ids, val_ids = train_test_split(
    image_ids, test_size=0.2, random_state=42
)

print("Train Images:", len(train_ids))
print("Val Images:", len(val_ids))

# =============================
# BBOX CONVERSION FUNCTION
# =============================
def convert_bbox(row, img_w, img_h):
    x_min = row["x_min"]
    y_min = row["y_min"]
    x_max = row["x_max"]
    y_max = row["y_max"]

    center_x = ((x_min + x_max) / 2) / img_w
    center_y = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    return f"{int(row['class_id'])} {center_x} {center_y} {width} {height}"

# =============================
# PROCESS IMAGES
# =============================
for image_id in tqdm(image_ids):

    img_path = os.path.join(IMAGE_FOLDER, image_id + ".jpg")

    # Skip if image not found
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    img_w, img_h = img.size

    rows = df[df["image_id"] == image_id]

    # Decide train or val
    if image_id in train_ids:
        img_out = os.path.join(OUTPUT_FOLDER, "images/train", image_id + ".jpg")
        label_out = os.path.join(OUTPUT_FOLDER, "labels/train", image_id + ".txt")
    else:
        img_out = os.path.join(OUTPUT_FOLDER, "images/val", image_id + ".jpg")
        label_out = os.path.join(OUTPUT_FOLDER, "labels/val", image_id + ".txt")

    # Copy image
    shutil.copy(img_path, img_out)

    # Write label file
    with open(label_out, "w") as f:
        for _, row in rows.iterrows():
            bbox = convert_bbox(row, img_w, img_h)
            f.write(bbox + "\n")

print("YOLO Labels Created Successfully!")