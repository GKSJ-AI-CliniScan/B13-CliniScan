import os
import shutil
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==== PATHS ====
CSV_PATH = r"C:\CliniScan\data\train.csv"
IMAGE_ROOT = r"C:\CliniScan\data\images"
OUTPUT_ROOT = r"C:\CliniScan\classification_data"

# Clean previous dataset (IMPORTANT)
if os.path.exists(OUTPUT_ROOT):
    shutil.rmtree(OUTPUT_ROOT)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH)

print("Total rows in CSV:", len(df))
print("Unique images:", df["image_id"].nunique())

# Group classes per image
grouped = df.groupby("image_id")["class_name"].apply(list)

image_labels = {}

for img_id, classes in grouped.items():
    unique_classes = list(set(classes))

    src_train = os.path.join(IMAGE_ROOT, "train", img_id + ".png")
    src_val   = os.path.join(IMAGE_ROOT, "val", img_id + ".png")

    if not (os.path.exists(src_train) or os.path.exists(src_val)):
        continue

    # Binary labeling
    if len(unique_classes) == 1 and unique_classes[0] == "No finding":
        image_labels[img_id] = "Normal"
    else:
        image_labels[img_id] = "Abnormal"

print("Total labeled images:", len(image_labels))

all_ids = list(image_labels.keys())
all_labels = list(image_labels.values())

train_ids, val_ids = train_test_split(
    all_ids,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)

print("Train size:", len(train_ids))
print("Validation size:", len(val_ids))


def copy_images(ids, split):
    for img_id in tqdm(ids, desc=f"Copying {split} images"):
        label = image_labels[img_id]

        src_train = os.path.join(IMAGE_ROOT, "train", img_id + ".png")
        src_val   = os.path.join(IMAGE_ROOT, "val", img_id + ".png")

        if os.path.exists(src_train):
            src = src_train
        elif os.path.exists(src_val):
            src = src_val
        else:
            print("Missing image:", img_id)
            continue

        dst_dir = os.path.join(OUTPUT_ROOT, split, label)
        os.makedirs(dst_dir, exist_ok=True)

        shutil.copy(src, os.path.join(dst_dir, img_id + ".png"))


copy_images(train_ids, "train")
copy_images(val_ids, "val")

print("Classification dataset creation complete.")