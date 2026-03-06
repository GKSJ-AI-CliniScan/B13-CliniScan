from pathlib import Path
from collections import Counter

LABELS_DIR = "dataset/labels/train"  

label_files = list(Path(LABELS_DIR).glob("*.txt"))

total_images = len(label_files)
images_with_objects = 0
total_boxes = 0
class_counts = Counter()

for label_file in label_files:
    with open(label_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) > 0:
        images_with_objects += 1

    total_boxes += len(lines)

    for line in lines:
        cls = int(line.split()[0])
        class_counts[cls] += 1

# ---- Results ----
print("\n===== DATASET STATS =====")
print("Total images           :", total_images)
print("Images with objects    :", images_with_objects)
print("Empty images           :", total_images - images_with_objects)
print("Total bounding boxes   :", total_boxes)

if total_images > 0:
    print("Avg boxes per image    :", round(total_boxes / total_images, 2))

print("\nClass distribution:")
for cls, count in class_counts.items():

    print(f"Class {cls}: {count}")
