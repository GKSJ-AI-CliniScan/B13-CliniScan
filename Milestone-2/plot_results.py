import os
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import matplotlib.pyplot as plt
from PIL import Image

print("\n----- YOLOv8 Training Analysis -----\n")

# Paths
train_folder = "runs/detect/train"
results_csv = os.path.join(train_folder, "results.csv")
weights_path = os.path.join(train_folder, "weights/best.pt")

# 1️⃣ Generate Training Graph
print("Generating training graph...")
plot_results(results_csv)

# 2️⃣ Display metrics from results.csv
print("\nFinal Model Metrics:")
df = pd.read_csv(results_csv)
last_epoch = df.iloc[-1]

print(f"Precision: {last_epoch['metrics/precision(B)']:.3f}")
print(f"Recall: {last_epoch['metrics/recall(B)']:.3f}")
print(f"mAP50: {last_epoch['metrics/mAP50(B)']:.3f}")
print(f"mAP50-95: {last_epoch['metrics/mAP50-95(B)']:.3f}")

# 3️⃣ Show important training images
files_to_show = [
    "labels.jpg",
    "train_batch0.jpg",
    "train_batch1.jpg",
    "train_batch2.jpg",
    "results.png",
    "confusion_matrix.png"
]

print("\nOpening visualization files...")

for file in files_to_show:
    file_path = os.path.join(train_folder, file)
    if os.path.exists(file_path):
        img = Image.open(file_path)
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(file)
        plt.show()
    else:
        print(f"{file} not found.")

# 4️⃣ Run prediction using best model
print("\nRunning prediction using best.pt ...")

model = YOLO(weights_path)

model.predict(
    source="dataset/images/val",
    conf=0.25,
    save=True
)

print("\nPrediction images saved to:")
print("runs/detect/predict/")

print("\n----- Analysis Complete -----")