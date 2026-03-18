from ultralytics import YOLO
import cv2
import os

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Folder containing X-ray images
image_folder = "test_images"

# Output folder
output_folder = "detection_results"
os.makedirs(output_folder, exist_ok=True)

# Run detection on each image
for image_name in os.listdir(image_folder):

    image_path = os.path.join(image_folder, image_name)

    results = model(image_path)

    # Save detection image
    save_path = os.path.join(output_folder, image_name)
    results[0].save(filename=save_path)

    print("Detection saved:", save_path)

print("Detection completed!")