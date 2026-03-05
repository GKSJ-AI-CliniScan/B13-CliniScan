from ultralytics import YOLO

model = YOLO("runs/detect/clinicscan_detection/weights/best.pt")

metrics = model.val(data="dataset/dataset.yaml")
print(metrics)