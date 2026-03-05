from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/dataset.yaml",
    epochs=50,
    imgsz=512,
    batch=16,
    lr0=0.001,
    name="clinicscan_detection"
)