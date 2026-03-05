from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="D:/Module_2/dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu"
)
