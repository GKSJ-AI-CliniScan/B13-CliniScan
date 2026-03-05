from ultralytics import YOLO

model = YOLO("C:/Users/admin/runs/detect/train3/weights/best.pt")

results = model.predict(
    source="test.png",
    conf=0.25,
    save=True
)

print("Detection completed!")
