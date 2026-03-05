from ultralytics import YOLO
import torch

def main():
    model = YOLO("yolov8n.pt")  # lighter model

    model.train(
        data="data.yaml",
        epochs=40,            # slightly reduced
        imgsz=512,            # reduced size
        batch=2,              # low memory
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0,            # Windows safe
        amp=False,
        cache=False,
        patience=15
    )

if __name__ == "__main__":
    main()