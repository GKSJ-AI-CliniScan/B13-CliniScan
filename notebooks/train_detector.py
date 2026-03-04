from ultralytics import YOLO
import torch
import multiprocessing


def main():

    model = YOLO("yolov8n.pt")

    model.train(
        data="data/dataset.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=4
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()  
    main()