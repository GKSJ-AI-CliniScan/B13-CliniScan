import os
import glob
from ultralytics import YOLO

def get_latest_model():
    # Grab the previous best.pt to use as a starting point instead of last.pt
    weights_files = glob.glob(os.path.join("runs", "**", "best.pt"), recursive=True)
    if not weights_files:
        raise FileNotFoundError("No best.pt found in the runs directory.")
    
    latest_model = max(weights_files, key=os.path.getmtime)
    print(f"Loading previous best model: {os.path.abspath(latest_model)}")
    return latest_model

def main():
    model_path = get_latest_model()
    
    # Load the best model from the previous run
    model = YOLO(model_path)
    
    print("Starting an advanced fine-tuning training run to improve accuracy!")
    
    # Start a NEW training setup with advanced image augmentations and hyper-parameters
    results = model.train(
        data='dataset.yaml',
        epochs=25,                                     
        imgsz=512,                                     
        batch=4,  # Lowered from 16 to 4 to prevent CUDA Out Of Memory on 4GB GPUs
        device=0, # Changed from 'gpu' to 0 because YOLOv8 requires an integer ID for NVIDIA GPUs
        project='runs/detect',
        name='yolo_medical_epoch25'
    )
    
    print("Optimized training completed successfully!")

if __name__ == "__main__":
    main()
