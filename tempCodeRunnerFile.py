from ultralytics import YOLO

print("Status: 1/3 - Loading YOLOv8 model...")
# Load a pre-trained YOLOv8 model (the 'nano' version is fastest for Intel CPUs)
model = YOLO('yolov8n.pt') 

print("Status: 2/3 - Starting YOLO Training for 20 epochs...")
# Train the model on your 2,000 images
results = model.train(
    data='vinbigdata_yolo_preprocessed/data.yaml', # Points to your dataset
    epochs=20,                                     # Mandatory minimum
    imgsz=512,                                     # Matches your classification size
    batch=16,
    device='cpu',                                  # Forces it to use your Intel CPU
    project='runs/detect',
    name='CliniScan_Detection'
)

print(" Status: 3/3 - YOLO Training Complete!")
print("Check the 'runs/detect/CliniScan_Detection' folder for your final images with boxes.")