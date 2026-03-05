from ultralytics import YOLO

if __name__ == '__main__':
    print("Status: 1/3 - Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt') 

    print("Status: 2/3 - Starting YOLO Training for 20 epochs...")
    
    # FIX: We changed the path here to just look in the main folder
    results = model.train(
        data='data.yaml', 
        epochs=20,                                     
        imgsz=512,                                     
        batch=16,
        device='cpu',                                  
        project='runs/detect',
        name='CliniScan_Detection'
    )

    print("Status: 3/3 - YOLO Training Complete!")