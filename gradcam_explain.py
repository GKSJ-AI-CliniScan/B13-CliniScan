import torch
import cv2
import numpy as np
import os
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# =====================================
# 1. DEVICE
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# =====================================
# 2. LOAD TRAINED MODEL
# =====================================

model = models.resnet50(weights=None)

# number of classes
num_classes = 6

model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("best_resnet50_final.pth", map_location=device))

model = model.to(device)
model.eval()

print("Model Loaded Successfully")

# =====================================
# 3. CLASS NAMES (EDIT IF NEEDED)
# =====================================

class_names = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Normal"
]

# =====================================
# 4. IMAGE TRANSFORMATION
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================
# 5. GRADCAM TARGET LAYER
# =====================================

target_layers = [model.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers)

# =====================================
# 6. INPUT IMAGE FOLDER
# =====================================

image_folder = "test_images"

# =====================================
# 7. OUTPUT FOLDER
# =====================================

output_folder = "gradcam_results"
os.makedirs(output_folder, exist_ok=True)

# =====================================
# 8. PROCESS EACH IMAGE
# =====================================

for image_name in os.listdir(image_folder):

    if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(image_folder, image_name)

    print("\nProcessing:", image_name)

    # ---------------------------------
    # Load image
    # ---------------------------------

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    # ---------------------------------
    # Prediction
    # ---------------------------------

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output).item()

    predicted_label = class_names[predicted_class]

    print("Predicted Class:", predicted_label)

    # ---------------------------------
    # GradCAM
    # ---------------------------------

    targets = [ClassifierOutputTarget(predicted_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # ---------------------------------
    # Visualization
    # ---------------------------------

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255

    visualization = show_cam_on_image(img, grayscale_cam[0], use_rgb=True)

    # ---------------------------------
    # Save Result
    # ---------------------------------

    save_path = os.path.join(output_folder, f"gradcam_{image_name}")

    cv2.imwrite(save_path, visualization)

    print("GradCAM Saved:", save_path)

print("\nAll GradCAM Results Generated Successfully!")