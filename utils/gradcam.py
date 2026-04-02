import torch
import cv2
import numpy as np
from PIL import Image

# ---------------------------
# Grad-CAM Function
# ---------------------------
def generate_gradcam(model, image, target_layer):
    model.eval()

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    # Preprocess
    img = image.resize((260, 260))
    img = np.array(img) / 255.0

    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    input_tensor = torch.tensor(img, dtype=torch.float32)

    # Forward
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

    # Backward
    model.zero_grad()
    output[0, pred_class].backward()

    # Get gradients & activations
    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]

    # Global average pooling
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    cam = cv2.resize(cam, (image.size[0], image.size[1]))

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Superimpose
    original = np.array(image)
    superimposed = heatmap * 0.4 + original

    # Remove hooks
    handle_f.remove()
    handle_b.remove()

    return Image.fromarray(np.uint8(superimposed))