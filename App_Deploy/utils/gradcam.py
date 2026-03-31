import torch
import numpy as np
import cv2
from torchvision import models
import torch.nn as nn

def generate_gradcam(model, image_tensor):
    model.eval()

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    image_tensor = image_tensor.to(next(model.parameters()).device)
    output = model(image_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return heatmap

def overlay_gradcam(original_image, heatmap):
    original = np.array(original_image)
    original = cv2.resize(original, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return overlay