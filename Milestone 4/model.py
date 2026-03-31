import torch.nn as nn
import torchvision.models as models

def get_model(num_classes):

    model = models.efficientnet_b0(weights="DEFAULT")

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last TWO blocks
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model