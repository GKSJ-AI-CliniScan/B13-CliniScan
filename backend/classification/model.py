import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CliniScanClassifier(nn.Module):
    def __init__(self, num_classes=15, freeze_features=True):
        """
        Original ResNet50 implementation for AI-CliniScan abnormality classification.
        """
        super(CliniScanClassifier, self).__init__()
        # Load pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        if freeze_features:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Unfreeze layer4 for fine-tuning
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
            
        in_features = self.backbone.fc.in_features
        # Replace the fully connected layer for multi-label classification
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """Used for Grad-CAM Visualization"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)
        return features
