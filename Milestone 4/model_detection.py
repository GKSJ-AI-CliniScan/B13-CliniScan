"""
model_detection.py
══════════════════════════════════════════════════════════════
Primary model : Faster R-CNN  with ResNet-50 FPN v2 backbone
Alternative   : Faster R-CNN  with MobileNet V3 (CPU fallback)

Why ResNet-50 FPN?
──────────────────
• Feature Pyramid Network detects findings at multiple scales
  (small nodules AND large effusions in the same image)
• Pretrained on COCO → strong transfer to chest X-ray
• Replaces only the box-predictor head; backbone stays frozen
  for the first few layers

Classes  (15 pathologies + 1 background = 16 total)
──────────────────────────────────────────────────────
 0  background          8  Nodule/Mass
 1  Aortic enlargement  9  Other lesion
 2  Atelectasis        10  Pleural effusion
 3  Calcification      11  Pleural thickening
 4  Cardiomegaly       12  Pneumothorax
 5  Consolidation      13  Pulmonary fibrosis
 6  ILD                14  No finding (rare boxes)
 7  Infiltration       15  Lung Opacity
"""

import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_detection_model(
    num_classes: int,
    backbone: str = "resnet50",
    trainable_backbone_layers: int = 3,
):
    """
    Parameters
    ──────────
    num_classes               : foreground classes + 1 background (e.g. 16)
    backbone                  :"mobilenet" → 3× faster      (use on CPU)
    trainable_backbone_layers : 0 = fully frozen backbone
                                3 = unfreeze last 3 FPN layers (recommended)
                                5 = fully trainable backbone
    """
    backbone = backbone.lower()

    if backbone == "resnet50":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    elif backbone == "mobilenet":
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose 'resnet50' or 'mobilenet'."
        )

    # Replace only the classification + regression head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def model_summary(model) -> None:
    """Print trainable vs frozen parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  Model params → total: {total:,} | "
          f"trainable: {trainable:,} | frozen: {frozen:,}")