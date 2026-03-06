
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
    backbone                  : "resnet50"  → best accuracy  (use with GPU)
                                "mobilenet" → 3× faster      (use on CPU)
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