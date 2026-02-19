"""
Chest X-ray Analysis System

A comprehensive medical imaging pipeline for chest X-ray preprocessing,
deep learning-based classification and detection, and visualization.
"""

__version__ = "0.1.0"

from chest_xray_analysis.preprocessing import (
    DICOMLoader,
    ImagePreprocessor,
    PreprocessingConfig,
)
from chest_xray_analysis.data import (
    AnnotationConverter,
    MedicalAugmentor,
    DatasetManager,
    BoundingBox,
    AugmentationConfig,
)
from chest_xray_analysis.models import (
    ChestXrayClassifier,
    ChestXrayDetector,
    TrainingConfig,
)
from chest_xray_analysis.evaluation import (
    MetricsCalculator,
    ClassificationMetrics,
    DetectionMetrics,
)
from chest_xray_analysis.visualization import Visualizer

__all__ = [
    "DICOMLoader",
    "ImagePreprocessor",
    "PreprocessingConfig",
    "AnnotationConverter",
    "MedicalAugmentor",
    "DatasetManager",
    "BoundingBox",
    "AugmentationConfig",
    "ChestXrayClassifier",
    "ChestXrayDetector",
    "TrainingConfig",
    "MetricsCalculator",
    "ClassificationMetrics",
    "DetectionMetrics",
    "Visualizer",
]
