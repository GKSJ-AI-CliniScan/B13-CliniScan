"""
Metrics calculation for classification and detection.
"""

from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from chest_xray_analysis.models.detector import Detection


@dataclass
class ClassificationMetrics:
    """Classification metrics container."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
    per_class_metrics: Dict[str, Dict[str, float]]


@dataclass
class DetectionMetrics:
    """Detection metrics container."""
    map_50: float
    map_75: float
    map_50_95: float
    per_class_ap: Dict[str, float]
    mean_iou: float


class MetricsCalculator:
    """Compute evaluation metrics."""
    
    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> ClassificationMetrics:
        """Compute AUC, F1, precision, recall, confusion matrix."""
        raise NotImplementedError("To be implemented in task 14.2")
    
    def compute_detection_metrics(
        self,
        predictions: List[List['Detection']],
        ground_truth: List[List['Detection']],
        iou_threshold: float = 0.5
    ) -> DetectionMetrics:
        """Compute mAP, IoU, per-class AP."""
        raise NotImplementedError("To be implemented in task 14.3")
