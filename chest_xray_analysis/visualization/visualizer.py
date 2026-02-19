"""
Visualization for Grad-CAM and bounding boxes.
"""

from typing import Optional, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn
    
from chest_xray_analysis.models.detector import Detection


class Visualizer:
    """Generate interpretable visualizations."""
    
    def generate_gradcam(
        self,
        model: 'nn.Module',
        image: np.ndarray,
        target_layer: str,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        raise NotImplementedError("To be implemented in task 16.2")
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """Overlay heatmap on original image."""
        raise NotImplementedError("To be implemented in task 16.5")
    
    def draw_bounding_boxes(
        self,
        image: np.ndarray,
        detections: List[Detection],
        ground_truth: Optional[List[Detection]] = None,
        show_confidence: bool = True
    ) -> np.ndarray:
        """Draw bounding boxes with labels."""
        raise NotImplementedError("To be implemented in task 17.1")
