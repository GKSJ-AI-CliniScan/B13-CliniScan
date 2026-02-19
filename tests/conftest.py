"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Tuple


@pytest.fixture
def sample_image() -> np.ndarray:
    """Generate a sample grayscale image for testing."""
    return np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)


@pytest.fixture
def sample_color_image() -> np.ndarray:
    """Generate a sample color image for testing."""
    return np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_bbox() -> dict:
    """Generate a sample bounding box."""
    return {
        'x_min': 100.0,
        'y_min': 100.0,
        'width': 200.0,
        'height': 150.0,
        'class_id': 0,
        'class_name': 'pathology',
    }
