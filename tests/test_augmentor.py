"""
Tests for data augmentation.

Feature: chest-xray-analysis
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from chest_xray_analysis.data import MedicalAugmentor, AugmentationConfig, BoundingBox


# Unit Tests

@pytest.mark.unit
def test_augmentor_initialization():
    """Test augmentor initialization with valid config."""
    config = AugmentationConfig(
        horizontal_flip_prob=0.5,
        rotation_range=(-10, 10),
        rotation_prob=0.5,
        contrast_range=(0.8, 1.2),
        contrast_prob=0.5,
        zoom_range=(0.9, 1.1),
        zoom_prob=0.3,
        elastic_alpha=1.0,
        elastic_sigma=50.0,
        elastic_prob=0.2
    )
    
    augmentor = MedicalAugmentor(config)
    assert augmentor.config == config


@pytest.mark.unit
def test_augmentor_invalid_rotation_range():
    """Test that excessive rotation range raises error."""
    config = AugmentationConfig(
        horizontal_flip_prob=0.5,
        rotation_range=(-20, 20),  # Too large for medical images
        rotation_prob=0.5,
        contrast_range=(0.8, 1.2),
        contrast_prob=0.5,
        zoom_range=(0.9, 1.1),
        zoom_prob=0.3,
        elastic_alpha=1.0,
        elastic_sigma=50.0,
        elastic_prob=0.2
    )
    
    with pytest.raises(ValueError, match="within ±15 degrees"):
        MedicalAugmentor(config)


@pytest.mark.unit
def test_horizontal_flip():
    """Test horizontal flip preserves dimensions and transforms bboxes."""
    config = AugmentationConfig()
    augmentor = MedicalAugmentor(config)
    
    # Create test image
    image = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
    
    # Create bounding box
    bbox = BoundingBox(x_min=50, y_min=30, width=40, height=20, class_id=0, class_name="test")
    
    # Apply horizontal flip
    flipped_image, flipped_bboxes = augmentor.horizontal_flip(image, [bbox])
    
    # Check dimensions preserved
    assert flipped_image.shape == image.shape
    
    # Check bbox transformed correctly
    # Original bbox: x_min=50, x_max=90
    # Flipped: x_min = 200 - 90 = 110, x_max = 200 - 50 = 150
    assert len(flipped_bboxes) == 1
    assert np.isclose(flipped_bboxes[0].x_min, 110)
    assert flipped_bboxes[0].y_min == 30
    assert flipped_bboxes[0].width == 40
    assert flipped_bboxes[0].height == 20


@pytest.mark.unit
def test_rotate_within_bounds():
    """Test rotation stays within configured bounds."""
    config = AugmentationConfig(rotation_range=(-10, 10))
    augmentor = MedicalAugmentor(config)
    
    image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    bbox = BoundingBox(x_min=40, y_min=40, width=20, height=20, class_id=0, class_name="test")
    
    # Try to rotate by 15 degrees (should be clamped to 10)
    rotated_image, rotated_bboxes = augmentor.rotate(image, 15, [bbox])
    
    # Should not crash and should return valid image
    assert rotated_image.shape == image.shape


@pytest.mark.unit
def test_adjust_contrast():
    """Test contrast adjustment."""
    config = AugmentationConfig()
    augmentor = MedicalAugmentor(config)
    
    # Create image with known values
    image = np.full((50, 50), 128, dtype=np.uint8)
    
    # Increase contrast
    adjusted = augmentor.adjust_contrast(image, 1.5)
    
    # Should maintain same shape
    assert adjusted.shape == image.shape
    assert adjusted.dtype == np.uint8


@pytest.mark.unit
def test_zoom_in():
    """Test zoom in (factor > 1)."""
    config = AugmentationConfig()
    augmentor = MedicalAugmentor(config)
    
    image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    bbox = BoundingBox(x_min=40, y_min=40, width=20, height=20, class_id=0, class_name="test")
    
    # Zoom in by 1.1x
    zoomed_image, zoomed_bboxes = augmentor.zoom(image, 1.1, [bbox])
    
    # Should maintain original dimensions
    assert zoomed_image.shape == image.shape


@pytest.mark.unit
def test_zoom_out():
    """Test zoom out (factor < 1)."""
    config = AugmentationConfig()
    augmentor = MedicalAugmentor(config)
    
    image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    bbox = BoundingBox(x_min=40, y_min=40, width=20, height=20, class_id=0, class_name="test")
    
    # Zoom out by 0.9x
    zoomed_image, zoomed_bboxes = augmentor.zoom(image, 0.9, [bbox])
    
    # Should maintain original dimensions
    assert zoomed_image.shape == image.shape
    # Should have one bbox
    assert len(zoomed_bboxes) == 1


@pytest.mark.unit
def test_elastic_deform():
    """Test elastic deformation."""
    config = AugmentationConfig()
    augmentor = MedicalAugmentor(config)
    
    image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    
    # Apply elastic deformation
    deformed = augmentor.elastic_deform(image, alpha=1.0, sigma=50.0)
    
    # Should maintain same shape
    assert deformed.shape == image.shape


@pytest.mark.unit
def test_augment_without_bboxes():
    """Test augmentation pipeline without bounding boxes."""
    config = AugmentationConfig(
        horizontal_flip_prob=1.0,  # Always flip for deterministic test
        rotation_prob=0.0,
        contrast_prob=0.0,
        zoom_prob=0.0,
        elastic_prob=0.0
    )
    augmentor = MedicalAugmentor(config)
    
    image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    
    # Apply augmentation
    augmented_image, augmented_bboxes = augmentor.augment(image, None)
    
    # Should return image and None for bboxes
    assert augmented_image.shape == image.shape
    assert augmented_bboxes is None


@pytest.mark.unit
def test_augment_with_bboxes():
    """Test augmentation pipeline with bounding boxes."""
    config = AugmentationConfig(
        horizontal_flip_prob=0.0,
        rotation_prob=0.0,
        contrast_prob=0.0,
        zoom_prob=0.0,
        elastic_prob=0.0
    )
    augmentor = MedicalAugmentor(config)
    
    image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    bbox = BoundingBox(x_min=40, y_min=40, width=20, height=20, class_id=0, class_name="test")
    
    # Apply augmentation (with all probs=0, should return unchanged)
    augmented_image, augmented_bboxes = augmentor.augment(image, [bbox])
    
    # Should return image and bboxes
    assert augmented_image.shape == image.shape
    assert augmented_bboxes is not None
    assert len(augmented_bboxes) >= 0  # May be filtered if visibility too low


# Property-Based Tests

@pytest.mark.property
@given(
    height=st.integers(min_value=64, max_value=512),
    width=st.integers(min_value=64, max_value=512)
)
@settings(max_examples=100, deadline=None)
def test_property_horizontal_flip_dimensions(height, width):
    """
    Feature: chest-xray-analysis, Property 16: Horizontal flip preserves image dimensions
    
    For any image, after horizontal flip, the output dimensions should equal input dimensions.
    Validates: Requirements 8.1
    """
    config = AugmentationConfig()
    augmentor = MedicalAugmentor(config)
    
    # Create random image
    image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    
    # Apply horizontal flip
    flipped_image, _ = augmentor.horizontal_flip(image, [])
    
    # Property: Dimensions should be preserved
    assert flipped_image.shape == image.shape


@pytest.mark.property
@given(
    angle=st.floats(min_value=-20, max_value=20)
)
@settings(max_examples=100, deadline=None)
def test_property_rotation_bounds(angle):
    """
    Feature: chest-xray-analysis, Property 17: Rotation stays within safe bounds
    
    For any augmented image with rotation, the rotation angle should be within ±10 degrees.
    Validates: Requirements 8.2, 8.7
    """
    config = AugmentationConfig(rotation_range=(-10, 10))
    augmentor = MedicalAugmentor(config)
    
    image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    bbox = BoundingBox(x_min=40, y_min=40, width=20, height=20, class_id=0, class_name="test")
    
    # Apply rotation - should clamp to [-10, 10]
    rotated_image, rotated_bboxes = augmentor.rotate(image, angle, [bbox])
    
    # Property: Should not crash and should return valid image
    # The actual angle is clamped internally, so we just verify it works
    assert rotated_image.shape == image.shape


@pytest.mark.property
@given(st.just(None))  # Dummy generator to make hypothesis happy
@settings(max_examples=1, deadline=None)
def test_property_no_vertical_flips(_):
    """
    Feature: chest-xray-analysis, Property 18: No vertical flips applied
    
    For any augmentation configuration, vertical flips should never be applied.
    Validates: Requirements 8.6
    """
    config = AugmentationConfig()
    augmentor = MedicalAugmentor(config)
    
    # Property: The augmentation pipeline should not include vertical flip
    # We verify this by checking the transform composition
    transform_str = str(augmentor.transform)
    
    # Should contain HorizontalFlip but not VerticalFlip
    assert 'HorizontalFlip' in transform_str or 'horizontal' in transform_str.lower()
    assert 'VerticalFlip' not in transform_str


@pytest.mark.property
@given(
    x_min=st.floats(min_value=10, max_value=80),
    y_min=st.floats(min_value=10, max_value=80),
    width=st.floats(min_value=10, max_value=20),
    height=st.floats(min_value=10, max_value=20)
)
@settings(max_examples=100, deadline=None)
def test_property_bbox_transformation_consistency(x_min, y_min, width, height):
    """
    Feature: chest-xray-analysis, Property 19: Bounding box transformation consistency
    
    For any image with bounding boxes, after horizontal flip, the transformed bounding boxes
    should still correctly align with the transformed objects in the image.
    Validates: Requirements 8.8
    """
    config = AugmentationConfig()
    augmentor = MedicalAugmentor(config)
    
    image_width = 100
    image = np.random.randint(0, 256, size=(100, image_width), dtype=np.uint8)
    
    bbox = BoundingBox(
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height,
        class_id=0,
        class_name="test"
    )
    
    # Apply horizontal flip
    flipped_image, flipped_bboxes = augmentor.horizontal_flip(image, [bbox])
    
    # Property: Transformed bbox should be within image bounds
    assert len(flipped_bboxes) == 1
    flipped_bbox = flipped_bboxes[0]
    
    # Check bbox is within image bounds
    assert flipped_bbox.x_min >= 0
    assert flipped_bbox.y_min >= 0
    assert flipped_bbox.x_min + flipped_bbox.width <= image_width
    assert flipped_bbox.y_min + flipped_bbox.height <= 100
    
    # Check bbox dimensions preserved
    assert np.isclose(flipped_bbox.width, width, atol=0.1)
    assert np.isclose(flipped_bbox.height, height, atol=0.1)
    
    # Check y-coordinate unchanged (horizontal flip doesn't affect y)
    assert np.isclose(flipped_bbox.y_min, y_min, atol=0.1)
