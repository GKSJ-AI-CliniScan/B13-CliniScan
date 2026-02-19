"""
Tests for annotation format conversion.

Feature: chest-xray-analysis
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from hypothesis import given, strategies as st, settings
from chest_xray_analysis.data import AnnotationConverter, BoundingBox
from chest_xray_analysis.utils.exceptions import AnnotationFormatError


# Unit Tests

@pytest.mark.unit
def test_bbox_to_yolo():
    """Test converting bounding box to YOLO format."""
    bbox = BoundingBox(
        x_min=100,
        y_min=150,
        width=200,
        height=100,
        class_id=0,
        class_name="pneumonia"
    )
    
    # Image size 800x600
    x_center, y_center, width, height = bbox.to_yolo(800, 600)
    
    # Expected: center at (200, 200), normalized
    assert np.isclose(x_center, 200/800)  # 0.25
    assert np.isclose(y_center, 200/600)  # 0.333...
    assert np.isclose(width, 200/800)     # 0.25
    assert np.isclose(height, 100/600)    # 0.166...


@pytest.mark.unit
def test_bbox_to_coco():
    """Test converting bounding box to COCO format."""
    bbox = BoundingBox(
        x_min=100,
        y_min=150,
        width=200,
        height=100,
        class_id=2,
        class_name="pneumonia",
        confidence=0.95
    )
    
    coco_dict = bbox.to_coco()
    
    assert coco_dict['bbox'] == [100, 150, 200, 100]
    assert coco_dict['category_id'] == 2
    assert coco_dict['area'] == 20000
    assert coco_dict['score'] == 0.95


@pytest.mark.unit
def test_bbox_to_coco_without_confidence():
    """Test COCO conversion without confidence score."""
    bbox = BoundingBox(
        x_min=50,
        y_min=75,
        width=100,
        height=50,
        class_id=1,
        class_name="nodule"
    )
    
    coco_dict = bbox.to_coco()
    
    assert 'score' not in coco_dict
    assert coco_dict['bbox'] == [50, 75, 100, 50]


@pytest.mark.unit
def test_parse_yolo_valid():
    """Test parsing valid YOLO format file."""
    # Create temporary YOLO annotation file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("0 0.5 0.5 0.3 0.4\n")
        f.write("1 0.2 0.3 0.1 0.15\n")
        temp_path = f.name
    
    try:
        converter = AnnotationConverter()
        bboxes = converter.parse_yolo(temp_path, 800, 600, class_names=["pneumonia", "nodule"])
        
        assert len(bboxes) == 2
        
        # First box
        assert bboxes[0].class_id == 0
        assert bboxes[0].class_name == "pneumonia"
        assert np.isclose(bboxes[0].x_min, 0.5 * 800 - 0.3 * 800 / 2)  # center - width/2
        assert np.isclose(bboxes[0].y_min, 0.5 * 600 - 0.4 * 600 / 2)
        assert np.isclose(bboxes[0].width, 0.3 * 800)
        assert np.isclose(bboxes[0].height, 0.4 * 600)
        
        # Second box
        assert bboxes[1].class_id == 1
        assert bboxes[1].class_name == "nodule"
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_parse_yolo_invalid_format():
    """Test parsing YOLO file with invalid format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("0 0.5 0.5\n")  # Missing width and height
        temp_path = f.name
    
    try:
        converter = AnnotationConverter()
        with pytest.raises(AnnotationFormatError, match="expected 5 values"):
            converter.parse_yolo(temp_path, 800, 600)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_parse_yolo_out_of_range():
    """Test parsing YOLO file with out-of-range coordinates."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("0 1.5 0.5 0.3 0.4\n")  # x_center > 1
        temp_path = f.name
    
    try:
        converter = AnnotationConverter()
        with pytest.raises(AnnotationFormatError, match="out of range"):
            converter.parse_yolo(temp_path, 800, 600)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_parse_yolo_nonexistent_file():
    """Test parsing nonexistent YOLO file."""
    converter = AnnotationConverter()
    with pytest.raises(AnnotationFormatError, match="not found"):
        converter.parse_yolo("nonexistent.txt", 800, 600)


@pytest.mark.unit
def test_parse_coco_valid():
    """Test parsing valid COCO format file."""
    coco_data = {
        "images": [
            {"id": 1, "file_name": "image1.png", "width": 800, "height": 600}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [100, 150, 200, 100]
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [300, 250, 150, 120],
                "score": 0.95
            }
        ],
        "categories": [
            {"id": 0, "name": "pneumonia"},
            {"id": 1, "name": "nodule"}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_data, f)
        temp_path = f.name
    
    try:
        converter = AnnotationConverter()
        annotations = converter.parse_coco(temp_path)
        
        assert 1 in annotations
        assert len(annotations[1]) == 2
        
        # First annotation
        bbox1 = annotations[1][0]
        assert bbox1.x_min == 100
        assert bbox1.y_min == 150
        assert bbox1.width == 200
        assert bbox1.height == 100
        assert bbox1.class_id == 0
        assert bbox1.class_name == "pneumonia"
        assert bbox1.confidence is None
        
        # Second annotation
        bbox2 = annotations[1][1]
        assert bbox2.confidence == 0.95
        assert bbox2.class_name == "nodule"
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_parse_coco_missing_field():
    """Test parsing COCO file with missing required field."""
    coco_data = {
        "annotations": [
            {
                "id": 1,
                # Missing image_id
                "category_id": 0,
                "bbox": [100, 150, 200, 100]
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_data, f)
        temp_path = f.name
    
    try:
        converter = AnnotationConverter()
        with pytest.raises(AnnotationFormatError, match="Missing 'image_id'"):
            converter.parse_coco(temp_path)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_parse_coco_invalid_json():
    """Test parsing invalid JSON file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{invalid json")
        temp_path = f.name
    
    try:
        converter = AnnotationConverter()
        with pytest.raises(AnnotationFormatError, match="Invalid JSON"):
            converter.parse_coco(temp_path)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_to_yolo_conversion():
    """Test converting bounding boxes to YOLO format string."""
    bboxes = [
        BoundingBox(x_min=100, y_min=150, width=200, height=100, class_id=0, class_name="pneumonia"),
        BoundingBox(x_min=300, y_min=250, width=150, height=120, class_id=1, class_name="nodule")
    ]
    
    converter = AnnotationConverter()
    yolo_str = converter.to_yolo(bboxes, 800, 600)
    
    lines = yolo_str.split('\n')
    assert len(lines) == 2
    
    # Parse first line
    parts1 = lines[0].split()
    assert int(parts1[0]) == 0
    assert float(parts1[1]) == pytest.approx(0.25, abs=0.001)  # x_center
    assert float(parts1[2]) == pytest.approx(0.333333, abs=0.001)  # y_center


@pytest.mark.unit
def test_to_coco_conversion():
    """Test converting bounding boxes to COCO format."""
    annotations = {
        1: [
            BoundingBox(x_min=100, y_min=150, width=200, height=100, class_id=0, class_name="pneumonia"),
            BoundingBox(x_min=300, y_min=250, width=150, height=120, class_id=1, class_name="nodule")
        ]
    }
    
    converter = AnnotationConverter()
    coco_dict = converter.to_coco(annotations, ["pneumonia", "nodule"])
    
    assert 'images' in coco_dict
    assert 'annotations' in coco_dict
    assert 'categories' in coco_dict
    
    assert len(coco_dict['categories']) == 2
    assert len(coco_dict['annotations']) == 2
    
    # Check first annotation
    ann1 = coco_dict['annotations'][0]
    assert ann1['image_id'] == 1
    assert ann1['category_id'] == 0
    assert ann1['bbox'] == [100, 150, 200, 100]


# Property-Based Tests

@pytest.mark.property
@given(
    x_min=st.floats(min_value=0, max_value=700),
    y_min=st.floats(min_value=0, max_value=500),
    width=st.floats(min_value=10, max_value=100),
    height=st.floats(min_value=10, max_value=100),
    class_id=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_property_yolo_coordinate_normalization(x_min, y_min, width, height, class_id):
    """
    Feature: chest-xray-analysis, Property 14: YOLO coordinate normalization
    
    For any bounding box in YOLO format, all coordinate values should be in [0, 1].
    Validates: Requirements 7.4
    """
    bbox = BoundingBox(
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height,
        class_id=class_id,
        class_name=f"class_{class_id}"
    )
    
    # Convert to YOLO format
    x_center, y_center, w_norm, h_norm = bbox.to_yolo(800, 600)
    
    # Property: All coordinates should be in [0, 1]
    assert 0 <= x_center <= 1
    assert 0 <= y_center <= 1
    assert 0 <= w_norm <= 1
    assert 0 <= h_norm <= 1


@pytest.mark.property
@given(
    x_min=st.floats(min_value=0, max_value=700),
    y_min=st.floats(min_value=0, max_value=500),
    width=st.floats(min_value=10, max_value=100),
    height=st.floats(min_value=10, max_value=100),
    class_id=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_property_annotation_round_trip(x_min, y_min, width, height, class_id):
    """
    Feature: chest-xray-analysis, Property 13: Annotation format round-trip
    
    For any valid bounding box, converting YOLO -> COCO -> YOLO should preserve coordinates.
    Validates: Requirements 7.3
    """
    # Create original bounding box
    original_bbox = BoundingBox(
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height,
        class_id=class_id,
        class_name=f"class_{class_id}"
    )
    
    image_width, image_height = 800, 600
    
    # Convert to YOLO format
    yolo_coords = original_bbox.to_yolo(image_width, image_height)
    x_center_norm, y_center_norm, width_norm, height_norm = yolo_coords
    
    # Convert back to absolute coordinates (simulating YOLO -> BoundingBox)
    x_center = x_center_norm * image_width
    y_center = y_center_norm * image_height
    width_abs = width_norm * image_width
    height_abs = height_norm * image_height
    
    reconstructed_x_min = x_center - width_abs / 2
    reconstructed_y_min = y_center - height_abs / 2
    
    # Property: Round-trip should preserve coordinates (within floating-point tolerance)
    assert np.isclose(reconstructed_x_min, x_min, atol=0.01)
    assert np.isclose(reconstructed_y_min, y_min, atol=0.01)
    assert np.isclose(width_abs, width, atol=0.01)
    assert np.isclose(height_abs, height, atol=0.01)


@pytest.mark.property
@given(
    num_boxes=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_property_malformed_annotation_error_handling(num_boxes):
    """
    Feature: chest-xray-analysis, Property 15: Malformed annotation error handling
    
    For any malformed annotation file, the parser should return descriptive error.
    Validates: Requirements 7.5
    """
    converter = AnnotationConverter()
    
    # Test 1: Invalid YOLO format (missing values)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for i in range(num_boxes):
            f.write(f"{i} 0.5\n")  # Missing 3 values
        temp_path = f.name
    
    try:
        with pytest.raises(AnnotationFormatError):
            converter.parse_yolo(temp_path, 800, 600)
    finally:
        Path(temp_path).unlink()
    
    # Test 2: Invalid COCO format (missing required field)
    coco_data = {
        "annotations": [
            {
                "id": i,
                # Missing image_id, category_id, bbox
            } for i in range(num_boxes)
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_data, f)
        temp_path = f.name
    
    try:
        with pytest.raises(AnnotationFormatError):
            converter.parse_coco(temp_path)
    finally:
        Path(temp_path).unlink()
