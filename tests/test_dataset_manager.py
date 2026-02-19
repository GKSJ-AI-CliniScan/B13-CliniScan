"""
Tests for dataset management and splitting.

Feature: chest-xray-analysis
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from chest_xray_analysis.data import DatasetManager, DatasetSplit


# Unit Tests

@pytest.mark.unit
def test_simple_split():
    """Test basic dataset splitting."""
    manager = DatasetManager()
    
    images = [f"image_{i}.png" for i in range(100)]
    labels = [i % 3 for i in range(100)]  # 3 classes
    
    split = manager.split_dataset(
        images, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=False
    )
    
    # Check all images are accounted for
    total = len(split.train_images) + len(split.val_images) + len(split.test_images)
    assert total == 100
    
    # Check approximate ratios
    assert 65 <= len(split.train_images) <= 75
    assert 10 <= len(split.val_images) <= 20
    assert 10 <= len(split.test_images) <= 20


@pytest.mark.unit
def test_stratified_split():
    """Test stratified splitting maintains class distribution."""
    manager = DatasetManager()
    
    # Create imbalanced dataset: 70 class 0, 20 class 1, 10 class 2
    images = [f"image_{i}.png" for i in range(100)]
    labels = [0] * 70 + [1] * 20 + [2] * 10
    
    split = manager.split_dataset(
        images, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True
    )
    
    # Check class distribution in train set (should be ~70%, 20%, 10%)
    train_class_counts = [split.train_labels.count(i) for i in range(3)]
    train_total = len(split.train_labels)
    
    assert abs(train_class_counts[0] / train_total - 0.7) < 0.1
    assert abs(train_class_counts[1] / train_total - 0.2) < 0.1
    assert abs(train_class_counts[2] / train_total - 0.1) < 0.1


@pytest.mark.unit
def test_patient_level_split():
    """Test patient-level splitting prevents leakage."""
    manager = DatasetManager()
    
    # 3 patients with multiple images each
    images = [
        "p1_img1.png", "p1_img2.png", "p1_img3.png",
        "p2_img1.png", "p2_img2.png",
        "p3_img1.png", "p3_img2.png", "p3_img3.png", "p3_img4.png"
    ]
    labels = [0, 0, 0, 1, 1, 0, 0, 0, 0]
    patient_ids = ["p1", "p1", "p1", "p2", "p2", "p3", "p3", "p3", "p3"]
    
    split = manager.split_dataset(
        images, labels,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratify=False,
        patient_ids=patient_ids
    )
    
    # Extract patient IDs from split images
    def get_patient_id(img_path):
        return img_path.split('_')[0]
    
    train_patients = set(get_patient_id(img) for img in split.train_images)
    val_patients = set(get_patient_id(img) for img in split.val_images)
    test_patients = set(get_patient_id(img) for img in split.test_images)
    
    # Check no patient appears in multiple splits
    assert len(train_patients & val_patients) == 0
    assert len(train_patients & test_patients) == 0
    assert len(val_patients & test_patients) == 0
    
    # Check all patients are accounted for
    all_patients = train_patients | val_patients | test_patients
    assert all_patients == {"p1", "p2", "p3"}


@pytest.mark.unit
def test_invalid_ratios():
    """Test that invalid ratios raise errors."""
    manager = DatasetManager()
    
    images = [f"image_{i}.png" for i in range(10)]
    labels = [0] * 10
    
    # Ratios don't sum to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        manager.split_dataset(images, labels, 0.5, 0.3, 0.3)
    
    # Negative ratio
    with pytest.raises(ValueError, match="non-negative"):
        manager.split_dataset(images, labels, 0.8, -0.1, 0.3)


@pytest.mark.unit
def test_mismatched_lengths():
    """Test that mismatched input lengths raise errors."""
    manager = DatasetManager()
    
    images = [f"image_{i}.png" for i in range(10)]
    labels = [0] * 5  # Wrong length
    
    with pytest.raises(ValueError, match="same length"):
        manager.split_dataset(images, labels, 0.7, 0.15, 0.15)


@pytest.mark.unit
def test_empty_dataset():
    """Test that empty dataset raises error."""
    manager = DatasetManager()
    
    with pytest.raises(ValueError, match="empty dataset"):
        manager.split_dataset([], [], 0.7, 0.15, 0.15)


@pytest.mark.unit
def test_no_validation_set():
    """Test splitting with no validation set."""
    manager = DatasetManager()
    
    images = [f"image_{i}.png" for i in range(100)]
    labels = [i % 2 for i in range(100)]
    
    split = manager.split_dataset(
        images, labels,
        train_ratio=0.8,
        val_ratio=0.0,
        test_ratio=0.2,
        stratify=False
    )
    
    assert len(split.val_images) == 0
    assert len(split.val_labels) == 0
    assert len(split.train_images) + len(split.test_images) == 100


@pytest.mark.unit
def test_no_test_set():
    """Test splitting with no test set."""
    manager = DatasetManager()
    
    images = [f"image_{i}.png" for i in range(100)]
    labels = [i % 2 for i in range(100)]
    
    split = manager.split_dataset(
        images, labels,
        train_ratio=0.85,
        val_ratio=0.15,
        test_ratio=0.0,
        stratify=False
    )
    
    assert len(split.test_images) == 0
    assert len(split.test_labels) == 0
    assert len(split.train_images) + len(split.val_images) == 100


# Property-Based Tests

@pytest.mark.property
@given(
    num_samples=st.integers(min_value=10, max_value=200),
    train_ratio=st.floats(min_value=0.5, max_value=0.8),
    val_ratio=st.floats(min_value=0.1, max_value=0.3)
)
@settings(max_examples=100, deadline=None)
def test_property_split_completeness(num_samples, train_ratio, val_ratio):
    """
    Feature: chest-xray-analysis, Property 20: Dataset split completeness and disjointness
    
    For any dataset split, every sample should appear in exactly one subset.
    Validates: Requirements 9.1, 9.3
    """
    # Ensure ratios sum to 1.0
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0 or test_ratio > 0.5:
        assume(False)
    
    manager = DatasetManager()
    
    images = [f"image_{i}.png" for i in range(num_samples)]
    labels = [i % 3 for i in range(num_samples)]
    
    split = manager.split_dataset(
        images, labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=False
    )
    
    # Property 1: All samples accounted for
    total_images = len(split.train_images) + len(split.val_images) + len(split.test_images)
    assert total_images == num_samples
    
    # Property 2: No duplicates within each split
    assert len(set(split.train_images)) == len(split.train_images)
    assert len(set(split.val_images)) == len(split.val_images)
    assert len(set(split.test_images)) == len(split.test_images)
    
    # Property 3: No overlap between splits
    train_set = set(split.train_images)
    val_set = set(split.val_images)
    test_set = set(split.test_images)
    
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
    
    # Property 4: Labels match images
    assert len(split.train_images) == len(split.train_labels)
    assert len(split.val_images) == len(split.val_labels)
    assert len(split.test_images) == len(split.test_labels)


@pytest.mark.property
@given(
    num_samples=st.integers(min_value=50, max_value=200),
    train_ratio=st.floats(min_value=0.6, max_value=0.8)
)
@settings(max_examples=100, deadline=None)
def test_property_split_ratios(num_samples, train_ratio):
    """
    Feature: chest-xray-analysis, Property 21: Split ratios respected
    
    For any dataset split with specified ratios, the actual sizes should match
    the requested ratios within acceptable tolerance.
    Validates: Requirements 9.2
    """
    val_ratio = 0.15
    test_ratio = 1.0 - train_ratio - val_ratio
    
    if test_ratio < 0.05 or test_ratio > 0.4:
        assume(False)
    
    manager = DatasetManager()
    
    images = [f"image_{i}.png" for i in range(num_samples)]
    labels = [i % 2 for i in range(num_samples)]
    
    split = manager.split_dataset(
        images, labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=False
    )
    
    # Property: Actual ratios should be close to requested ratios
    # Allow tolerance of ±2 samples for rounding
    actual_train_ratio = len(split.train_images) / num_samples
    actual_val_ratio = len(split.val_images) / num_samples
    actual_test_ratio = len(split.test_images) / num_samples
    
    tolerance = 3.0 / num_samples  # ±3 samples
    
    assert abs(actual_train_ratio - train_ratio) < tolerance + 0.05
    assert abs(actual_val_ratio - val_ratio) < tolerance + 0.05
    assert abs(actual_test_ratio - test_ratio) < tolerance + 0.05


@pytest.mark.property
@given(
    num_samples=st.integers(min_value=60, max_value=150)
)
@settings(max_examples=100, deadline=None)
def test_property_stratified_distribution(num_samples):
    """
    Feature: chest-xray-analysis, Property 22: Stratified split maintains class distribution
    
    For any dataset with stratified splitting, the class distribution in each subset
    should be similar to the overall dataset distribution.
    Validates: Requirements 9.4
    """
    manager = DatasetManager()
    
    # Create dataset with known class distribution (60% class 0, 40% class 1)
    num_class_0 = int(num_samples * 0.6)
    num_class_1 = num_samples - num_class_0
    
    images = [f"image_{i}.png" for i in range(num_samples)]
    labels = [0] * num_class_0 + [1] * num_class_1
    
    split = manager.split_dataset(
        images, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True
    )
    
    # Property: Class distribution should be maintained in each split
    # Overall distribution
    overall_class_0_ratio = num_class_0 / num_samples
    
    # Train distribution
    if len(split.train_labels) > 0:
        train_class_0_ratio = split.train_labels.count(0) / len(split.train_labels)
        assert abs(train_class_0_ratio - overall_class_0_ratio) < 0.15
    
    # Val distribution
    if len(split.val_labels) > 0:
        val_class_0_ratio = split.val_labels.count(0) / len(split.val_labels)
        assert abs(val_class_0_ratio - overall_class_0_ratio) < 0.2
    
    # Test distribution
    if len(split.test_labels) > 0:
        test_class_0_ratio = split.test_labels.count(0) / len(split.test_labels)
        assert abs(test_class_0_ratio - overall_class_0_ratio) < 0.2


@pytest.mark.property
@given(
    num_patients=st.integers(min_value=5, max_value=20),
    images_per_patient=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=100, deadline=None)
def test_property_patient_level_no_leakage(num_patients, images_per_patient):
    """
    Feature: chest-xray-analysis, Property 23: Patient-level split prevents leakage
    
    For any dataset with patient IDs, when patient-level splitting is enabled,
    all images from the same patient should appear in exactly one subset.
    Validates: Requirements 9.5
    """
    manager = DatasetManager()
    
    # Create dataset with multiple images per patient
    images = []
    labels = []
    patient_ids = []
    
    for patient_idx in range(num_patients):
        patient_id = f"patient_{patient_idx}"
        patient_label = patient_idx % 2  # Alternate between classes
        
        for img_idx in range(images_per_patient):
            images.append(f"{patient_id}_image_{img_idx}.png")
            labels.append(patient_label)
            patient_ids.append(patient_id)
    
    split = manager.split_dataset(
        images, labels,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratify=False,
        patient_ids=patient_ids
    )
    
    # Extract patient IDs from each split
    def extract_patient_id(img_path):
        return img_path.split('_image_')[0]
    
    train_patients = set(extract_patient_id(img) for img in split.train_images)
    val_patients = set(extract_patient_id(img) for img in split.val_images)
    test_patients = set(extract_patient_id(img) for img in split.test_images)
    
    # Property: No patient should appear in multiple splits
    assert len(train_patients & val_patients) == 0
    assert len(train_patients & test_patients) == 0
    assert len(val_patients & test_patients) == 0
    
    # Property: All patients should be accounted for
    all_split_patients = train_patients | val_patients | test_patients
    unique_patients = set(f"patient_{i}" for i in range(num_patients))
    assert all_split_patients == unique_patients
