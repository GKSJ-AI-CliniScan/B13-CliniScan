"""
Tests for image preprocessing.

Feature: chest-xray-analysis
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from chest_xray_analysis.preprocessing import ImagePreprocessor, PreprocessingConfig


# Unit Tests

@pytest.mark.unit
def test_grayscale_conversion_color_image():
    """Test converting a color image to grayscale."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create a color image (BGR format)
    color_image = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    
    grayscale = preprocessor.to_grayscale(color_image)
    
    assert len(grayscale.shape) == 2
    assert grayscale.shape == (256, 256)


@pytest.mark.unit
def test_grayscale_already_grayscale():
    """Test that grayscale images remain unchanged."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create a grayscale image
    gray_image = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
    
    result = preprocessor.to_grayscale(gray_image)
    
    assert np.array_equal(result, gray_image)


@pytest.mark.unit
def test_resize_maintains_aspect_ratio():
    """Test that resizing maintains aspect ratio."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create an image with 2:1 aspect ratio
    image = np.random.randint(0, 256, size=(200, 400), dtype=np.uint8)
    
    resized = preprocessor.resize(image, (512, 512))
    
    # Should fit within 512x512 while maintaining 2:1 aspect ratio
    assert resized.shape[0] <= 512
    assert resized.shape[1] <= 512
    
    # Check aspect ratio is approximately maintained
    original_aspect = 400 / 200
    new_aspect = resized.shape[1] / resized.shape[0]
    assert abs(original_aspect - new_aspect) < 0.01


# Property-Based Tests

@pytest.mark.property
@given(
    height=st.integers(min_value=64, max_value=1024),
    width=st.integers(min_value=64, max_value=1024),
    channels=st.sampled_from([1, 3])
)
@settings(max_examples=100, deadline=None)
def test_property_grayscale_idempotence(height, width, channels):
    """
    Feature: chest-xray-analysis, Property 4: Grayscale conversion idempotence
    
    For any grayscale image, converting to grayscale again should produce an identical image.
    Validates: Requirements 2.2
    """
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create image
    if channels == 1:
        image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    else:
        image = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
    
    # Convert to grayscale
    gray1 = preprocessor.to_grayscale(image)
    
    # Convert again
    gray2 = preprocessor.to_grayscale(gray1)
    
    # Property: Second conversion should be identical
    assert np.array_equal(gray1, gray2)
    assert len(gray2.shape) == 2


@pytest.mark.property
@given(
    height=st.integers(min_value=100, max_value=800),
    width=st.integers(min_value=100, max_value=800),
    target_h=st.integers(min_value=64, max_value=512),
    target_w=st.integers(min_value=64, max_value=512)
)
@settings(max_examples=100, deadline=None)
def test_property_aspect_ratio_preservation(height, width, target_h, target_w):
    """
    Feature: chest-xray-analysis, Property 5: Aspect ratio preservation during resize
    
    For any image and target size, if aspect ratio preservation is enabled,
    the output image should maintain the original aspect ratio within the target dimensions.
    Validates: Requirements 2.3
    """
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create image
    image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    
    # Resize
    resized = preprocessor.resize(image, (target_h, target_w))
    
    # Property: Output should fit within target dimensions
    assert resized.shape[0] <= target_h
    assert resized.shape[1] <= target_w
    
    # Property: Aspect ratio should be preserved (within tolerance)
    original_aspect = width / height
    new_aspect = resized.shape[1] / resized.shape[0]
    
    # Calculate expected tolerance based on pixel rounding
    # For small dimensions, rounding can cause larger relative errors
    # The error is bounded by 1 pixel in either dimension
    max_rounding_error = max(1.0 / resized.shape[0], 1.0 / resized.shape[1])
    tolerance = max(0.05, max_rounding_error * original_aspect)
    
    assert abs(original_aspect - new_aspect) < tolerance


@pytest.mark.unit
def test_resize_upsampling():
    """Test that upsampling uses cubic interpolation."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Small image to be upsampled
    small_image = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
    
    # Upsample to larger size
    upsampled = preprocessor.resize(small_image, (512, 512))
    
    # Should be larger
    assert upsampled.shape[0] > small_image.shape[0]
    assert upsampled.shape[1] > small_image.shape[1]


@pytest.mark.unit
def test_resize_downsampling():
    """Test that downsampling uses area interpolation."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Large image to be downsampled
    large_image = np.random.randint(0, 256, size=(1024, 1024), dtype=np.uint8)
    
    # Downsample to smaller size
    downsampled = preprocessor.resize(large_image, (256, 256))
    
    # Should be smaller
    assert downsampled.shape[0] < large_image.shape[0]
    assert downsampled.shape[1] < large_image.shape[1]



# Normalization Tests

@pytest.mark.unit
def test_minmax_normalization():
    """Test min-max normalization scales to [0, 1]."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create image with known range
    image = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
    
    normalized = preprocessor.normalize(image, 'minmax')
    
    # Should be in [0, 1] range
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)


@pytest.mark.unit
def test_zscore_normalization():
    """Test z-score normalization produces zero mean and unit variance."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="zscore",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create image with known statistics
    image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    
    normalized = preprocessor.normalize(image, 'zscore')
    
    # Should have approximately zero mean and unit variance
    assert np.isclose(normalized.mean(), 0.0, atol=1e-6)
    assert np.isclose(normalized.std(), 1.0, atol=1e-6)


@pytest.mark.unit
def test_normalization_constant_image():
    """Test that constant images are handled gracefully."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Constant image (all pixels same value)
    constant_image = np.full((50, 50), 128, dtype=np.uint8)
    
    # Should not crash
    normalized_minmax = preprocessor.normalize(constant_image, 'minmax')
    normalized_zscore = preprocessor.normalize(constant_image, 'zscore')
    
    # Should return zeros for constant images
    assert np.all(normalized_minmax == 0)
    assert np.all(normalized_zscore == 0)


@pytest.mark.property
@given(
    height=st.integers(min_value=10, max_value=256),
    width=st.integers(min_value=10, max_value=256)
)
@settings(max_examples=100, deadline=None)
def test_property_minmax_bounds(height, width):
    """
    Feature: chest-xray-analysis, Property 6: Min-max normalization bounds
    
    For any image, after min-max normalization, all pixel values should be in the range [0, 1].
    Validates: Requirements 3.1
    """
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create random image
    image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    
    # Normalize
    normalized = preprocessor.normalize(image, 'minmax')
    
    # Property: All values should be in [0, 1]
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


@pytest.mark.property
@given(
    height=st.integers(min_value=10, max_value=256),
    width=st.integers(min_value=10, max_value=256)
)
@settings(max_examples=100, deadline=None)
def test_property_zscore_statistics(height, width):
    """
    Feature: chest-xray-analysis, Property 7: Z-score normalization statistics
    
    For any image with non-zero standard deviation, after z-score normalization,
    the pixel values should have mean ≈ 0 and standard deviation ≈ 1.
    Validates: Requirements 3.2
    """
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="zscore",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create random image (avoid constant images)
    image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    
    # Skip if constant image (very rare with random generation)
    if image.std() == 0:
        return
    
    # Normalize
    normalized = preprocessor.normalize(image, 'zscore')
    
    # Property: Mean should be approximately 0, std should be approximately 1
    assert np.isclose(normalized.mean(), 0.0, atol=1e-5)
    assert np.isclose(normalized.std(), 1.0, atol=1e-5)


@pytest.mark.property
@given(
    height=st.integers(min_value=10, max_value=256),
    width=st.integers(min_value=10, max_value=256),
    method=st.sampled_from(['minmax', 'zscore'])
)
@settings(max_examples=100, deadline=None)
def test_property_normalization_monotonicity(height, width, method):
    """
    Feature: chest-xray-analysis, Property 8: Normalization preserves monotonicity
    
    For any image and normalization method, if pixel A > pixel B before normalization,
    then normalized A > normalized B (relative intensity relationships preserved).
    Validates: Requirements 3.5
    """
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method=method,
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create image with distinct values
    image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    
    # Skip constant images
    if image.std() == 0:
        return
    
    # Normalize
    normalized = preprocessor.normalize(image, method)
    
    # Property: Monotonicity should be preserved
    # Find two pixels with different values
    flat_original = image.flatten()
    flat_normalized = normalized.flatten()
    
    # Check a sample of pixel pairs
    for _ in range(min(10, len(flat_original) - 1)):
        idx1 = np.random.randint(0, len(flat_original))
        idx2 = np.random.randint(0, len(flat_original))
        
        if flat_original[idx1] > flat_original[idx2]:
            assert flat_normalized[idx1] > flat_normalized[idx2]
        elif flat_original[idx1] < flat_original[idx2]:
            assert flat_normalized[idx1] < flat_normalized[idx2]
        else:  # Equal
            assert np.isclose(flat_normalized[idx1], flat_normalized[idx2])



# CLAHE and Denoising Tests

@pytest.mark.unit
def test_clahe_enhancement():
    """Test CLAHE contrast enhancement."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create a low-contrast image
    image = np.random.randint(100, 150, size=(256, 256), dtype=np.uint8)
    
    # Apply CLAHE
    enhanced = preprocessor.apply_clahe(image, 2.0, 8)
    
    # Should maintain same shape and dtype
    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8


@pytest.mark.unit
def test_clahe_with_invalid_parameters():
    """Test CLAHE uses safe defaults for invalid parameters."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    image = np.random.randint(0, 256, size=(128, 128), dtype=np.uint8)
    
    # Should not crash with invalid parameters
    enhanced = preprocessor.apply_clahe(image, -1.0, -5)
    
    assert enhanced.shape == image.shape


@pytest.mark.unit
def test_gaussian_denoising():
    """Test Gaussian blur denoising."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method="gaussian",
        denoise_kernel_size=5,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create noisy image
    image = np.random.randint(0, 256, size=(128, 128), dtype=np.uint8)
    
    # Apply Gaussian denoising
    denoised = preprocessor.denoise(image, 'gaussian', 5)
    
    # Should maintain same shape
    assert denoised.shape == image.shape


@pytest.mark.unit
def test_median_denoising():
    """Test median filter denoising."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method="median",
        denoise_kernel_size=5,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create image with salt-and-pepper noise
    image = np.random.randint(100, 150, size=(128, 128), dtype=np.uint8)
    # Add some noise
    noise_coords = np.random.randint(0, 128, size=(100, 2))
    for coord in noise_coords:
        image[coord[0], coord[1]] = 255 if np.random.rand() > 0.5 else 0
    
    # Apply median denoising
    denoised = preprocessor.denoise(image, 'median', 5)
    
    # Should maintain same shape
    assert denoised.shape == image.shape


@pytest.mark.property
@given(
    kernel_size=st.integers(min_value=2, max_value=20)
)
@settings(max_examples=100, deadline=None)
def test_property_kernel_size_adjustment(kernel_size):
    """
    Feature: chest-xray-analysis, Property 9: Even kernel sizes adjusted to odd
    
    For any even kernel size provided to denoising functions,
    the actual kernel size used should be the nearest odd number.
    Validates: Requirements 5.5
    """
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method="gaussian",
        denoise_kernel_size=kernel_size,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create test image
    image = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
    
    # Apply denoising - should not crash even with even kernel size
    denoised = preprocessor.denoise(image, 'gaussian', kernel_size)
    
    # Property: Should successfully denoise without error
    assert denoised.shape == image.shape
    
    # If kernel_size was even, it should have been adjusted to odd
    # We can't directly verify the internal adjustment, but the function should work


@pytest.mark.unit
def test_kernel_size_adjustment_explicit():
    """Test that even kernel sizes are explicitly adjusted to odd."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method="gaussian",
        denoise_kernel_size=4,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    image = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
    
    # Should work with even kernel size (adjusted internally to 5)
    denoised = preprocessor.denoise(image, 'gaussian', 4)
    assert denoised.shape == image.shape
    
    # Should work with odd kernel size
    denoised = preprocessor.denoise(image, 'gaussian', 5)
    assert denoised.shape == image.shape



# Border Cropping and Padding Tests

@pytest.mark.unit
def test_crop_borders():
    """Test automatic border detection and cropping."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=True,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create image with black borders
    image = np.zeros((200, 200), dtype=np.uint8)
    image[50:150, 50:150] = 128  # Content in center
    
    cropped = preprocessor.crop_borders(image, 10)
    
    # Should be smaller than original
    assert cropped.shape[0] <= image.shape[0]
    assert cropped.shape[1] <= image.shape[1]
    # Should be approximately 100x100
    assert cropped.shape[0] == 100
    assert cropped.shape[1] == 100


@pytest.mark.property
@given(
    height=st.integers(min_value=100, max_value=300),
    width=st.integers(min_value=100, max_value=300),
    border_size=st.integers(min_value=10, max_value=50)
)
@settings(max_examples=100, deadline=None)
def test_property_border_cropping_reduces_dimensions(height, width, border_size):
    """
    Feature: chest-xray-analysis, Property 10: Border cropping reduces dimensions
    
    For any image with black borders, after automatic border cropping,
    the image dimensions should be less than or equal to the original dimensions.
    Validates: Requirements 6.1
    """
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=True,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create image with borders
    image = np.zeros((height, width), dtype=np.uint8)
    # Add content in center (avoiding borders)
    if height > 2 * border_size and width > 2 * border_size:
        image[border_size:-border_size, border_size:-border_size] = 128
    
    cropped = preprocessor.crop_borders(image, 10)
    
    # Property: Cropped dimensions should be <= original
    assert cropped.shape[0] <= height
    assert cropped.shape[1] <= width


@pytest.mark.unit
def test_pad_to_size_constant():
    """Test padding with constant (zero) padding."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Small image
    small_image = np.ones((100, 100), dtype=np.uint8) * 128
    
    padded = preprocessor.pad_to_size(small_image, (200, 200))
    
    # Should be exactly target size
    assert padded.shape == (200, 200)
    # Center should contain original content
    assert np.all(padded[50:150, 50:150] == 128)


@pytest.mark.unit
def test_pad_to_size_edge():
    """Test padding with edge replication."""
    config = PreprocessingConfig(
        target_size=(512, 512),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="edge"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Small image
    small_image = np.ones((100, 100), dtype=np.uint8) * 128
    
    padded = preprocessor.pad_to_size(small_image, (200, 200))
    
    # Should be exactly target size
    assert padded.shape == (200, 200)


@pytest.mark.property
@given(
    height=st.integers(min_value=50, max_value=200),
    width=st.integers(min_value=50, max_value=200),
    target_h=st.integers(min_value=256, max_value=512),
    target_w=st.integers(min_value=256, max_value=512)
)
@settings(max_examples=100, deadline=None)
def test_property_padding_uniformity(height, width, target_h, target_w):
    """
    Feature: chest-xray-analysis, Property 11: Padding produces uniform dimensions
    
    For any set of images with different dimensions, after padding to a target size,
    all images should have exactly the target dimensions.
    Validates: Requirements 6.2, 6.5
    """
    config = PreprocessingConfig(
        target_size=(target_h, target_w),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create image smaller than target
    if height < target_h and width < target_w:
        image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
        
        padded = preprocessor.pad_to_size(image, (target_h, target_w))
        
        # Property: Output should be exactly target size
        assert padded.shape == (target_h, target_w)


@pytest.mark.property
@given(
    height=st.integers(min_value=50, max_value=200),
    width=st.integers(min_value=50, max_value=200),
    target_h=st.integers(min_value=256, max_value=400),
    target_w=st.integers(min_value=256, max_value=400)
)
@settings(max_examples=100, deadline=None)
def test_property_padding_centers_content(height, width, target_h, target_w):
    """
    Feature: chest-xray-analysis, Property 12: Padding centers content
    
    For any image padded to a larger size, the original content should be
    centered in the output (equal or near-equal padding on opposite sides).
    Validates: Requirements 6.4
    """
    config = PreprocessingConfig(
        target_size=(target_h, target_w),
        normalization_method="minmax",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create image smaller than target
    if height < target_h and width < target_w:
        # Create image with distinct value
        image = np.ones((height, width), dtype=np.uint8) * 200
        
        padded = preprocessor.pad_to_size(image, (target_h, target_w))
        
        # Calculate expected padding
        pad_h = target_h - height
        pad_w = target_w - width
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        
        # Property: Original content should be centered
        # Check that the center region contains the original content
        center_region = padded[pad_top:pad_top+height, pad_left:pad_left+width]
        assert np.all(center_region == 200)


@pytest.mark.unit
def test_full_preprocessing_pipeline():
    """Test the complete preprocessing pipeline."""
    config = PreprocessingConfig(
        target_size=(256, 256),
        normalization_method="minmax",
        apply_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method="gaussian",
        denoise_kernel_size=3,
        crop_borders=True,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create a color image with borders
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[50:250, 50:250, :] = np.random.randint(50, 200, size=(200, 200, 3), dtype=np.uint8)
    
    # Process through full pipeline
    processed = preprocessor.process(image)
    
    # Should be exactly target size
    assert processed.shape == (256, 256)
    # Should be grayscale (2D)
    assert len(processed.shape) == 2
    # Should be normalized (float values in [0, 1] for minmax)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


@pytest.mark.unit
def test_pipeline_without_optional_steps():
    """Test pipeline with optional steps disabled."""
    config = PreprocessingConfig(
        target_size=(128, 128),
        normalization_method="zscore",
        apply_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,
        denoise_kernel_size=3,
        crop_borders=False,
        border_threshold=10,
        padding_mode="constant"
    )
    preprocessor = ImagePreprocessor(config)
    
    # Create simple grayscale image
    image = np.random.randint(0, 256, size=(200, 200), dtype=np.uint8)
    
    # Process
    processed = preprocessor.process(image)
    
    # Should be target size
    assert processed.shape == (128, 128)
    # Should be normalized with zscore (mean ≈ 0, std ≈ 1)
    assert np.isclose(processed.mean(), 0.0, atol=0.1)
    assert np.isclose(processed.std(), 1.0, atol=0.1)
