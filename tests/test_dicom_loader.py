"""
Tests for DICOM loading and conversion.

Feature: chest-xray-analysis
"""

import pytest
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck
from chest_xray_analysis.preprocessing import DICOMLoader, DICOMImage
from chest_xray_analysis.utils.exceptions import DICOMLoadError


def create_test_dicom(
    filepath: Path,
    rows: int = 512,
    cols: int = 512,
    num_frames: int = 1,
    bits_stored: int = 8,
    patient_id: str = "TEST001"
) -> None:
    """Helper function to create a test DICOM file."""
    # Create a minimal DICOM dataset
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = '1.2.3.4'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'  # Implicit VR Little Endian
    
    ds = FileDataset(
        str(filepath),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128
    )
    
    # Add required DICOM tags
    ds.PatientID = patient_id
    ds.StudyInstanceUID = '1.2.3'
    ds.SeriesInstanceUID = '1.2.3.4'
    ds.SOPInstanceUID = '1.2.3.4.5'
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    ds.Modality = 'CR'  # Computed Radiography
    ds.StudyDate = '20240101'
    ds.SeriesDate = '20240101'
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsStored = bits_stored
    ds.BitsAllocated = 8 if bits_stored <= 8 else 16
    ds.HighBit = bits_stored - 1
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.SamplesPerPixel = 1
    
    # Create pixel data
    if num_frames > 1:
        ds.NumberOfFrames = num_frames
        if bits_stored <= 8:
            pixel_array = np.random.randint(0, 256, size=(num_frames, rows, cols), dtype=np.uint8)
        else:
            pixel_array = np.random.randint(0, 2**bits_stored, size=(num_frames, rows, cols), dtype=np.uint16)
    else:
        if bits_stored <= 8:
            pixel_array = np.random.randint(0, 256, size=(rows, cols), dtype=np.uint8)
        else:
            pixel_array = np.random.randint(0, 2**bits_stored, size=(rows, cols), dtype=np.uint16)
    
    ds.PixelData = pixel_array.tobytes()
    
    # Save DICOM file
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(filepath), write_like_original=False)


# Unit Tests

@pytest.mark.unit
def test_load_single_frame_dicom(temp_dir):
    """Test loading a single-frame DICOM file."""
    dicom_path = temp_dir / "test.dcm"
    create_test_dicom(dicom_path, rows=256, cols=256, num_frames=1)
    
    loader = DICOMLoader()
    dicom_image = loader.load_dicom(str(dicom_path))
    
    assert isinstance(dicom_image, DICOMImage)
    assert dicom_image.pixel_array.shape == (256, 256)
    assert dicom_image.patient_id == "TEST001"
    assert dicom_image.modality == "CR"


@pytest.mark.unit
def test_load_multi_frame_dicom(temp_dir):
    """Test loading a multi-frame DICOM file."""
    dicom_path = temp_dir / "test_multi.dcm"
    create_test_dicom(dicom_path, rows=128, cols=128, num_frames=5)
    
    loader = DICOMLoader()
    dicom_image = loader.load_dicom(str(dicom_path))
    
    assert isinstance(dicom_image, DICOMImage)
    assert dicom_image.pixel_array.shape == (5, 128, 128)
    assert dicom_image.metadata['NumberOfFrames'] == 5


@pytest.mark.unit
def test_load_invalid_dicom(temp_dir):
    """Test that loading an invalid DICOM file raises an error."""
    invalid_path = temp_dir / "invalid.dcm"
    invalid_path.write_text("This is not a DICOM file")
    
    loader = DICOMLoader()
    with pytest.raises(DICOMLoadError):
        loader.load_dicom(str(invalid_path))


@pytest.mark.unit
def test_load_nonexistent_file():
    """Test that loading a nonexistent file raises an error."""
    loader = DICOMLoader()
    with pytest.raises(DICOMLoadError):
        loader.load_dicom("nonexistent_file.dcm")


# Property-Based Tests

@pytest.mark.property
@given(
    num_frames=st.integers(min_value=1, max_value=10),
    rows=st.integers(min_value=64, max_value=512),
    cols=st.integers(min_value=64, max_value=512)
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_multi_frame_extraction_count(temp_dir, num_frames, rows, cols):
    """
    Feature: chest-xray-analysis, Property 2: Multi-frame extraction count
    
    For any DICOM file with N frames, extracting frames should produce exactly N separate images.
    Validates: Requirements 1.3
    """
    dicom_path = temp_dir / f"test_{num_frames}_{rows}_{cols}.dcm"
    create_test_dicom(dicom_path, rows=rows, cols=cols, num_frames=num_frames)
    
    loader = DICOMLoader()
    dicom_image = loader.load_dicom(str(dicom_path))
    frames = loader.extract_frames(dicom_image)
    
    # Property: Number of extracted frames equals number of frames in DICOM
    assert len(frames) == num_frames
    
    # Each frame should have the correct dimensions
    for frame in frames:
        assert frame.shape == (rows, cols)


@pytest.mark.property
def test_property_invalid_dicom_error_handling(temp_dir):
    """
    Feature: chest-xray-analysis, Property 3: Invalid DICOM error handling
    
    For any corrupted or malformed DICOM file, the loader should return an error
    rather than crash or produce invalid output.
    Validates: Requirements 1.4
    """
    # Test various invalid file contents
    invalid_contents = [
        b"",  # Empty file
        b"Not a DICOM file",  # Plain text
        b"\x00" * 100,  # Null bytes
        b"DICM" + b"\x00" * 100,  # Partial DICOM header
    ]
    
    loader = DICOMLoader()
    
    for i, content in enumerate(invalid_contents):
        invalid_path = temp_dir / f"invalid_{i}.dcm"
        invalid_path.write_bytes(content)
        
        # Property: Invalid files should raise DICOMLoadError, not crash
        with pytest.raises(DICOMLoadError):
            loader.load_dicom(str(invalid_path))


@pytest.mark.unit
def test_convert_to_png_single_frame(temp_dir):
    """Test converting single-frame DICOM to PNG."""
    dicom_path = temp_dir / "test.dcm"
    png_path = temp_dir / "output.png"
    create_test_dicom(dicom_path, rows=128, cols=128, num_frames=1)
    
    loader = DICOMLoader()
    dicom_image = loader.load_dicom(str(dicom_path))
    loader.convert_to_png(dicom_image, str(png_path))
    
    assert png_path.exists()
    
    # Load PNG and verify it's valid
    from PIL import Image
    img = Image.open(png_path)
    assert img.size == (128, 128)


@pytest.mark.unit
def test_convert_to_png_multi_frame(temp_dir):
    """Test converting multi-frame DICOM to PNG (should use first frame)."""
    dicom_path = temp_dir / "test_multi.dcm"
    png_path = temp_dir / "output_multi.png"
    create_test_dicom(dicom_path, rows=128, cols=128, num_frames=3)
    
    loader = DICOMLoader()
    dicom_image = loader.load_dicom(str(dicom_path))
    loader.convert_to_png(dicom_image, str(png_path))
    
    assert png_path.exists()
    
    # Load PNG and verify it's valid
    from PIL import Image
    img = Image.open(png_path)
    assert img.size == (128, 128)


@pytest.mark.property
@given(
    bits_stored=st.sampled_from([8, 12, 16]),
    rows=st.integers(min_value=64, max_value=256),
    cols=st.integers(min_value=64, max_value=256)
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_dicom_png_round_trip_preserves_data(temp_dir, bits_stored, rows, cols):
    """
    Feature: chest-xray-analysis, Property 1: DICOM to PNG round-trip preserves image data
    
    For any valid DICOM image, converting to PNG and loading back should preserve
    the essential pixel data within acceptable tolerance.
    Validates: Requirements 1.2, 1.5
    """
    dicom_path = temp_dir / f"test_{bits_stored}_{rows}_{cols}.dcm"
    png_path = temp_dir / f"output_{bits_stored}_{rows}_{cols}.png"
    
    create_test_dicom(dicom_path, rows=rows, cols=cols, num_frames=1, bits_stored=bits_stored)
    
    loader = DICOMLoader()
    dicom_image = loader.load_dicom(str(dicom_path))
    original_array = dicom_image.pixel_array.copy()
    
    # Convert to PNG
    loader.convert_to_png(dicom_image, str(png_path))
    
    # Load PNG back
    from PIL import Image
    png_image = Image.open(png_path)
    png_array = np.array(png_image)
    
    # Property: Dimensions should be preserved
    assert png_array.shape == original_array.shape
    
    # Property: Pixel data should be preserved (accounting for 8-bit conversion)
    # Normalize both to 0-1 range for comparison
    if bits_stored > 8:
        original_normalized = (original_array - original_array.min()) / (original_array.max() - original_array.min() + 1e-8)
    else:
        original_normalized = original_array / 255.0
    
    png_normalized = png_array / 255.0
    
    # Allow small tolerance due to quantization
    assert np.allclose(original_normalized, png_normalized, atol=0.01)


@pytest.mark.unit
def test_extract_frames_single_frame(temp_dir):
    """Test extracting frames from single-frame DICOM."""
    dicom_path = temp_dir / "test_single.dcm"
    create_test_dicom(dicom_path, rows=128, cols=128, num_frames=1)
    
    loader = DICOMLoader()
    dicom_image = loader.load_dicom(str(dicom_path))
    frames = loader.extract_frames(dicom_image)
    
    assert len(frames) == 1
    assert frames[0].shape == (128, 128)


@pytest.mark.unit
def test_extract_frames_multi_frame(temp_dir):
    """Test extracting frames from multi-frame DICOM."""
    dicom_path = temp_dir / "test_multi.dcm"
    create_test_dicom(dicom_path, rows=128, cols=128, num_frames=4)
    
    loader = DICOMLoader()
    dicom_image = loader.load_dicom(str(dicom_path))
    frames = loader.extract_frames(dicom_image)
    
    assert len(frames) == 4
    for frame in frames:
        assert frame.shape == (128, 128)
