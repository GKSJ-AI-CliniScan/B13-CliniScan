"""
DICOM file loading and conversion.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import pydicom
from PIL import Image
from chest_xray_analysis.utils.exceptions import DICOMLoadError


@dataclass
class DICOMImage:
    """Container for DICOM image data and metadata."""
    pixel_array: np.ndarray
    metadata: Dict[str, Any]
    patient_id: str
    study_id: str
    series_id: str
    modality: str
    bits_stored: int
    photometric_interpretation: str


class DICOMLoader:
    """Load and convert DICOM medical images."""
    
    def load_dicom(self, filepath: str) -> DICOMImage:
        """
        Load DICOM file and extract pixel data and metadata.
        
        Args:
            filepath: Path to DICOM file
            
        Returns:
            DICOMImage containing pixel data and metadata
            
        Raises:
            DICOMLoadError: If file cannot be loaded or is invalid
        """
        try:
            # Read DICOM file
            dcm = pydicom.dcmread(filepath)
            
            # Extract pixel array
            pixel_array = dcm.pixel_array
            
            # Extract metadata
            metadata = {
                'SOPInstanceUID': str(getattr(dcm, 'SOPInstanceUID', '')),
                'StudyDate': str(getattr(dcm, 'StudyDate', '')),
                'SeriesDate': str(getattr(dcm, 'SeriesDate', '')),
                'Rows': int(getattr(dcm, 'Rows', 0)),
                'Columns': int(getattr(dcm, 'Columns', 0)),
                'PixelSpacing': getattr(dcm, 'PixelSpacing', None),
                'SliceThickness': getattr(dcm, 'SliceThickness', None),
                'NumberOfFrames': int(getattr(dcm, 'NumberOfFrames', 1)),
            }
            
            # Create DICOMImage object
            dicom_image = DICOMImage(
                pixel_array=pixel_array,
                metadata=metadata,
                patient_id=str(getattr(dcm, 'PatientID', 'UNKNOWN')),
                study_id=str(getattr(dcm, 'StudyInstanceUID', 'UNKNOWN')),
                series_id=str(getattr(dcm, 'SeriesInstanceUID', 'UNKNOWN')),
                modality=str(getattr(dcm, 'Modality', 'UNKNOWN')),
                bits_stored=int(getattr(dcm, 'BitsStored', 8)),
                photometric_interpretation=str(getattr(dcm, 'PhotometricInterpretation', 'UNKNOWN'))
            )
            
            return dicom_image
            
        except pydicom.errors.InvalidDicomError as e:
            raise DICOMLoadError(f"Invalid DICOM file: {filepath}. Error: {str(e)}")
        except Exception as e:
            raise DICOMLoadError(f"Failed to load DICOM file: {filepath}. Error: {str(e)}")
    
    def convert_to_png(self, dicom_image: DICOMImage, output_path: str) -> None:
        """
        Convert DICOM pixel data to PNG format.
        
        Args:
            dicom_image: DICOMImage object
            output_path: Path to save PNG file
            
        Raises:
            DICOMLoadError: If conversion fails
        """
        try:
            # Get pixel array
            pixel_array = dicom_image.pixel_array
            
            # Handle multi-frame DICOM - use first frame
            if len(pixel_array.shape) == 3:
                pixel_array = pixel_array[0]
            
            # Normalize to 8-bit range if needed
            if dicom_image.bits_stored > 8:
                # Scale to 0-255 range
                pixel_min = pixel_array.min()
                pixel_max = pixel_array.max()
                if pixel_max > pixel_min:
                    pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
                else:
                    pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
            else:
                pixel_array = pixel_array.astype(np.uint8)
            
            # Create output directory if needed
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as PNG
            image = Image.fromarray(pixel_array)
            image.save(output_path)
            
        except Exception as e:
            raise DICOMLoadError(f"Failed to convert DICOM to PNG: {str(e)}")
    
    def extract_frames(self, dicom_image: DICOMImage) -> List[np.ndarray]:
        """
        Extract individual frames from multi-frame DICOM.
        
        Args:
            dicom_image: DICOMImage object
            
        Returns:
            List of frame arrays
        """
        pixel_array = dicom_image.pixel_array
        
        # Check if multi-frame
        if len(pixel_array.shape) == 3:
            # Multi-frame: shape is (frames, height, width)
            return [pixel_array[i] for i in range(pixel_array.shape[0])]
        else:
            # Single frame: shape is (height, width)
            return [pixel_array]
