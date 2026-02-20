# CliniScan: Chest X-ray Analysis System

A comprehensive medical imaging pipeline for chest X-ray preprocessing, deep learning-based classification and detection, and visualization of model predictions.

## Features

- **DICOM Processing**: Load and convert DICOM medical images to standard formats
- **Preprocessing Pipeline**: Standardized image transformations including:
  - Photometric correction (MONOCHROME1/MONOCHROME2 handling)
  - Grayscale conversion
  - Resizing with aspect ratio preservation
  - Intensity normalization (min-max, z-score)
  - CLAHE contrast enhancement
  - Denoising (Gaussian blur, median filtering)
  - Border cropping and padding
- **Dataset Preprocessing**: VinDr-CXR dataset-specific pipeline with:
  - YOLO annotation conversion
  - Train/validation split management
  - Class imbalance handling
  - Batch processing with progress tracking
- **Annotation Management**: Convert between YOLO and COCO annotation formats
- **Safe Augmentation**: Medical image-appropriate augmentations
- **Model Training**: Support for classification (ResNet, EfficientNet) and detection (YOLOv8, Faster R-CNN)
- **Evaluation**: Comprehensive metrics (AUC, F1, mAP, IoU)
- **Visualization**: Grad-CAM heatmaps and bounding box overlays

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd B13-CliniScan

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Dataset Preprocessing

```python
from chest_xray_analysis.data.dataset_preprocessor import DatasetPreprocessor
from chest_xray_analysis.utils.config import PreprocessingConfig

# Initialize preprocessor
config = PreprocessingConfig(
    target_size=(640, 640),
    normalization_method="minmax",
    apply_clahe=True,
)

preprocessor = DatasetPreprocessor(
    input_dir="/path/to/vindr-cxr",
    output_dir="/path/to/processed",
    config=config,
    seed=42,
    train_split=0.8,
    exclude_classes=[14]  # Exclude "No finding" class
)

# Process entire dataset
preprocessor.process_dataset()
```

### Image Preprocessing

```python
from chest_xray_analysis.preprocessing import DICOMLoader, ImagePreprocessor, PreprocessingConfig

# Load DICOM file
loader = DICOMLoader()
dicom_image = loader.load_dicom("path/to/file.dcm")

# Preprocess image
config = PreprocessingConfig(
    target_size=(512, 512),
    normalization_method="minmax",
    apply_clahe=True,
)
preprocessor = ImagePreprocessor(config)
processed_image = preprocessor.process(dicom_image.pixel_array)
```

## Project Structure

```
B13-CliniScan/
├── chest_xray_analysis/
│   ├── preprocessing/      # DICOM loading and image preprocessing
│   ├── data/              # Dataset management and annotation handling
│   ├── utils/             # Configuration and utilities
│   ├── config/            # Configuration files
│   └── __init__.py
├── tests/                 # Test suite
├── config/                # Configuration files
├── README.md
├── LICENSE
└── pyproject.toml
```

## Configuration

Configuration files are located in the `config/` directory:
- `preprocessing_default.yaml`: Preprocessing parameters
- `augmentation_default.yaml`: Augmentation settings
- `training_default.yaml`: Training hyperparameters

## Dataset Information

### VinDr-CXR Dataset

The VinDr-CXR (VinBigData Chest X-ray) dataset contains 18,000 PA view chest X-ray scans:

- **Training**: 15,000 images (publicly labeled)
- **Test**: 3,000 images (labels withheld for competition)
- **Format**: DICOM (.dicom or .dcm)
- **Bit Depth**: High dynamic range (12-bit or 16-bit) → normalized to 8-bit
- **Classes**: 14 critical radiographic findings + 1 "No finding" class

#### Class Labels

| ID | Class Name | Count | Description |
|----|-----------|-------|-------------|
| 0 | Aortic enlargement | 7,162 | Abnormal widening of the aorta |
| 1 | Atelectasis | 279 | Collapse of lung tissue |
| 2 | Calcification | 960 | Calcium deposits in the lung |
| 3 | Cardiomegaly | 5,427 | Enlarged heart |
| 4 | Consolidation | 556 | Lung tissue filled with liquid |
| 5 | ILD | 1,000 | Interstitial Lung Disease |
| 6 | Infiltration | 1,247 | Substance denser than air in lungs |
| 7 | Lung Opacity | 2,483 | Opaque areas in the lung |
| 8 | Nodule/Mass | 2,580 | Growths or lumps in the lung |
| 9 | Other lesion | 2,203 | Lesions not fitting other categories |
| 10 | Pleural effusion | 2,476 | Fluid build-up between lung layers |
| 11 | Pleural thickening | 4,842 | Thickening of the pleura |
| 12 | Pneumothorax | 226 | Air leaking into lung-chest space |
| 13 | Pulmonary fibrosis | 4,655 | Scarring of lung tissue |
| 14 | No finding | 31,818 | No abnormalities detected (EXCLUDED) |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=chest_xray_analysis

# Run property-based tests only
pytest -m property

# Run unit tests only
pytest -m unit
```

## License

MIT License

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting a pull request.
