# Chest X-ray Analysis System

A comprehensive medical imaging pipeline for chest X-ray preprocessing, deep learning-based classification and detection, and visualization of model predictions.

## Features

- **DICOM Processing**: Load and convert DICOM medical images to standard formats
- **Preprocessing Pipeline**: Standardized image transformations including:
  - Grayscale conversion
  - Resizing with aspect ratio preservation
  - Intensity normalization (min-max, z-score)
  - CLAHE contrast enhancement
  - Denoising (Gaussian blur, median filtering)
  - Border cropping and padding
- **Annotation Management**: Convert between YOLO and COCO annotation formats
- **Safe Augmentation**: Medical image-appropriate augmentations
- **Model Training**: Support for classification (ResNet, EfficientNet) and detection (YOLOv8, Faster R-CNN)
- **Evaluation**: Comprehensive metrics (AUC, F1, mAP, IoU)
- **Visualization**: Grad-CAM heatmaps and bounding box overlays

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd chest-xray-analysis

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from chest_xray_analysis import (
    DICOMLoader,
    ImagePreprocessor,
    PreprocessingConfig,
    ChestXrayClassifier,
)

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

# Train classifier
classifier = ChestXrayClassifier(
    architecture="resnet50",
    num_classes=2,
    pretrained=True,
)
# Training code here...
```

## Project Structure

```
chest-xray-analysis/
├── chest_xray_analysis/
│   ├── preprocessing/      # DICOM loading and image preprocessing
│   ├── data/              # Annotations, augmentation, dataset management
│   ├── models/            # Classification and detection models
│   ├── evaluation/        # Metrics calculation
│   ├── visualization/     # Grad-CAM and bounding box visualization
│   └── utils/             # Configuration and utilities
├── tests/                 # Test suite
├── config/                # Configuration files
└── examples/              # Example scripts
```

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

## Configuration

Configuration files are located in the `config/` directory:
- `preprocessing_default.yaml`: Preprocessing parameters
- `augmentation_default.yaml`: Augmentation settings
- `training_default.yaml`: Training hyperparameters

## License

MIT License

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting a pull request.
