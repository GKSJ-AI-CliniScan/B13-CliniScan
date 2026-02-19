# Implementation Plan: Chest X-ray Analysis System

## Overview

This implementation plan breaks down the chest X-ray analysis system into discrete, manageable tasks. The system will be built incrementally, starting with core data processing capabilities, then adding model training and evaluation features, and finally visualization tools. Each task builds on previous work, with checkpoints to ensure quality and correctness.

The implementation uses Python with PyTorch for deep learning, OpenCV for image processing, and hypothesis for property-based testing.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create Python package structure with modules for preprocessing, models, evaluation, and visualization
  - Set up pyproject.toml or requirements.txt with all dependencies (pydicom, opencv-python, torch, torchvision, albumentations, hypothesis, pytest, scikit-learn, pytorch-grad-cam, ultralytics)
  - Create configuration management system using YAML or JSON
  - Set up pytest configuration and test directory structure
  - _Requirements: 15.1, 15.2_

- [x] 2. Implement DICOM loading and conversion
  - [x] 2.1 Create DICOMLoader class with load_dicom method
    - Implement DICOM file loading using pydicom
    - Extract pixel data and metadata (patient_id, study_id, bits_stored, etc.)
    - Handle both single-frame and multi-frame DICOM files
    - _Requirements: 1.1, 1.3_

  - [x] 2.2 Write property test for DICOM loading
    - **Property 2: Multi-frame extraction count**
    - **Validates: Requirements 1.3**

  - [x] 2.3 Write property test for invalid DICOM handling
    - **Property 3: Invalid DICOM error handling**
    - **Validates: Requirements 1.4**

  - [x] 2.4 Implement convert_to_png method
    - Convert DICOM pixel data to PNG format
    - Preserve bit-depth information in metadata
    - Handle different photometric interpretations
    - _Requirements: 1.2, 1.5_

  - [x] 2.5 Write property test for DICOM-PNG round-trip
    - **Property 1: DICOM to PNG round-trip preserves image data**
    - **Validates: Requirements 1.2, 1.5**

- [x] 3. Implement core image preprocessing
  - [x] 3.1 Create ImagePreprocessor class with configuration support
    - Implement PreprocessingConfig dataclass
    - Initialize preprocessor with configuration
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

  - [x] 3.2 Implement grayscale conversion
    - Convert color images to grayscale using standard luminance weights
    - Preserve grayscale images without modification
    - _Requirements: 2.1, 2.2_

  - [x] 3.3 Write property test for grayscale idempotence
    - **Property 4: Grayscale conversion idempotence**
    - **Validates: Requirements 2.2**

  - [x] 3.4 Implement resize with aspect ratio preservation
    - Resize images to target dimensions
    - Maintain aspect ratio when configured
    - Use high-quality interpolation (INTER_CUBIC for upsampling, INTER_AREA for downsampling)
    - _Requirements: 2.3, 2.4, 2.5_

  - [x] 3.5 Write property test for aspect ratio preservation
    - **Property 5: Aspect ratio preservation during resize**
    - **Validates: Requirements 2.3**

- [x] 4. Implement intensity normalization
  - [x] 4.1 Implement min-max normalization
    - Scale pixel values to [0, 1] range
    - Handle edge cases (constant images)
    - _Requirements: 3.1_

  - [x] 4.2 Write property test for min-max bounds
    - **Property 6: Min-max normalization bounds**
    - **Validates: Requirements 3.1**

  - [x] 4.3 Implement z-score normalization
    - Standardize to zero mean and unit variance
    - Handle zero standard deviation case
    - Support per-image and dataset-level normalization
    - _Requirements: 3.2, 3.3, 3.4_

  - [x] 4.4 Write property test for z-score statistics
    - **Property 7: Z-score normalization statistics**
    - **Validates: Requirements 3.2**

  - [x] 4.5 Write property test for normalization monotonicity
    - **Property 8: Normalization preserves monotonicity**
    - **Validates: Requirements 3.5**

- [x] 5. Implement contrast enhancement and denoising
  - [x] 5.1 Implement CLAHE contrast enhancement
    - Apply CLAHE with configurable clip limit and tile size
    - Use safe defaults for invalid parameters
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [x] 5.2 Implement denoising methods
    - Implement Gaussian blur with configurable kernel size
    - Implement median filtering
    - Adjust even kernel sizes to nearest odd number
    - _Requirements: 5.1, 5.2, 5.4, 5.5_

  - [x] 5.3 Write property test for kernel size adjustment
    - **Property 9: Even kernel sizes adjusted to odd**
    - **Validates: Requirements 5.5**

- [x] 6. Implement border cropping and padding
  - [x] 6.1 Implement automatic border detection and cropping
    - Detect black borders using threshold
    - Crop borders automatically
    - _Requirements: 6.1_

  - [x] 6.2 Write property test for border cropping
    - **Property 10: Border cropping reduces dimensions**
    - **Validates: Requirements 6.1**

  - [x] 6.3 Implement padding to uniform dimensions
    - Pad images to target size
    - Support zero-padding and edge-replication
    - Center original content in padded frame
    - _Requirements: 6.2, 6.3, 6.4, 6.5_

  - [x] 6.4 Write property test for padding uniformity
    - **Property 11: Padding produces uniform dimensions**
    - **Validates: Requirements 6.2, 6.5**

  - [x] 6.5 Write property test for padding centering
    - **Property 12: Padding centers content**
    - **Validates: Requirements 6.4**

  - [x] 6.6 Implement complete preprocessing pipeline
    - Chain all preprocessing operations in process() method
    - Apply operations in correct order
    - _Requirements: All preprocessing requirements_

- [x] 7. Checkpoint - Ensure preprocessing tests pass
  - All 38 preprocessing tests passing (11 DICOM + 27 image preprocessing)
  - 94% coverage on dicom_loader.py, 90% coverage on image_preprocessor.py
  - 11 property-based tests with 100 examples each

- [ ] 8. Implement annotation format conversion
  - [ ] 8.1 Create AnnotationConverter class
    - Implement BoundingBox dataclass with conversion methods
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ] 8.2 Implement YOLO format parser
    - Parse YOLO format annotations (class x_center y_center width height)
    - Validate normalized coordinates [0, 1]
    - _Requirements: 7.1, 7.4_

  - [ ] 8.3 Write property test for YOLO coordinate normalization
    - **Property 14: YOLO coordinate normalization**
    - **Validates: Requirements 7.4**

  - [ ] 8.4 Implement COCO format parser
    - Parse COCO JSON structure
    - Extract bounding boxes and class labels
    - _Requirements: 7.2_

  - [ ] 8.5 Implement bidirectional conversion
    - Convert YOLO to COCO format
    - Convert COCO to YOLO format
    - _Requirements: 7.3, 7.4_

  - [ ] 8.6 Write property test for annotation round-trip
    - **Property 13: Annotation format round-trip**
    - **Validates: Requirements 7.3**

  - [ ] 8.7 Write property test for malformed annotation handling
    - **Property 15: Malformed annotation error handling**
    - **Validates: Requirements 7.5**

- [x] 9. Implement data augmentation
  - [x] 9.1 Create MedicalAugmentor class with configuration
    - Implement AugmentationConfig dataclass
    - Initialize augmentor with safe parameters for medical images
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 9.2 Implement safe augmentation methods
    - Implement horizontal flip (no vertical flip)
    - Implement small rotations (±10 degrees)
    - Implement contrast adjustment within safe ranges
    - Implement mild zoom
    - Implement mild elastic deformation
    - Transform bounding boxes along with images
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [x] 9.3 Write property test for horizontal flip dimensions
    - **Property 16: Horizontal flip preserves image dimensions**
    - **Validates: Requirements 8.1**

  - [x] 9.4 Write property test for rotation bounds
    - **Property 17: Rotation stays within safe bounds**
    - **Validates: Requirements 8.2, 8.7**

  - [x] 9.5 Write property test for no vertical flips
    - **Property 18: No vertical flips applied**
    - **Validates: Requirements 8.6**

  - [x] 9.6 Write property test for bounding box transformation
    - **Property 19: Bounding box transformation consistency**
    - **Validates: Requirements 8.8**

- [-] 10. Implement dataset management
  - [x] 10.1 Create DatasetManager class
    - Implement DatasetSplit dataclass
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 10.2 Implement dataset splitting
    - Split dataset into train/val/test with configurable ratios
    - Support stratified splitting for class balance
    - Support patient-level splitting to prevent leakage
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 10.3 Write property test for split completeness
    - **Property 20: Dataset split completeness and disjointness**
    - **Validates: Requirements 9.1, 9.3**

  - [x] 10.4 Write property test for split ratios
    - **Property 21: Split ratios respected**
    - **Validates: Requirements 9.2**

  - [x] 10.5 Write property test for stratified splitting
    - **Property 22: Stratified split maintains class distribution**
    - **Validates: Requirements 9.4**

  - [x] 10.6 Write property test for patient-level splitting
    - **Property 23: Patient-level split prevents leakage**
    - **Validates: Requirements 9.5**

  - [ ] 10.7 Implement PyTorch Dataset and DataLoader creation
    - Create custom Dataset class for chest X-rays
    - Implement data loading with augmentation
    - Create DataLoader instances for train/val/test
    - _Requirements: 9.1_

- [ ] 11. Checkpoint - Ensure data pipeline tests pass
  - Ensure all data processing and augmentation tests pass, ask the user if questions arise.

- [ ] 12. Implement classification model
  - [ ] 12.1 Create ChestXrayClassifier class
    - Support ResNet and EfficientNet architectures
    - Load pre-trained weights from ImageNet
    - Implement TrainingConfig dataclass
    - _Requirements: 10.1, 10.2_

  - [ ] 12.2 Implement training loop
    - Configure optimizer (Adam, SGD, AdamW)
    - Select appropriate loss function (BCE for binary, CE for multi-class)
    - Implement learning rate scheduling
    - Implement early stopping based on validation metrics
    - Save model checkpoints and training history
    - _Requirements: 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ] 12.3 Write property test for loss function selection
    - **Property 24: Loss function selection**
    - **Validates: Requirements 10.2**

  - [ ] 12.4 Write property test for learning rate scheduling
    - **Property 25: Learning rate scheduling**
    - **Validates: Requirements 10.4**

  - [ ] 12.5 Write property test for model save/load round-trip
    - **Property 26: Model save/load round-trip**
    - **Validates: Requirements 10.5**

  - [ ] 12.6 Write property test for early stopping
    - **Property 27: Early stopping triggers**
    - **Validates: Requirements 10.6**

  - [ ] 12.7 Implement prediction and evaluation methods
    - Implement predict() for single image inference
    - Implement evaluate() for test set evaluation
    - _Requirements: 10.1_

- [ ] 13. Implement detection model
  - [ ] 13.1 Create ChestXrayDetector class
    - Support YOLOv8 architecture (via ultralytics)
    - Support Faster R-CNN architecture (via torchvision)
    - Implement Detection dataclass
    - _Requirements: 11.1, 11.2_

  - [ ] 13.2 Implement detection training
    - Configure detection-specific loss functions
    - Handle bounding box coordinate normalization per architecture
    - Save model weights and configuration
    - _Requirements: 11.3, 11.4, 11.5_

  - [ ] 13.3 Write property test for bbox coordinate normalization
    - **Property 28: Bounding box coordinate normalization**
    - **Validates: Requirements 11.4**

  - [ ] 13.4 Implement detection prediction and evaluation
    - Implement predict() for bounding box detection
    - Implement evaluate() for detection metrics
    - _Requirements: 11.1, 11.2_

- [ ] 14. Implement metrics calculation
  - [ ] 14.1 Create MetricsCalculator class
    - Implement ClassificationMetrics dataclass
    - Implement DetectionMetrics dataclass
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

  - [ ] 14.2 Implement classification metrics
    - Compute AUC-ROC using sklearn
    - Compute F1, precision, recall
    - Generate confusion matrix
    - Support per-class and overall metrics
    - _Requirements: 12.1, 12.2, 12.5, 12.6_

  - [ ] 14.3 Implement detection metrics
    - Implement IoU calculation for bounding boxes
    - Compute mAP at different IoU thresholds (0.5, 0.75, 0.5:0.95)
    - Compute per-class Average Precision
    - _Requirements: 12.3, 12.4, 12.5_

  - [ ] 14.4 Write property test for IoU symmetry
    - **Property 29: IoU symmetry and bounds**
    - **Validates: Requirements 12.4**

  - [ ] 14.5 Write property test for metrics consistency
    - **Property 30: Per-class and overall metrics consistency**
    - **Validates: Requirements 12.5**

  - [ ] 14.6 Write unit tests for metric computation
    - Test AUC, F1, precision, recall with known values
    - Test mAP computation with known detections
    - Test confusion matrix generation
    - _Requirements: 12.1, 12.2, 12.3, 12.6_

- [ ] 15. Checkpoint - Ensure model training and evaluation work
  - Ensure all model and metrics tests pass, ask the user if questions arise.

- [ ] 16. Implement Grad-CAM visualization
  - [ ] 16.1 Create Visualizer class
    - Set up pytorch-grad-cam integration
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

  - [ ] 16.2 Implement Grad-CAM generation
    - Generate Grad-CAM heatmaps for classification predictions
    - Support target layer selection
    - Use appropriate colormaps (jet, hot) for medical imaging
    - _Requirements: 13.1, 13.3, 13.4_

  - [ ] 16.3 Write property test for Grad-CAM generation
    - **Property 31: Grad-CAM heatmap generation**
    - **Validates: Requirements 13.1**

  - [ ] 16.4 Write property test for target layer selection
    - **Property 33: Target layer selection affects heatmap**
    - **Validates: Requirements 13.3**

  - [ ] 16.5 Implement heatmap overlay
    - Overlay heatmaps on original images with configurable transparency
    - Save visualization outputs in PNG format
    - _Requirements: 13.2, 13.5_

  - [ ] 16.6 Write property test for heatmap transparency
    - **Property 32: Heatmap overlay transparency**
    - **Validates: Requirements 13.2**

- [ ] 17. Implement bounding box visualization
  - [ ] 17.1 Implement bounding box drawing
    - Draw bounding boxes on images
    - Display class labels and confidence scores
    - Use distinct colors for different classes
    - Support visualization of predictions and ground truth
    - Ensure all boxes are visible when overlapping
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [ ] 17.2 Write property test for all boxes drawn
    - **Property 34: All bounding boxes drawn**
    - **Validates: Requirements 14.1, 14.5**

  - [ ] 17.3 Write property test for class-specific colors
    - **Property 35: Class-specific colors**
    - **Validates: Requirements 14.3**

  - [ ] 17.4 Write property test for prediction and ground truth visualization
    - **Property 36: Prediction and ground truth visualization**
    - **Validates: Requirements 14.4**

- [ ] 18. Implement configuration management
  - [ ] 18.1 Create configuration validation
    - Validate all configuration parameters before use
    - Return descriptive errors for invalid configurations
    - _Requirements: 15.2_

  - [ ] 18.2 Write property test for configuration validation
    - **Property 37: Configuration validation**
    - **Validates: Requirements 15.2**

  - [ ] 18.2 Implement configuration logging
    - Log all hyperparameters and settings for reproducibility
    - Save configuration alongside model checkpoints
    - _Requirements: 15.5_

  - [ ] 18.3 Write property test for configuration logging
    - **Property 38: Configuration logging completeness**
    - **Validates: Requirements 15.5**

- [ ] 19. Create example scripts and documentation
  - [ ] 19.1 Create example training script
    - Demonstrate full training pipeline for classification
    - Demonstrate full training pipeline for detection
    - Include configuration examples
    - _Requirements: 10.1, 11.1_

  - [ ] 19.2 Create example inference script
    - Demonstrate loading trained models
    - Demonstrate making predictions on new images
    - Demonstrate generating visualizations
    - _Requirements: 13.1, 14.1_

  - [ ] 19.3 Write README documentation
    - Document installation and setup
    - Document usage examples
    - Document configuration options
    - _Requirements: All requirements_

- [ ] 20. Final checkpoint - Integration testing
  - Run end-to-end tests with sample DICOM files
  - Verify complete preprocessing pipeline
  - Verify training and evaluation workflows
  - Verify visualization outputs
  - Ensure all property tests pass with 100+ iterations
  - Ask the user if questions arise.

## Notes

- Each property test should run with minimum 100 iterations
- Property tests use hypothesis library for random input generation
- Unit tests focus on specific examples and edge cases
- Checkpoints ensure incremental validation and quality
- All code should follow PEP 8 style guidelines
- Use type hints throughout for better code quality
