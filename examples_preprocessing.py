"""
Example script for preprocessing VinDr-CXR dataset.

This script demonstrates how to use the DatasetPreprocessor to convert
the raw VinDr-CXR dataset into YOLO-compatible format.
"""

from chest_xray_analysis import (
    DatasetPreprocessor,
    PreprocessingConfig,
)


def main():
    """Main preprocessing function."""
    
    # Configuration parameters
    INPUT_DIR = "/path/to/vindr-cxr"  # Path to downloaded VinDr-CXR dataset
    OUTPUT_DIR = "./processed_dataset"  # Output directory for processed data
    
    # Create preprocessing configuration
    config = PreprocessingConfig(
        target_size=(640, 640),           # Target image size for YOLO
        normalization_method="minmax",    # Normalization method
        apply_clahe=True,                 # Apply CLAHE contrast enhancement
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_method=None,              # No denoising
        crop_borders=False,               # Don't crop borders
        padding_mode="constant",          # Zero padding
    )
    
    # Initialize dataset preprocessor
    preprocessor = DatasetPreprocessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        config=config,
        seed=42,                          # Random seed for reproducibility
        train_split=0.8,                  # 80% train, 20% validation
        exclude_classes=[14],             # Exclude "No finding" class
        verbose=True,                     # Print progress
    )
    
    # Process the entire dataset
    print("Starting VinDr-CXR dataset preprocessing...")
    print("="*60)
    
    stats = preprocessor.process_dataset()
    
    # Create dataset.yaml for YOLO training
    preprocessor.create_dataset_yaml()
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nDataset structure:")
    print("  images/")
    print("    train/  - {stats['train_images']} training images")
    print("    val/    - {stats['val_images']} validation images")
    print("  labels/")
    print("    train/  - YOLO format annotations")
    print("    val/    - YOLO format annotations")
    print("  dataset.yaml - YOLO dataset configuration")
    print("  statistics.txt - Processing statistics")


if __name__ == "__main__":
    main()
