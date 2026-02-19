"""
Image preprocessing transformations.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    target_size: Tuple[int, int]
    normalization_method: str  # 'minmax' or 'zscore'
    apply_clahe: bool
    clahe_clip_limit: float
    clahe_tile_size: int
    denoise_method: Optional[str]  # 'gaussian', 'median', or None
    denoise_kernel_size: int
    crop_borders: bool
    border_threshold: int
    padding_mode: str  # 'constant' or 'edge'


class ImagePreprocessor:
    """Apply standardized preprocessing transformations to images."""
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize with preprocessing configuration."""
        self.config = config
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale using standard luminance weights.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            Grayscale image
        """
        # Check if already grayscale
        if len(image.shape) == 2:
            return image
        
        # Check if color image (3 channels)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert using standard luminance weights
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # If single channel but 3D, squeeze it
        if len(image.shape) == 3 and image.shape[2] == 1:
            return image.squeeze(axis=2)
        
        return image
    
    def resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target dimensions with aspect ratio preservation.
        
        Args:
            image: Input image
            target_size: Target (height, width)
            
        Returns:
            Resized image
        """
        target_h, target_w = target_size
        current_h, current_w = image.shape[:2]
        
        # Calculate aspect ratios
        current_aspect = current_w / current_h
        target_aspect = target_w / target_h
        
        # Determine scaling to fit within target while preserving aspect ratio
        if current_aspect > target_aspect:
            # Width is the limiting factor
            new_w = target_w
            new_h = int(target_w / current_aspect)
        else:
            # Height is the limiting factor
            new_h = target_h
            new_w = int(target_h * current_aspect)
        
        # Choose interpolation method based on scaling direction
        if new_h * new_w > current_h * current_w:
            # Upsampling - use cubic interpolation
            interpolation = cv2.INTER_CUBIC
        else:
            # Downsampling - use area interpolation
            interpolation = cv2.INTER_AREA
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        return resized
    
    def normalize(self, image: np.ndarray, method: str) -> np.ndarray:
        """
        Normalize pixel intensities using minmax or z-score.
        
        Args:
            image: Input image
            method: Normalization method ('minmax' or 'zscore')
            
        Returns:
            Normalized image
        """
        # Convert to float for normalization
        image_float = image.astype(np.float32)
        
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            img_min = image_float.min()
            img_max = image_float.max()
            
            # Handle constant images (all pixels same value)
            if img_max == img_min:
                return np.zeros_like(image_float)
            
            normalized = (image_float - img_min) / (img_max - img_min)
            return normalized
            
        elif method == 'zscore':
            # Z-score normalization (zero mean, unit variance)
            mean = image_float.mean()
            std = image_float.std()
            
            # Handle zero standard deviation (constant images)
            if std == 0:
                return np.zeros_like(image_float)
            
            normalized = (image_float - mean) / std
            return normalized
            
        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'zscore'.")
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
        """
        Apply CLAHE contrast enhancement.
        
        Args:
            image: Input grayscale image
            clip_limit: Threshold for contrast limiting
            tile_size: Size of grid for histogram equalization
            
        Returns:
            Contrast-enhanced image
        """
        # Ensure image is uint8 for CLAHE
        if image.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image_uint8 = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image_uint8 = np.zeros_like(image, dtype=np.uint8)
        else:
            image_uint8 = image
        
        # Use safe defaults for invalid parameters
        if clip_limit <= 0:
            clip_limit = 2.0
        if tile_size <= 0:
            tile_size = 8
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        
        # Apply CLAHE
        enhanced = clahe.apply(image_uint8)
        
        return enhanced
    
    def denoise(self, image: np.ndarray, method: str, kernel_size: int) -> np.ndarray:
        """
        Apply denoising filter.
        
        Args:
            image: Input image
            method: Denoising method ('gaussian' or 'median')
            kernel_size: Size of the kernel
            
        Returns:
            Denoised image
        """
        # Adjust even kernel sizes to nearest odd number
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        
        # Ensure minimum kernel size
        if kernel_size < 3:
            kernel_size = 3
        
        if method == 'gaussian':
            # Apply Gaussian blur
            denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            return denoised
            
        elif method == 'median':
            # Apply median filter
            denoised = cv2.medianBlur(image, kernel_size)
            return denoised
            
        else:
            raise ValueError(f"Unknown denoising method: {method}. Use 'gaussian' or 'median'.")
    
    def crop_borders(self, image: np.ndarray, threshold: int) -> np.ndarray:
        """
        Detect and crop black borders automatically.
        
        Args:
            image: Input image
            threshold: Pixel value threshold for border detection
            
        Returns:
            Cropped image
        """
        # Find non-border pixels (pixels above threshold)
        mask = image > threshold
        
        # Find bounding box of non-border region
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        # If no pixels above threshold, return original
        if not np.any(rows) or not np.any(cols):
            return image
        
        # Get crop coordinates
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        
        # Crop image
        cropped = image[row_min:row_max+1, col_min:col_max+1]
        
        return cropped
    
    def pad_to_size(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Pad image to uniform dimensions, centering the original content.
        
        Args:
            image: Input image
            target_size: Target (height, width)
            
        Returns:
            Padded image
        """
        target_h, target_w = target_size
        current_h, current_w = image.shape[:2]
        
        # If image is already larger than target, return as is
        if current_h >= target_h and current_w >= target_w:
            return image
        
        # Calculate padding amounts to center the image
        pad_h = max(0, target_h - current_h)
        pad_w = max(0, target_w - current_w)
        
        # Split padding evenly on both sides
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Apply padding based on mode
        if self.config.padding_mode == 'constant':
            # Zero padding
            padded = cv2.copyMakeBorder(
                image, 
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, 
                value=0
            )
        elif self.config.padding_mode == 'edge':
            # Edge replication
            padded = cv2.copyMakeBorder(
                image,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REPLICATE
            )
        else:
            # Default to constant padding
            padded = cv2.copyMakeBorder(
                image,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=0
            )
        
        return padded
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            image: Input image
            
        Returns:
            Fully preprocessed image
        """
        # 1. Convert to grayscale
        processed = self.to_grayscale(image)
        
        # 2. Crop borders if enabled
        if self.config.crop_borders:
            processed = self.crop_borders(processed, self.config.border_threshold)
        
        # 3. Resize to target size
        processed = self.resize(processed, self.config.target_size)
        
        # 4. Pad to exact target size if needed
        processed = self.pad_to_size(processed, self.config.target_size)
        
        # 5. Apply denoising if specified
        if self.config.denoise_method:
            processed = self.denoise(
                processed, 
                self.config.denoise_method, 
                self.config.denoise_kernel_size
            )
        
        # 6. Apply CLAHE if enabled
        if self.config.apply_clahe:
            processed = self.apply_clahe(
                processed,
                self.config.clahe_clip_limit,
                self.config.clahe_tile_size
            )
        
        # 7. Normalize
        processed = self.normalize(processed, self.config.normalization_method)
        
        return processed
