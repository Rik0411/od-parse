"""
Image enhancement utilities for improving low-resolution images.

Provides upscaling and sharpening capabilities to improve OCR and detection accuracy.
"""

import tempfile
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


def enhance_low_res_image(image_path: str, min_resolution: int = 1000, scale_factor: float = 2.0) -> str:
    """
    Enhance a low-resolution image by upscaling and sharpening.
    
    This function:
    1. Loads the image using OpenCV
    2. Checks if resolution is below threshold
    3. Upscales using bicubic interpolation
    4. Applies sharpening filter
    5. Saves to temporary file
    
    Args:
        image_path: Path to the input image file
        min_resolution: Minimum width/height threshold (default: 1000px)
                        If image is smaller, it will be upscaled
        scale_factor: Scaling factor for upscaling (default: 2.0 = 2x)
        
    Returns:
        Path to the enhanced temporary image file
        Returns original path if enhancement is not needed or fails
    """
    image_path_obj = Path(image_path)
    
    if not image_path_obj.exists():
        logger.warning(f"Image file not found: {image_path}")
        return str(image_path)
    
    try:
        # Load image with OpenCV
        img = cv2.imread(str(image_path_obj))
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return str(image_path)
        
        height, width = img.shape[:2]
        
        # Check if enhancement is needed
        needs_enhancement = width < min_resolution or height < min_resolution
        
        if not needs_enhancement:
            logger.debug(f"Image resolution ({width}x{height}) is sufficient. No enhancement needed.")
            return str(image_path)
        
        logger.info(f"Enhancing low-resolution image ({width}x{height})...")
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Upscale using bicubic interpolation (better quality than linear)
        upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening filter (unsharp masking)
        # Create sharpening kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        
        # Apply kernel with reduced strength to avoid over-sharpening
        sharpened = cv2.filter2D(upscaled, -1, kernel * 0.3)
        
        # Blend original upscaled and sharpened (70% sharpened, 30% original)
        # This prevents over-sharpening artifacts
        enhanced = cv2.addWeighted(upscaled, 0.3, sharpened, 0.7, 0)
        
        # Save to temporary file
        temp_dir = Path(tempfile.gettempdir())
        temp_file = temp_dir / f"enhanced_{image_path_obj.stem}_{id(enhanced)}.png"
        
        # Ensure temp directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save enhanced image
        cv2.imwrite(str(temp_file), enhanced)
        
        logger.info(f"Enhanced image saved to: {temp_file} ({new_width}x{new_height})")
        
        return str(temp_file)
        
    except Exception as e:
        logger.error(f"Error enhancing image {image_path}: {e}", exc_info=True)
        # Return original path on error
        return str(image_path)


def cleanup_enhanced_image(enhanced_path: str) -> None:
    """
    Clean up a temporary enhanced image file.
    
    Args:
        enhanced_path: Path to the enhanced image file to delete
    """
    try:
        enhanced_path_obj = Path(enhanced_path)
        if enhanced_path_obj.exists() and enhanced_path_obj.name.startswith("enhanced_"):
            enhanced_path_obj.unlink()
            logger.debug(f"Cleaned up temporary enhanced image: {enhanced_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup enhanced image {enhanced_path}: {e}")

