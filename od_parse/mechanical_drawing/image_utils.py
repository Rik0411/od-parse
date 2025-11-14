"""
Image Processing Utilities for Mechanical Drawing Parser

This module provides helper functions for image manipulation, particularly
for cropping patches from full images based on bounding boxes.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple
from PIL import Image

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


def crop_image_patch(
    image_path: Path,
    x: int,
    y: int,
    width: int,
    height: int
) -> str:
    """
    Crop an image patch from the full image and return it as base64 string.
    
    Args:
        image_path: Path to the full image file
        x: Top-left x coordinate (pixels)
        y: Top-left y coordinate (pixels)
        width: Width of the patch (pixels)
        height: Height of the patch (pixels)
    
    Returns:
        Base64 encoded string of the cropped patch (without data URI prefix)
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Ensure we're working with RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get image dimensions
            img_width, img_height = img.size
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            width = max(1, min(width, img_width - x))
            height = max(1, min(height, img_height - y))
            
            # Crop the patch
            patch = img.crop((x, y, x + width, y + height))
            
            # Convert to base64
            buffer = BytesIO()
            # Save as PNG to preserve quality
            patch.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            base64_data = base64.b64encode(buffer.read()).decode('utf-8')
            
            return base64_data
            
    except Exception as e:
        logger.error(f"Failed to crop image patch: {e}")
        raise

