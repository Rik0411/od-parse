"""
Robust PDF image extraction with inline image support.

This module provides enhanced PDF image extraction that handles:
- Standard XObject images
- Inline images (embedded in content streams)
- Image location tracking
- Contextual text association
"""

from dataclasses import dataclass
from typing import List, Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddedImage:
    """Represents an embedded image extracted from a PDF."""
    image_bytes: bytes
    format: str  # 'png', 'jpeg', etc.
    page_num: int
    location: dict  # {'x': float, 'y': float, 'w': float, 'h': float}
    nearest_text: Optional[str] = None


def extract_images_from_pdf(doc: fitz.Document) -> List[EmbeddedImage]:
    """
    Extract all embedded images from a PDF document, including inline images.
    
    Uses a multi-strategy approach:
    1. First tries standard get_images(full=True) to find XObject images
    2. If no images found, calls page.clean_contents() to convert inline images to XObjects
    3. Retries get_images(full=True) after cleaning
    
    Args:
        doc: PyMuPDF Document object (fitz.Document)
        
    Returns:
        List of EmbeddedImage objects, one for each instance of each image on each page
        
    Raises:
        ImportError: If PyMuPDF is not available
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF (fitz) is required for image extraction. Install with: pip install PyMuPDF")
    
    extracted_images = []
    
    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        
        try:
            # Strategy 1: Try the standard approach first
            image_list = page.get_images(full=True)
            
            # Strategy 2: If that fails, check for "Inline Images" (stored in content stream)
            # PyMuPDF doesn't natively list inline images in get_images() until recent versions.
            # We can try to sanitize the page to convert inline images to XObjects.
            if not image_list:
                logger.debug(f"Page {page_num}: No images found with standard method, trying clean_contents() for inline images...")
                try:
                    page.clean_contents()  # This converts inline images to XObjects!
                    image_list = page.get_images(full=True)
                    if image_list:
                        logger.info(f"Page {page_num}: Found {len(image_list)} image(s) after clean_contents()")
                except Exception as e:
                    logger.warning(f"Page {page_num}: Error calling clean_contents(): {e}")
                    image_list = []
            
            if not image_list:
                logger.debug(f"Page {page_num}: No images found")
                continue
            
            # Process each image
            for img in image_list:
                xref = img[0]
                
                try:
                    # Extract image bytes
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    
                    # Get location (Bounding Box)
                    # This is the tricky part. An image can appear multiple times.
                    rects = page.get_image_rects(xref)
                    
                    if not rects:
                        # Image exists but is not displayed (hidden resource)? Skip.
                        logger.debug(f"Page {page_num}: Image xref {xref} exists but has no display rects, skipping")
                        continue
                    
                    # Create an EmbeddedImage for EACH instance of the image on the page
                    for rect in rects:
                        # Find nearest text
                        nearest_text = _find_nearest_text_pdf(page, rect)
                        
                        extracted_images.append(EmbeddedImage(
                            image_bytes=image_bytes,
                            format=ext,
                            page_num=page_num,
                            location={'x': rect.x0, 'y': rect.y0, 'w': rect.width, 'h': rect.height},
                            nearest_text=nearest_text
                        ))
                        
                        logger.debug(f"Page {page_num}: Extracted image xref {xref} at ({rect.x0:.1f}, {rect.y0:.1f}), size {rect.width:.1f}x{rect.height:.1f}")
                
                except Exception as e:
                    logger.warning(f"Page {page_num}: Error extracting image xref {xref}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Page {page_num}: Error processing page: {e}", exc_info=True)
            continue
    
    logger.info(f"Extracted {len(extracted_images)} image instance(s) from {len(doc)} page(s)")
    return extracted_images


def _find_nearest_text_pdf(page: fitz.Page, image_rect: fitz.Rect) -> str:
    """
    Finds text blocks spatially closest to the image.
    
    Args:
        page: PyMuPDF Page object
        image_rect: PyMuPDF Rect object representing image bounding box
        
    Returns:
        String containing the nearest text, or empty string if none found
    """
    try:
        # Get all text blocks
        blocks = page.get_text("dict")["blocks"]
        closest_text = ""
        min_dist = float('inf')
        
        image_center = ((image_rect.x0 + image_rect.x1) / 2, (image_rect.y0 + image_rect.y1) / 2)
        
        for b in blocks:
            if b['type'] == 0:  # Text block
                bbox = fitz.Rect(b['bbox'])
                # Simple distance from center to center
                text_center = ((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)
                dist = ((image_center[0] - text_center[0]) ** 2 + (image_center[1] - text_center[1]) ** 2) ** 0.5
                
                # Filter for "close enough" (e.g., within 200 units)
                if dist < min_dist and dist < 200:
                    min_dist = dist
                    # Get the text content
                    text_lines = []
                    for line in b["lines"]:
                        for span in line["spans"]:
                            text_lines.append(span["text"])
                    closest_text = " ".join(text_lines)
        
        return closest_text.strip()
    
    except Exception as e:
        logger.debug(f"Error finding nearest text: {e}")
        return ""

