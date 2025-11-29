"""
Word Document (.docx) Pipeline for parsing Microsoft Word files.

Extracts text, tables, and images from Word documents.
Saves extracted images to disk and returns their paths.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from docx import Document
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
from PIL import Image
import io

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Import image description function
try:
    from od_parse.excel.gemini_client import generate_excel_image_description
except ImportError:
    generate_excel_image_description = None


def iter_block_items(parent):
    """
    Iterate through document elements in order (paragraphs and tables).
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("something's not right")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def runDocxPipeline(file_path: str, args, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Parses a .docx file for text, tables, and images.
    Saves extracted images to disk and returns their paths.
    
    Args:
        file_path: Path to the Word document file
        args: argparse.Namespace object with .mech attribute
        output_dir: Optional output directory (if None, creates output/{filename}_images/)
        
    Returns:
        Dictionary with structure:
        {
            "file_type": "docx",
            "text_content": [...],
            "tables": [...],
            "images": [{"path": "...", "format": "...", "size_bytes": ..., "description": "..."}]
        }
    """
    logger.info(f"Starting DOCX Pipeline for: {file_path}")
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Word document not found: {file_path}")
    
    doc = Document(file_path)
    
    # Create an output directory for images based on the input filename
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = Path("output") / f"{base_name}_images"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get API key from environment if --mech flag is set
    api_key = os.getenv("GOOGLE_API_KEY") if args.mech else None
    if args.mech and not api_key:
        logger.warning("--mech flag is set but GOOGLE_API_KEY not found. Images will not be captioned.")
    
    result = {
        "file_type": "docx",
        "text_content": [],
        "tables": [],
        "images": []
    }

    # 1. Iterate through content in order
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                result["text_content"].append(text)
        
        elif isinstance(block, Table):
            # Extract table data to a list of dicts (like a DataFrame)
            data = []
            keys = None
            for i, row in enumerate(block.rows):
                text = (cell.text.strip() for cell in row.cells)
                if i == 0:
                    keys = tuple(text)
                    continue
                # Handle case where keys length doesn't match row length
                row_values = list(text)
                if keys and len(row_values) == len(keys):
                    row_data = dict(zip(keys, row_values))
                    data.append(row_data)
                else:
                    data.append(row_values)  # Fallback to list
            
            result["tables"].append(data)

    # 2. Extract Images (using document relationships - more reliable)
    logger.info("Extracting images from DOCX...")
    image_count = 0
    
    # Iterate through all relationships to find images
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            try:
                image_bytes = rel.target_part.blob
                content_type = rel.target_part.content_type
                
                # Determine extension
                ext = "png"  # Default
                if "jpeg" in content_type or "jpg" in content_type:
                    ext = "jpg"
                elif "gif" in content_type:
                    ext = "gif"
                elif "bmp" in content_type:
                    ext = "bmp"
                
                # Generate unique filename
                image_filename = f"image_{image_count}.{ext}"
                image_path = output_dir / image_filename
                
                # Save to disk (ALWAYS happen, regardless of flags)
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                image_info = {
                    "path": str(image_path),
                    "format": ext,
                    "size_bytes": len(image_bytes),
                    "description": "Image extracted from document"
                }
                
                # Semantic Analysis (ONLY if --mech is on)
                if args.mech and generate_excel_image_description and api_key:
                    try:
                        # Convert bytes to PIL Image (required by generate_excel_image_description)
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        desc = generate_excel_image_description(pil_image, api_key)
                        image_info["description"] = desc
                    except Exception as e:
                        logger.warning(f"Failed to generate description for image {image_count}: {e}")
                
                result["images"].append(image_info)
                image_count += 1
                logger.info(f"Extracted and saved image {image_count} to: {image_path}")
                
            except Exception as e:
                logger.warning(f"Failed to extract an image: {e}")

    logger.info(f"DOCX extraction complete. Found {len(result['tables'])} tables and {len(result['images'])} images.")
    return result
