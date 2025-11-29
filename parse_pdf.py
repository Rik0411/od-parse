#!/usr/bin/env python3
"""
Intelligent file parser that automatically routes different file types to appropriate parsing pipelines.

This script supports multiple file formats:
- Raster images (.jpg, .jpeg, .png) → Raster Pipeline (hybrid Roboflow + Gemini)
- PDF files (.pdf) → Intelligent PDF Triage:
    - Vector PDFs (text-based): Fast text-only parsing via Gemini (skips pdf2image and Roboflow)
    - Raster PDFs (scanned): Image-based pipeline (pdf2image + Roboflow + Gemini)
- Vector files (.dxf, .dwg) → Vector Pipeline (simulation)

The master controller (_route_file_by_type) automatically detects file type and delegates
to the correct pipeline, making this an intelligent, multi-format parser.
"""

import argparse
import io
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image

# Try to import PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    # Set stdout and stderr to UTF-8 encoding
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    # Configure logging to use UTF-8 encoding
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    # Ensure the handler uses UTF-8
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stdout)

# Create logger for this module
logger = logging.getLogger(__name__)

# Add project directory to Python path so od_parse can be imported
project_dir = Path(__file__).parent.resolve()
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass  # dotenv is optional

# Import from main module to get LLM-enhanced parser
from od_parse.main import parse_pdf
# Import Excel pipeline
from od_parse.excel import runExcelPipeline
# Import Office document pipelines
from od_parse.office import runDocxPipeline, runPptxPipeline
# Import robust image extraction
from od_parse.image_extraction import extract_images_from_pdf
# Import image description function
try:
    from od_parse.excel.gemini_client import generate_excel_image_description
except ImportError:
    generate_excel_image_description = None

def main():
    parser = argparse.ArgumentParser(
        description='Intelligent file parser supporting PDF, images (PNG, JPG, JPEG), vector files (DXF, DWG), and Excel files (.xlsx, .xls) using odparse with Gemini VLM'
    )
    parser.add_argument('input_file', help='Path to the file to parse (PDF, image, or vector file)')
    parser.add_argument('--output-dir', '-o', default='output', help='Output directory for JSON file (default: output)')
    parser.add_argument('--mech', '--mechanical', action='store_true', 
                       help='Use mechanical drawing pipeline for image files (hybrid Roboflow + Gemini)')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1
    
    # Check file extension
    file_ext = input_path.suffix.lower()
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.dxf', '.dwg', '.xlsx', '.xls', '.docx', '.pptx']
    if file_ext not in supported_extensions:
        print(f"Error: Unsupported file type: {file_ext}", file=sys.stderr)
        print(f"Supported formats: {', '.join(supported_extensions)}", file=sys.stderr)
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get API keys from environment
    api_keys = {
        "google": os.getenv("GOOGLE_API_KEY")
    }
    
    # Google API key is only required for raster/PDF/Excel pipelines, not vector
    if file_ext not in ['.dxf', '.dwg'] and not api_keys["google"]:
        # Excel files only need API key if --mech flag is set
        if file_ext in ['.xlsx', '.xls'] and not args.mech:
            pass  # Excel without --mech doesn't need API key
        else:
            print("Error: GOOGLE_API_KEY not found in environment variables", file=sys.stderr)
            print("Please set GOOGLE_API_KEY in your .env file or environment", file=sys.stderr)
            return 1
    
    # Process file using intelligent router
    try:
        print(f"Parsing file: {input_path.name}")
        result = _route_file_by_type(input_path, api_keys, use_mechanical_drawing=args.mech, args=args, output_dir=output_dir)
        
        # Write output to JSON file
        output_file = output_dir / f"{input_path.stem}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as json_error:
            # Handle JSON serialization errors (e.g., NaN, infinity)
            print(f"Warning: JSON serialization issue, attempting to clean data: {json_error}", file=sys.stderr)
            # Try with a simple cleaner for common issues
            def clean_for_json(obj):
                if isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_for_json(item) for item in obj]
                elif isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return None
                    return obj
                else:
                    return obj
            cleaned_result = clean_for_json(result)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Output saved to: {output_file}")
        return 0
        
    except Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


def _split_text_into_paragraphs(text: str) -> List[str]:
    """
    Split text into an array of paragraphs.
    
    Args:
        text: Input text string (may contain newlines)
        
    Returns:
        List of paragraph strings, with empty paragraphs filtered out
    """
    if not text or not text.strip():
        return []
    
    import re
    
    # First try splitting by double newlines (paragraph breaks)
    if '\n\n' in text:
        paragraphs = text.split('\n\n')
    elif '\n' in text:
        # Fallback: split by single newlines if no double newlines found
        paragraphs = text.split('\n')
    else:
        # If no newlines, split on sentence boundaries
        # Split on periods, exclamation marks, or question marks followed by space
        # This will create sentence boundaries
        sentences = re.split(r'([.!?])\s+', text)
        
        # Reconstruct sentences with their punctuation
        reconstructed_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i].strip() + sentences[i + 1]
                if sentence:
                    reconstructed_sentences.append(sentence)
        # Add last part if it exists and wasn't paired
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            reconstructed_sentences.append(sentences[-1].strip())
        
        # Group sentences into paragraphs (every 2-4 sentences, or ~200-400 chars)
        paragraphs = []
        current_para = ""
        sentence_count = 0
        
        for sentence in reconstructed_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_para:
                current_para += " " + sentence
            else:
                current_para = sentence
            
            sentence_count += 1
            
            # Start new paragraph if:
            # 1. We have 3-4 sentences, OR
            # 2. Current paragraph is getting long (>400 chars)
            if sentence_count >= 3 or len(current_para) > 400:
                paragraphs.append(current_para)
                current_para = ""
                sentence_count = 0
        
        # Add remaining text as last paragraph
        if current_para:
            paragraphs.append(current_para)
    
    # Filter and clean paragraphs
    result = []
    for para in paragraphs:
        cleaned = para.strip()
        if cleaned:  # Only add non-empty paragraphs
            result.append(cleaned)
    
    return result


def _parse_image_file(image_path: Path, api_keys: dict, use_mechanical_drawing: bool = False, page_number: int = 1) -> dict:
    """
    Parse an image file using either the mechanical drawing pipeline or LLM-based parsing.
    
    Args:
        image_path: Path to the image file
        api_keys: Dictionary of API keys (must contain 'google' key for Gemini, 'roboflow' is optional)
        use_mechanical_drawing: If True, use the hybrid three-stage mechanical drawing pipeline.
                               If False (default), use LLM-based parsing.
        page_number: The page number this image corresponds to (for labeling, default: 1)
    
    Returns:
        Dictionary containing parsed results in the format compatible with existing code
    """
    # Validate image file exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # If --mech flag is not set, use LLM fallback directly
    if not use_mechanical_drawing:
        return _parse_image_file_llm_fallback(image_path, api_keys, page_number)
    
    # Use mechanical drawing pipeline when --mech flag is set
    from od_parse.mechanical_drawing.pipeline import run_full_hybrid_pipeline_sync
    import os
    
    # Extract API keys
    gemini_api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it in api_keys dict.")
    
    # Roboflow API key is optional (has default)
    roboflow_api_key = api_keys.get("roboflow") or os.getenv("ROBOFLOW_API_KEY")
    
    # Run the hybrid three-stage pipeline
    try:
        pipeline_result = run_full_hybrid_pipeline_sync(
            image_path,
            roboflow_api_key=roboflow_api_key,
            gemini_api_key=gemini_api_key,
            page_number=page_number
        )
        
        # Transform pipeline output to match expected format structure
        enhanced_data = {
            'text_content': [],  # No text extraction in mechanical drawing pipeline
            'images': [],
            'tables': [],
            'forms': [],
            'handwritten_content': [],
            'metadata': {
                'file_name': image_path.name,
                'file_size': image_path.stat().st_size,
                'page_count': 1,
                'page_number': page_number,
                'extraction_method': 'mechanical_drawing_pipeline'
            },
            'document_classification': {
                'document_type': 'mechanical_drawing',
                'confidence': 1.0,
                'detected_patterns': ['technical_drawing', 'engineering_diagram'],
                'key_indicators': {},
                'suggestions': []
            },
            'processing_metadata': {
                'model_used': 'hybrid_mechanical_drawing_pipeline',
                'document_type': 'mechanical_drawing',
                'processing_strategy': 'three_stage_hybrid_pipeline',
                'vision_enabled': True
            },
            'extracted_data': pipeline_result
        }
        
        return enhanced_data
        
    except Exception as e:
        # Fallback to LLM if pipeline fails (for non-mechanical drawings)
        print(f"Warning: Mechanical drawing pipeline failed: {e}")
        print("Falling back to LLM-based parsing...")
        return _parse_image_file_llm_fallback(image_path, api_keys, page_number)


def runPdfPipeline(pdf_path: str, args, output_dir: Path = None) -> dict:
    """
    PIPELINE (TRIAGE): Opens the PDF and decides which pipeline to use.
    
    - Vector: If text is found AND --mech is True.
    - Raster: If no text is found, OR if --mech is False.
    
    Args:
        pdf_path: Path to the PDF file as string
        args: argparse.Namespace object with .mech attribute
        output_dir: Optional output directory for saving extracted images
        
    Returns:
        Dictionary containing parsed results
    """
    logger.info("Starting PDF Triage: Checking for vector text...")
    doc = None
    try:
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available. Falling back to raster pipeline.")
            return runRasterPdfPipeline(pdf_path, args, output_dir)
        
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        has_vector_text = False
        
        # Check first 5 pages or all pages, whichever is smaller
        pages_to_check = min(len(doc), 5)
        for i in range(pages_to_check):
            page = doc[i]
            # page.get_text("text") is faster for a simple check
            if page.get_text("text").strip():
                has_vector_text = True
                break
        
        # --- NEW TRIAGE LOGIC ---
        
        # PATH 1: Vector PDF + Mech Flag
        # This is the "fast path": PyMuPDF text -> Gemini text parser
        if has_vector_text and args.mech:
            logger.info("Vector PDF detected. Routing to PyMuPDF + Gemini-Text pipeline.")
            # We pass the *open document* to avoid re-opening
            # doc is closed inside runVectorPdfPipeline
            return runVectorPdfPipeline(doc, args, output_dir)
        
        # PATH 2: Raster PDF OR Default Parser
        # This is the "fallback path": pdf2image -> Image Parser
        else:
            doc.close()  # We don't need the doc anymore
            doc = None
            
            if has_vector_text:
                logger.info("Vector PDF detected, but --mech flag is off. Routing to default Raster pipeline (as it requires images).")
            else:
                logger.info("Raster PDF (scanned image) detected. Routing to Raster pipeline.")
            
            # Fall back to the "convert all to image" method
            # This function handles both --mech (Roboflow) and default (single-stage)
            return runRasterPdfPipeline(pdf_path, args, output_dir)
        # --- END NEW TRIAGE LOGIC ---
        
    except Exception as e:
        if doc:
            doc.close()
        logger.error(f"Error during PDF triage: {e}", exc_info=True)
        raise


def runVectorPdfPipeline(doc, args, output_dir: Path = None) -> dict:
    """
    Process a vector PDF using the open PyMuPDF document.
    
    Extracts text from all pages and processes through Gemini text-only parser.
    
    Args:
        doc: Open PyMuPDF document (will be closed after processing)
        args: argparse.Namespace object
        output_dir: Optional output directory for saving extracted images
        
    Returns:
        Dictionary containing parsed results
    """
    try:
        # Extract text_data from all pages
        text_data = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text_dict = page.get_text("dict")
            text_data.append(page_text_dict)
        
        # Get API keys from environment
        api_keys = {
            "google": os.getenv("GOOGLE_API_KEY")
        }
        
        # Convert pdf_path from document name (doc.name is the file path)
        pdf_path = Path(doc.name) if hasattr(doc, 'name') and doc.name else None
        if not pdf_path:
            raise ValueError("Cannot determine PDF path from document")
        
        # Extract embedded images before closing document
        image_objects = []
        if output_dir:
            try:
                embedded_images = extract_images_from_pdf(doc)
                api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY")
                
                # Save images to output directory
                for img in embedded_images:
                    img_filename = f"page_{img.page_num}_embedded_{len(image_objects)}.{img.format}"
                    img_path = output_dir / img_filename
                    
                    # Ensure output directory exists
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save image bytes to file
                    with open(img_path, 'wb') as f:
                        f.write(img.image_bytes)
                    
                    # Create image object with standardized format
                    image_info = {
                        "path": str(img_path),
                        "format": img.format,
                        "size_bytes": len(img.image_bytes),
                        "description": "Image extracted from PDF"
                    }
                    
                    # Generate description if API key is available
                    if generate_excel_image_description and api_key:
                        try:
                            pil_image = Image.open(io.BytesIO(img.image_bytes))
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            desc = generate_excel_image_description(pil_image, api_key)
                            image_info["description"] = desc
                        except Exception as e:
                            logger.warning(f"Failed to generate description for image from page {img.page_num}: {e}")
                    
                    image_objects.append(image_info)
                    logger.info(f"Extracted embedded image from page {img.page_num}: {img_path}")
            except Exception as e:
                logger.warning(f"Failed to extract embedded images: {e}")
        
        # Call existing vector pipeline
        result = _run_vector_pdf_pipeline(pdf_path, api_keys, text_data)
        
        # Add file_type to result
        result['file_type'] = 'pdf'
        
        # Add images to result if we have a standard structure
        # Note: Vector pipeline returns mechanical drawing format, so we add images to metadata or top level
        if image_objects:
            if 'images' not in result:
                result['images'] = []
            result['images'].extend(image_objects)
        
        # Remove empty images array if it exists and is empty
        if 'images' in result and not result['images']:
            del result['images']
        
        return result
        
    finally:
        # Always close the document
        if doc:
            doc.close()


def runRasterPdfPipeline(pdf_path: str, args, output_dir: Path = None) -> dict:
    """
    Process a raster PDF (or vector PDF when --mech is False) using image-based pipeline.
    
    Converts PDF to images and processes through appropriate pipeline based on --mech flag.
    
    Args:
        pdf_path: Path to the PDF file as string
        args: argparse.Namespace object with .mech attribute
        output_dir: Optional output directory for saving extracted images
        
    Returns:
        Dictionary containing parsed results
    """
    # Get API keys from environment
    api_keys = {
        "google": os.getenv("GOOGLE_API_KEY")
    }
    
    pdf_path_obj = Path(pdf_path)
    
    # If --mech flag is set, use mechanical drawing pipeline
    if args.mech:
        result = _run_raster_pdf_pipeline(pdf_path_obj, api_keys)
        
        # Extract embedded images and add to result
        if output_dir:
            try:
                doc = fitz.open(pdf_path_obj)
                embedded_images = extract_images_from_pdf(doc)
                doc.close()
                
                api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY")
                
                # Save images to output directory
                image_objects = []
                for img in embedded_images:
                    img_filename = f"page_{img.page_num}_embedded_{len(image_objects)}.{img.format}"
                    img_path = output_dir / img_filename
                    
                    # Ensure output directory exists
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save image bytes to file
                    with open(img_path, 'wb') as f:
                        f.write(img.image_bytes)
                    
                    # Create image object with standardized format
                    image_info = {
                        "path": str(img_path),
                        "format": img.format,
                        "size_bytes": len(img.image_bytes),
                        "description": "Image extracted from PDF"
                    }
                    
                    # Generate description if API key is available
                    if generate_excel_image_description and api_key:
                        try:
                            pil_image = Image.open(io.BytesIO(img.image_bytes))
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            desc = generate_excel_image_description(pil_image, api_key)
                            image_info["description"] = desc
                        except Exception as e:
                            logger.warning(f"Failed to generate description for image from page {img.page_num}: {e}")
                    
                    image_objects.append(image_info)
                    logger.info(f"Extracted embedded image from page {img.page_num}: {img_path}")
                
                # Add images to result
                if 'images' not in result:
                    result['images'] = []
                result['images'].extend(image_objects)
            except Exception as e:
                logger.warning(f"Failed to extract embedded images: {e}")
        
        # Add file_type to result
        result['file_type'] = 'pdf'
        
        # Remove empty images array if it exists and is empty
        if 'images' in result and not result['images']:
            del result['images']
        
        return result
    else:
        # Non-mechanical drawing mode: use existing LLM fallback logic
        temp_image_paths = []
        aggregated_result = None
        
        try:
            # Convert ALL pages of PDF to PNG images
            temp_image_paths = _convert_pdf_to_image(pdf_path_obj)
            total_pages = len(temp_image_paths)
            print(f"PDF converted to {total_pages} image(s)")
            
            # Get API key for image descriptions
            api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY")
            
            # For LLM fallback, aggregate text, tables, etc.
            aggregated_result = {
                'file_type': 'pdf',
                'text_content': [],
                'images': [],
                'tables': []
            }
            
            # Extract embedded images from PDF
            if output_dir:
                try:
                    doc = fitz.open(pdf_path_obj)
                    embedded_images = extract_images_from_pdf(doc)
                    doc.close()
                    
                    # Save embedded images to output directory
                    for img in embedded_images:
                        # Create filename: page_{page_num}_embedded_{index}.{format}
                        img_filename = f"page_{img.page_num}_embedded_{len(aggregated_result['images'])}.{img.format}"
                        img_path = output_dir / img_filename
                        
                        # Ensure output directory exists
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save image bytes to file
                        with open(img_path, 'wb') as f:
                            f.write(img.image_bytes)
                        
                        # Create image object with standardized format
                        image_info = {
                            "path": str(img_path),
                            "format": img.format,
                            "size_bytes": len(img.image_bytes),
                            "description": "Image extracted from PDF"
                        }
                        
                        # Generate description if API key is available
                        if generate_excel_image_description and api_key:
                            try:
                                pil_image = Image.open(io.BytesIO(img.image_bytes))
                                if pil_image.mode != 'RGB':
                                    pil_image = pil_image.convert('RGB')
                                desc = generate_excel_image_description(pil_image, api_key)
                                image_info["description"] = desc
                            except Exception as e:
                                logger.warning(f"Failed to generate description for image from page {img.page_num}: {e}")
                        
                        aggregated_result['images'].append(image_info)
                        logger.info(f"Extracted embedded image from page {img.page_num}: {img_path}")
                except Exception as e:
                    logger.warning(f"Failed to extract embedded images: {e}")
            
            # Process each page
            for page_num, temp_image_path in enumerate(temp_image_paths, start=1):
                try:
                    print(f"Processing page {page_num}/{total_pages}...")
                    page_result = _parse_image_file(
                        temp_image_path, 
                        api_keys, 
                        use_mechanical_drawing=False,
                        page_number=page_num
                    )
                    
                    # LLM fallback: merge text, tables, forms
                    # Handle both 'text' (old format) and 'text_content' (new format) for backward compatibility
                    page_text = page_result.get('text_content', [])
                    if not page_text and 'text' in page_result:
                        # Convert old format to new format
                        page_text = _split_text_into_paragraphs(page_result.get('text', ''))
                    
                    if page_text:
                        # Add page marker as a separate paragraph if we already have content
                        if aggregated_result['text_content']:
                            aggregated_result['text_content'].append(f'--- Page {page_num} ---')
                        # Append all paragraphs from this page
                        aggregated_result['text_content'].extend(page_text)
                    if 'tables' in page_result:
                        aggregated_result['tables'].extend(page_result.get('tables', []))
                    
                except Exception as page_error:
                    print(f"Warning: Failed to process page {page_num}: {page_error}", file=sys.stderr)
                    # Continue processing other pages
                    continue
            
            # Remove empty fields before returning
            return {k: v for k, v in aggregated_result.items() if v or k == 'file_type'}
            
        finally:
            # Clean up all temporary files
            for temp_path in temp_image_paths:
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception as e:
                        print(f"Warning: Failed to delete temporary file {temp_path}: {e}", file=sys.stderr)


def _triage_pdf(pdf_path: Path) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Triage a PDF to determine if it contains extractable text (vector) or is scanned (raster).
    
    Uses PyMuPDF (fitz) to check if pages contain extractable text.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (has_text: bool, text_data: List[Dict]) where:
        - has_text: True if any page has extractable text (vector PDF), False otherwise (raster PDF)
        - text_data: List of page-level text dictionaries from PyMuPDF
    """
    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF not available. Falling back to raster pipeline.")
        return (False, [])
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        text_data = []
        has_text = False
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text_dict = page.get_text("dict")
            text_data.append(page_text_dict)
            
            # Check if page has any text blocks with actual content
            blocks = page_text_dict.get("blocks", [])
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    lines = block.get("lines", [])
                    for line in lines:
                        spans = line.get("spans", [])
                        for span in spans:
                            text = span.get("text", "").strip()
                            if text:  # Found non-empty text
                                has_text = True
                                break
                        if has_text:
                            break
                    if has_text:
                        break
                if has_text:
                    break
        
        doc.close()
        
        logger.info(f"PDF triage complete: {'Vector' if has_text else 'Raster'} PDF detected ({len(text_data)} pages)")
        return (has_text, text_data)
        
    except Exception as e:
        logger.error(f"Error triaging PDF: {e}", exc_info=True)
        # Fallback to raster if triage fails
        return (False, [])


def _run_vector_pdf_pipeline(pdf_path: Path, api_keys: dict, text_data: List[Dict[str, Any]]) -> dict:
    """
    Process a vector PDF (text-based) using fast text-only parsing.
    
    This pipeline skips pdf2image and Roboflow, directly parsing extracted text.
    
    Args:
        pdf_path: Path to the PDF file
        api_keys: Dictionary of API keys (must contain 'google' key for Gemini)
        text_data: List of page-level text dictionaries from PyMuPDF
        
    Returns:
        Dictionary containing parsed results in mechanical drawing format
    """
    from od_parse.mechanical_drawing.gemini_client import stage2_runBatchTextParsing
    
    # Get API key
    gemini_api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it in api_keys dict.")
    
    # Initialize aggregated result structure
    aggregated_result = {
        'file_type': 'pdf',
        'Measures': [],
        'Radii': [],
        'Views': [],
        'GD_T': [],
        'Materials': [],
        'Notes': [],
        'Threads': [],
        'SurfaceRoughness': [],
        'GeneralTolerances': [],
        'TitleBlock': [],
        'Other': [],
        '_Errors': [],
        '_FalsePositives': [],
        'metadata': {
            'source_file': str(pdf_path),
            'source_type': 'pdf',
            'extraction_method': 'vector_pdf_pipeline',
            'total_pages': len(text_data),
            'pipeline_type': 'vector'
        }
    }
    
    # Extract text items from each page and process
    for page_num, page_text_dict in enumerate(text_data, start=1):
        try:
            logger.info(f"Processing vector PDF page {page_num}/{len(text_data)}...")
            
            # Extract text strings from the page text dict
            text_items = []
            blocks = page_text_dict.get("blocks", [])
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    lines = block.get("lines", [])
                    for line in lines:
                        spans = line.get("spans", [])
                        for span in spans:
                            text = span.get("text", "").strip()
                            if text:  # Only add non-empty text
                                text_items.append(text)
            
            if not text_items:
                logger.warning(f"No text found on page {page_num}")
                continue
            
            # Call text batch parser
            parsed_items = stage2_runBatchTextParsing(
                text_items=text_items,
                page_number=page_num,
                api_key=gemini_api_key
            )
            
            # Categorize parsed items
            for item in parsed_items:
                if not item or not isinstance(item, dict):
                    continue
                
                item_type = item.get("type", "").lower()
                item_id = item.get("id", f"item_{page_num}_{len(aggregated_result['Other'])}")
                
                # Map parsed types to categories
                if item_type in ["linear", "angular", "diameter"]:
                    aggregated_result['Measures'].append({
                        'id': item_id,
                        'type': 'LinearDimension',
                        'value': item.get("value"),
                        'units': item.get("units", "mm"),
                        'text': item.get("text", ""),
                        'page_number': page_num
                    })
                elif item_type == "radial":
                    aggregated_result['Radii'].append({
                        'id': item_id,
                        'type': 'Radius',
                        'value': item.get("value"),
                        'text': item.get("text", ""),
                        'page_number': page_num
                    })
                elif "view" in item_type or item.get("content"):
                    aggregated_result['Views'].append({
                        'id': item_id,
                        'type': 'ViewLabel',
                        'name': item.get("content", item.get("text", "")),
                        'text': item.get("text", ""),
                        'page_number': page_num
                    })
                else:
                    aggregated_result['Other'].append({
                        **item,
                        'page_number': page_num
                    })
                    
        except Exception as page_error:
            logger.error(f"Error processing vector PDF page {page_num}: {page_error}", exc_info=True)
            aggregated_result['_Errors'].append({
                'page': page_num,
                'error': str(page_error)
            })
            continue
    
    return aggregated_result


def _run_raster_pdf_pipeline(pdf_path: Path, api_keys: dict) -> dict:
    """
    Process a raster PDF (scanned/image-based) using existing image pipeline.
    
    This is the existing PDF processing logic that converts PDF to images
    and processes through the hybrid Roboflow + Gemini pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        api_keys: Dictionary of API keys (must contain 'google' key for Gemini)
        
    Returns:
        Dictionary containing parsed results in mechanical drawing format
    """
    temp_image_paths = []
    aggregated_result = None
    
    try:
        # Convert ALL pages of PDF to PNG images
        temp_image_paths = _convert_pdf_to_image(pdf_path)
        total_pages = len(temp_image_paths)
        logger.info(f"PDF converted to {total_pages} image(s) for raster processing")
        
        # Initialize aggregated result structure
        aggregated_result = {
            'file_type': 'pdf',
            'Measures': [],
            'Radii': [],
            'Views': [],
            'GD_T': [],
            'Materials': [],
            'Notes': [],
            'Threads': [],
            'SurfaceRoughness': [],
            'GeneralTolerances': [],
            'TitleBlock': [],
            'Other': [],
            '_Errors': [],
            '_FalsePositives': [],
            'metadata': {
                'source_file': str(pdf_path),
                'source_type': 'pdf',
                'extraction_method': 'raster_pdf_pipeline',
                'total_pages': total_pages,
                'pipeline_type': 'raster'
            }
        }
        
        # Process each page
        for page_num, temp_image_path in enumerate(temp_image_paths, start=1):
            try:
                logger.info(f"Processing raster PDF page {page_num}/{total_pages}...")
                page_result = _parse_image_file(
                    temp_image_path, 
                    api_keys, 
                    use_mechanical_drawing=True,
                    page_number=page_num
                )
                
                # Aggregate results from mechanical pipeline
                if 'extracted_data' in page_result:
                    extracted = page_result['extracted_data']
                    for category in ['Measures', 'Radii', 'Views', 'GD_T', 'Materials', 
                                   'Notes', 'Threads', 'SurfaceRoughness', 'GeneralTolerances', 
                                   'TitleBlock', 'Other', '_Errors', '_FalsePositives']:
                        if category in extracted:
                            aggregated_result[category].extend(extracted[category])
                    
            except Exception as page_error:
                logger.error(f"Error processing raster PDF page {page_num}: {page_error}", exc_info=True)
                aggregated_result['_Errors'].append({
                    'page': page_num,
                    'error': str(page_error)
                })
                continue
        
        return aggregated_result
        
    finally:
        # Clean up all temporary files
        for temp_path in temp_image_paths:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_path}: {e}")


def _convert_pdf_to_image(pdf_path: Path) -> List[Path]:
    """
    Convert ALL pages of a PDF to PNG images.
    
    This function is used by the PDF Pipeline to rasterize PDFs before
    processing them through the Raster Pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of paths to temporary PNG files (one per page)
        The caller is responsible for deleting these files.
        
    Raises:
        ValueError: If PDF conversion fails
        FileNotFoundError: If PDF file doesn't exist
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        import pdf2image
    except ImportError:
        raise ImportError("pdf2image library is required for PDF processing. Install it with: pip install pdf2image")
    
    # Determine Poppler path: check environment variable first, then use default
    poppler_path = os.getenv("POPPLER_PATH")
    if not poppler_path:
        # Default Windows path
        poppler_path = r"C:\Poppler\poppler-25.07.0\Library\bin"
    
    # Verify Poppler path exists
    poppler_path_obj = Path(poppler_path)
    if not poppler_path_obj.exists():
        raise ValueError(
            f"Poppler not found at: {poppler_path}\n"
            f"Please install Poppler or set POPPLER_PATH environment variable.\n"
            f"Download Poppler from: https://github.com/oschwartz10612/poppler-windows/releases"
        )
    
    temp_image_paths = []
    
    try:
        # Use tempfile.gettempdir() to create a unique prefix for all pages
        # This makes cleanup easier if one page fails.
        base_output_dir = tempfile.gettempdir()
        
        # Create a unique file prefix for this conversion
        file_prefix = f"pdf_page_{os.path.basename(tempfile.mktemp())}"
        
        # Convert ALL pages. This returns a list of PIL Image objects.
        # We remove 'first_page' and 'last_page' to process all pages
        images = pdf2image.convert_from_path(
            pdf_path,
            dpi=200,
            poppler_path=str(poppler_path_obj),
            output_folder=base_output_dir,
            output_file=file_prefix,
            fmt='png'
        )
        
        if not images:
            raise ValueError(f"Failed to extract pages from PDF: {pdf_path}")
        
        # 'images' is a list of PIL objects. Their paths are what we need.
        # pdf2image < 2.8.0 returns paths from convert_from_path,
        # >= 2.8.0 returns PIL objects and we get path from .filename
        for img in images:
            # Check if image has filename attribute (newer pdf2image)
            if hasattr(img, 'filename') and img.filename:
                temp_image_paths.append(Path(img.filename))
            else:
                # Older pdf2image or manual save needed
                # Create temporary file for this page
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.png',
                    prefix=f'{file_prefix}_',
                    dir=base_output_dir,
                    delete=False
                )
                temp_path = Path(temp_file.name)
                temp_file.close()
                img.save(temp_path, 'PNG')
                temp_image_paths.append(temp_path)
        
        logger.info(f"PDF converted to {len(temp_image_paths)} persistent temp images (prefix: {file_prefix})")
        return temp_image_paths
        
    except Exception as e:
        logger.error(f"Failed to convert PDF. Is 'poppler' installed and in your PATH? Error: {e}", exc_info=True)
        # Clean up any files that *were* created
        for path in temp_image_paths:
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass
        raise ValueError(f"Failed to convert PDF to images: {e}")


def _parse_vector_file(vector_path: Path) -> dict:
    """
    Simulate parsing a vector file (DXF, DWG).
    
    This is a simulation function that demonstrates what a vector file parser
    would extract. In a real implementation, this would use a DXF/DWG parsing
    library to extract geometry, layers, and annotations.
    
    Args:
        vector_path: Path to the vector file (DXF or DWG)
        
    Returns:
        Dictionary containing simulated rich JSON structure with views, layers,
        geometry primitives, blocks, and annotations
        
    Raises:
        FileNotFoundError: If vector file doesn't exist
    """
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector file not found: {vector_path}")
    
    # Read file as text to demonstrate file access
    try:
        with open(vector_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()
        file_size = len(file_content)
    except Exception:
        # If text reading fails (binary file), just get file size
        file_size = vector_path.stat().st_size
        file_content = f"<binary file, {file_size} bytes>"
    
    print(f"Simulating parse of {vector_path.name} ({file_size} bytes)")
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Return hardcoded rich JSON structure matching user's specification
    return {
        'views': [
            {
                'view_id': f'view-top-{vector_path.stem}',
                'name': f'Top view (from {vector_path.name})',
                'dimension': '2D',
                'projection': 'top',
                'scale': '1:100',
                'bbox': {
                    'xmin': 0,
                    'ymin': 0,
                    'zmin': 0,
                    'xmax': 120000,
                    'ymax': 30000,
                    'zmax': 0
                },
                'layers': [
                    {
                        'name': 'BOTTLES',
                        'color': 'bylayer',
                        'linetype': 'continuous',
                        'description': 'bottle outlines'
                    },
                    {
                        'name': 'CONVEYORS',
                        'color': 'bylayer',
                        'linetype': 'hidden',
                        'description': 'conveyor centerlines'
                    },
                    {
                        'name': 'DIM',
                        'color': 'bylayer',
                        'linetype': 'continuous',
                        'description': 'dimensions'
                    }
                ],
                'geometry': {
                    'primitives': [
                        {
                            'id': 'g1',
                            'type': 'polyline',
                            'layer': 'CONVEYORS',
                            'points': [[0, 0, 0], [5000, 0, 0], [5000, 10000, 0]],
                            'bbox': {
                                'xmin': 0,
                                'ymin': 0,
                                'zmin': 0,
                                'xmax': 5000,
                                'ymax': 10000,
                                'zmax': 0
                            },
                            'attributes': {
                                'line_width': 0.25
                            }
                        }
                    ],
                    'blocks': [
                        {
                            'id': 'bottle-symbol-1',
                            'block_name': 'BOTTLE_500ML',
                            'layer': 'BOTTLES',
                            'position': [2000, 3000, 0],
                            'rotation_deg': 0,
                            'scale': [1, 1, 1],
                            'attributes': {
                                'tag': 'BOTTLE-500-1',
                                'volume_ml': 500
                            }
                        }
                    ]
                },
                'annotations': {
                    'texts': [
                        {
                            'id': 't1',
                            'content': 'FILLER F-101',
                            'position': [10000, 5000, 0],
                            'layer': 'TEXT'
                        }
                    ],
                    'dimensions': [
                        {
                            'id': 'd1',
                            'type': 'linear',
                            'value': 6000,
                            'units': 'mm',
                            'start': [0, 0, 0],
                            'end': [6000, 0, 0],
                            'text': '6000'
                        }
                    ],
                    'symbols': []
                }
            }
        ],
        'metadata': {
            'file_name': vector_path.name,
            'file_size': file_size,
            'file_type': 'vector',
            'extraction_method': 'simulated_vector_parser',
            'processing_note': 'This is a simulated output. In production, a real DXF/DWG parser would be used.'
        }
    }


def _route_file_by_type(file_path: Path, api_keys: dict, use_mechanical_drawing: bool = False, args=None, output_dir: Path = None) -> dict:
    """
    Master controller that routes files to the appropriate parsing pipeline.
    
    This intelligent router detects the file type by extension and delegates
    to the correct pipeline:
    - Raster files (.jpg, .png, .jpeg) → Raster Pipeline
    - PDF files (.pdf) → Intelligent PDF Triage via runPdfPipeline:
        - Always checks for vector text
        - Uses vector path only if text found AND --mech flag is True
        - Otherwise uses raster path (pdf2image + appropriate pipeline)
    - Vector files (.dxf, .dwg) → Vector Pipeline (simulation)
    
    Args:
        file_path: Path to the file to parse
        api_keys: Dictionary of API keys (must contain 'google' key for Gemini)
        use_mechanical_drawing: If True, use mechanical drawing pipeline with intelligent PDF triage
        args: Optional argparse.Namespace object (required for PDF processing)
        output_dir: Optional output directory for saving extracted images
        
    Returns:
        Dictionary containing parsed results
        
    Raises:
        ValueError: If file type is unsupported
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = file_path.suffix.lower()
    
    # Route to appropriate pipeline
    if file_ext in ['.jpg', '.jpeg', '.png']:
        # Raster Pipeline - apply image enhancement if needed, then use existing image parser
        print(f"Routing to: Raster Pipeline")
        
        # Apply image enhancement for low-resolution images
        try:
            from od_parse.preprocessing.image_enhancement import enhance_low_res_image
            
            enhanced_image_path = enhance_low_res_image(str(file_path))
            # Use enhanced image if it's different from original (enhancement was applied)
            image_path_to_use = Path(enhanced_image_path) if enhanced_image_path != str(file_path) else file_path
            
            result = _parse_image_file(image_path_to_use, api_keys, use_mechanical_drawing=use_mechanical_drawing, page_number=1)
            
            # Clean up temporary enhanced image if one was created
            if enhanced_image_path != str(file_path):
                try:
                    from od_parse.preprocessing.image_enhancement import cleanup_enhanced_image
                    cleanup_enhanced_image(enhanced_image_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup enhanced image: {e}")
            
            return result
        except ImportError:
            logger.warning("Image enhancement module not available, processing original image")
            return _parse_image_file(file_path, api_keys, use_mechanical_drawing=use_mechanical_drawing, page_number=1)
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}, processing original image")
            return _parse_image_file(file_path, api_keys, use_mechanical_drawing=use_mechanical_drawing, page_number=1)
    
    elif file_ext == '.pdf':
        # PDF Pipeline - use new runPdfPipeline function for intelligent triage
        print(f"Routing to: PDF Pipeline")
        
        # Create args-like object if not provided (for backward compatibility)
        if args is None:
            class SimpleArgs:
                def __init__(self, mech):
                    self.mech = mech
            args = SimpleArgs(use_mechanical_drawing)
        
        return runPdfPipeline(str(file_path), args, output_dir)
    
    elif file_ext in ['.dxf', '.dwg']:
        # Vector Pipeline - simulation
        print(f"Routing to: Vector Pipeline (Simulation)")
        return _parse_vector_file(file_path)
    
    elif file_ext in ['.xlsx', '.xls']:
        # Excel Pipeline
        logger.info("Routing to: Excel Pipeline")
        return runExcelPipeline(str(file_path), args)
    
    elif file_ext == '.docx':
        # Word Document Pipeline
        logger.info("Routing to: Word Document Pipeline")
        # Create args-like object if not provided (for backward compatibility)
        if args is None:
            class SimpleArgs:
                def __init__(self, mech):
                    self.mech = mech
            args = SimpleArgs(use_mechanical_drawing)
        return runDocxPipeline(str(file_path), args, output_dir)
    
    elif file_ext == '.pptx':
        # PowerPoint Presentation Pipeline
        logger.info("Routing to: PowerPoint Presentation Pipeline")
        # Create args-like object if not provided (for backward compatibility)
        if args is None:
            class SimpleArgs:
                def __init__(self, mech):
                    self.mech = mech
            args = SimpleArgs(use_mechanical_drawing)
        return runPptxPipeline(str(file_path), args, output_dir)
    
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported formats: .jpg, .jpeg, .png, .pdf, .dxf, .dwg, .xlsx, .xls, .docx, .pptx")


def _parse_image_file_llm_fallback(image_path: Path, api_keys: dict, page_number: int = 1) -> dict:
    """
    Fallback LLM-based parser for non-mechanical drawing images.
    
    Uses direct Gemini REST API batch processing to avoid 429 rate limit errors.
    This replaces the previous LLMDocumentProcessor approach with a more efficient
    batch processing method that makes a single API call.
    
    Args:
        image_path: Path to the image file
        api_keys: Dictionary of API keys (must contain 'google' key for Gemini)
        page_number: The page number this image corresponds to (for labeling, default: 1)
    """
    from od_parse.mechanical_drawing.gemini_client import process_image_batch
    import os
    
    # Load image using PIL
    try:
        image = Image.open(image_path)
        # Convert to RGB if necessary (some images might be RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
    
    # Get API key
    gemini_api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it in api_keys dict.")
    
    # Build prompt for general document parsing
    prompt = "Please analyze this document image and extract all structured information including text, tables, forms, and any other relevant data."
    
    # Process image using batch processing (single API call)
    try:
        extracted_data = process_image_batch(
            images=[image],
            prompt=prompt,
            api_key=gemini_api_key,
            response_schema=None  # Let the model return flexible JSON structure
        )
        
        # Transform response to match expected format structure
        # Convert text to text_content array format
        extracted_text = extracted_data.get('text', '')
        text_content = _split_text_into_paragraphs(extracted_text) if extracted_text else []
        
        enhanced_data = {
            'text_content': text_content,
            'images': [],
            'tables': extracted_data.get('tables', []),
            'forms': extracted_data.get('forms', []),
            'handwritten_content': extracted_data.get('handwritten_content', []),
            'metadata': {
                'file_name': image_path.name,
                'file_size': image_path.stat().st_size,
                'page_count': 1,
                'page_number': page_number,
                'extraction_method': 'image_batch_processing'
            },
            'extracted_data': extracted_data
        }
        
        return enhanced_data
        
    except Exception as e:
        # If batch processing fails, return error structure
        return {
            'text_content': [],
            'images': [],
            'tables': [],
            'forms': [],
            'handwritten_content': [],
            'metadata': {
                'file_name': image_path.name,
                'file_size': image_path.stat().st_size,
                'page_count': 1,
                'page_number': page_number,
                'extraction_method': 'image_batch_processing',
                'error': str(e)
            }
        }


if __name__ == "__main__":
    sys.exit(main())

