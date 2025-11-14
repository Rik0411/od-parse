#!/usr/bin/env python3
"""
Intelligent file parser that automatically routes different file types to appropriate parsing pipelines.

This script supports multiple file formats:
- Raster images (.jpg, .jpeg, .png) → Raster Pipeline (hybrid Roboflow + Gemini)
- PDF files (.pdf) → PDF Pipeline (converts to image, then Raster Pipeline)
- Vector files (.dxf, .dwg) → Vector Pipeline (simulation)

The master controller (_route_file_by_type) automatically detects file type and delegates
to the correct pipeline, making this an intelligent, multi-format parser.
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import List
from PIL import Image

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

def main():
    parser = argparse.ArgumentParser(
        description='Intelligent file parser supporting PDF, images (PNG, JPG, JPEG), and vector files (DXF, DWG) using odparse with Gemini VLM'
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
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.dxf', '.dwg']
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
    
    # Google API key is only required for raster/PDF pipelines, not vector
    if file_ext not in ['.dxf', '.dwg'] and not api_keys["google"]:
        print("Error: GOOGLE_API_KEY not found in environment variables", file=sys.stderr)
        print("Please set GOOGLE_API_KEY in your .env file or environment", file=sys.stderr)
        return 1
    
    # Process file using intelligent router
    try:
        print(f"Parsing file: {input_path.name}")
        result = _route_file_by_type(input_path, api_keys, use_mechanical_drawing=args.mech)
        
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
        # (maintaining compatibility with existing code that expects llm_analysis structure)
        enhanced_data = {
            'text': '',  # No text extraction in mechanical drawing pipeline
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
            'llm_analysis': {
                'extracted_data': pipeline_result,
                'model_info': {
                    'provider': 'hybrid',
                    'model': 'roboflow + gemini-2.0-flash',
                    'tokens_used': 0,
                    'cost_estimate': 0
                },
                'processing_success': True
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
            }
        }
        
        return enhanced_data
        
    except Exception as e:
        # Fallback to LLM if pipeline fails (for non-mechanical drawings)
        print(f"Warning: Mechanical drawing pipeline failed: {e}")
        print("Falling back to LLM-based parsing...")
        return _parse_image_file_llm_fallback(image_path, api_keys, page_number)


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


def _route_file_by_type(file_path: Path, api_keys: dict, use_mechanical_drawing: bool = False) -> dict:
    """
    Master controller that routes files to the appropriate parsing pipeline.
    
    This intelligent router detects the file type by extension and delegates
    to the correct pipeline:
    - Raster files (.jpg, .png, .jpeg) → Raster Pipeline
    - PDF files (.pdf) → PDF Pipeline (converts to image, then Raster Pipeline)
    - Vector files (.dxf, .dwg) → Vector Pipeline (simulation)
    
    Args:
        file_path: Path to the file to parse
        api_keys: Dictionary of API keys (must contain 'google' key for Gemini)
        use_mechanical_drawing: If True, use mechanical drawing pipeline for images
        
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
        # Raster Pipeline - use existing image parser (single image, page 1)
        print(f"Routing to: Raster Pipeline")
        return _parse_image_file(file_path, api_keys, use_mechanical_drawing=use_mechanical_drawing, page_number=1)
    
    elif file_ext == '.pdf':
        # PDF Pipeline - convert ALL pages to images, then process each page
        print(f"Routing to: PDF Pipeline")
        temp_image_paths = []
        aggregated_result = None
        
        try:
            # Convert ALL pages of PDF to PNG images
            temp_image_paths = _convert_pdf_to_image(file_path)
            total_pages = len(temp_image_paths)
            print(f"PDF converted to {total_pages} image(s)")
            
            # Initialize aggregated result structure
            if use_mechanical_drawing:
                # For mechanical pipeline, aggregate by category
                aggregated_result = {
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
                        'source_file': str(file_path),
                        'source_type': 'pdf',
                        'extraction_method': 'pdf_to_raster_pipeline',
                        'total_pages': total_pages
                    }
                }
            else:
                # For LLM fallback, aggregate text, tables, forms, etc.
                aggregated_result = {
                    'text': '',
                    'images': [],
                    'tables': [],
                    'forms': [],
                    'handwritten_content': [],
                    'metadata': {
                        'source_file': str(file_path),
                        'source_type': 'pdf',
                        'extraction_method': 'pdf_to_raster_pipeline',
                        'total_pages': total_pages,
                        'page_count': total_pages
                    },
                    'llm_analysis': {
                        'extracted_data': {},
                        'model_info': {
                            'provider': 'google',
                            'model': 'gemini-2.5-flash-preview-09-2025',
                            'tokens_used': 0,
                            'cost_estimate': 0
                        },
                        'processing_success': True
                    }
                }
            
            # Process each page
            for page_num, temp_image_path in enumerate(temp_image_paths, start=1):
                try:
                    print(f"Processing page {page_num}/{total_pages}...")
                    page_result = _parse_image_file(
                        temp_image_path, 
                        api_keys, 
                        use_mechanical_drawing=use_mechanical_drawing,
                        page_number=page_num
                    )
                    
                    # Aggregate results based on pipeline type
                    if use_mechanical_drawing:
                        # Mechanical pipeline: merge category arrays
                        if 'llm_analysis' in page_result and 'extracted_data' in page_result['llm_analysis']:
                            extracted = page_result['llm_analysis']['extracted_data']
                            for category in ['Measures', 'Radii', 'Views', 'GD_T', 'Materials', 
                                           'Notes', 'Threads', 'SurfaceRoughness', 'GeneralTolerances', 
                                           'TitleBlock', 'Other', '_Errors', '_FalsePositives']:
                                if category in extracted:
                                    aggregated_result[category].extend(extracted[category])
                    else:
                        # LLM fallback: merge text, tables, forms
                        if 'text' in page_result:
                            if aggregated_result['text']:
                                aggregated_result['text'] += '\n\n--- Page ' + str(page_num) + ' ---\n\n'
                            aggregated_result['text'] += page_result.get('text', '')
                        if 'tables' in page_result:
                            aggregated_result['tables'].extend(page_result.get('tables', []))
                        if 'forms' in page_result:
                            aggregated_result['forms'].extend(page_result.get('forms', []))
                        if 'handwritten_content' in page_result:
                            aggregated_result['handwritten_content'].extend(page_result.get('handwritten_content', []))
                    
                except Exception as page_error:
                    print(f"Warning: Failed to process page {page_num}: {page_error}", file=sys.stderr)
                    # Continue processing other pages
                    continue
            
            return aggregated_result
            
        finally:
            # Clean up all temporary files
            for temp_path in temp_image_paths:
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception as e:
                        print(f"Warning: Failed to delete temporary file {temp_path}: {e}", file=sys.stderr)
    
    elif file_ext in ['.dxf', '.dwg']:
        # Vector Pipeline - simulation
        print(f"Routing to: Vector Pipeline (Simulation)")
        return _parse_vector_file(file_path)
    
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported formats: .jpg, .jpeg, .png, .pdf, .dxf, .dwg")


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
        # (maintaining compatibility with existing code that expects llm_analysis structure)
        enhanced_data = {
            'text': extracted_data.get('text', ''),
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
            'llm_analysis': {
                'extracted_data': extracted_data,
                'model_info': {
                    'provider': 'google',
                    'model': 'gemini-2.5-flash-preview-09-2025',
                    'tokens_used': 0,
                    'cost_estimate': 0
                },
                'processing_success': True
            }
        }
        
        return enhanced_data
        
    except Exception as e:
        # If batch processing fails, return error structure
        return {
            'text': '',
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
            },
            'llm_analysis': {
                'extracted_data': {},
                'error': str(e),
                'processing_success': False
            }
        }


if __name__ == "__main__":
    sys.exit(main())

