#!/usr/bin/env python3
"""
Minimal script to parse a PDF file or image file using odparse with Gemini VLM.
"""

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
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
        description='Parse a PDF file or image file (PNG, JPG, JPEG) using odparse with Gemini VLM'
    )
    parser.add_argument('input_file', help='Path to the PDF or image file to parse')
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
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
    if file_ext not in supported_extensions:
        print(f"Error: Unsupported file type: {file_ext}", file=sys.stderr)
        print(f"Supported formats: {', '.join(supported_extensions)}", file=sys.stderr)
        return 1
    
    is_image = file_ext in ['.png', '.jpg', '.jpeg']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get API keys from environment
    api_keys = {
        "google": os.getenv("GOOGLE_API_KEY")
    }
    
    if not api_keys["google"]:
        print("Error: GOOGLE_API_KEY not found in environment variables", file=sys.stderr)
        print("Please set GOOGLE_API_KEY in your .env file or environment", file=sys.stderr)
        return 1
    
    # Process file based on type
    try:
        if is_image:
            # Process image file
            print(f"Parsing image: {input_path.name}")
            result = _parse_image_file(input_path, api_keys, use_mechanical_drawing=args.mech)
        else:
            # Process PDF file
            print(f"Parsing PDF: {input_path.name}")
            result = parse_pdf(
                input_path,
                output_format="raw",
                llm_model="gemini-2.0-flash",  # Gemini VLM model with vision support
                api_keys=api_keys,
                require_llm=True,
                use_deep_learning=True
            )
        
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
        file_type = "image" if is_image else "PDF"
        print(f"Error parsing {file_type}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


def _parse_image_file(image_path: Path, api_keys: dict, use_mechanical_drawing: bool = False) -> dict:
    """
    Parse an image file using either the mechanical drawing pipeline or LLM-based parsing.
    
    Args:
        image_path: Path to the image file
        api_keys: Dictionary of API keys (must contain 'google' key for Gemini, 'roboflow' is optional)
        use_mechanical_drawing: If True, use the hybrid three-stage mechanical drawing pipeline.
                               If False (default), use LLM-based parsing.
    
    Returns:
        Dictionary containing parsed results in the format compatible with existing code
    """
    # Validate image file exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # If --mech flag is not set, use LLM fallback directly
    if not use_mechanical_drawing:
        return _parse_image_file_llm_fallback(image_path, api_keys)
    
    # Use mechanical drawing pipeline when --mech flag is set
    from od_parse.mechanical_drawing.pipeline import run_full_parsing_pipeline_sync
    import os
    
    # Extract API keys
    gemini_api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it in api_keys dict.")
    
    # Roboflow API key is optional (has default)
    roboflow_api_key = api_keys.get("roboflow") or os.getenv("ROBOFLOW_API_KEY")
    
    # Run the hybrid three-stage pipeline
    try:
        pipeline_result = run_full_parsing_pipeline_sync(
            image_path,
            roboflow_api_key=roboflow_api_key,
            gemini_api_key=gemini_api_key
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
        return _parse_image_file_llm_fallback(image_path, api_keys)


def _parse_image_file_llm_fallback(image_path: Path, api_keys: dict) -> dict:
    """
    Fallback LLM-based parser for non-mechanical drawing images.
    
    This is the original LLM-based implementation, kept as fallback.
    """
    from od_parse.llm import LLMDocumentProcessor
    
    # Load image using PIL
    try:
        image = Image.open(image_path)
        # Convert to RGB if necessary (some images might be RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
    
    # Create minimal parsed_data structure
    parsed_data = {
        'text': '',  # Empty text, LLM will extract from image
        'images': [],
        'tables': [],
        'forms': [],
        'handwritten_content': [],
        'metadata': {
            'file_name': image_path.name,
            'file_size': image_path.stat().st_size,
            'page_count': 1,
            'extraction_method': 'image_llm_vision'
        }
    }
    
    # Initialize LLM processor
    processor = LLMDocumentProcessor(
        model_id="gemini-2.0-flash",  # Gemini VLM model with vision support
        api_keys=api_keys
    )
    
    # Process with LLM (pass image directly)
    enhanced_data = processor.process_document(parsed_data, [image])
    
    return enhanced_data


if __name__ == "__main__":
    sys.exit(main())

