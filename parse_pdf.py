#!/usr/bin/env python3
"""
Minimal script to parse a PDF file using odparse with Gemini VLM.
"""

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

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
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Import from main module to get LLM-enhanced parser
from od_parse.main import parse_pdf

def main():
    parser = argparse.ArgumentParser(description='Parse a PDF file using odparse with Gemini VLM')
    parser.add_argument('pdf_file', help='Path to the PDF file to parse')
    parser.add_argument('--output-dir', '-o', default='output', help='Output directory for JSON file (default: output)')
    
    args = parser.parse_args()
    
    # Validate input file
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        return 1
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"Error: File is not a PDF: {pdf_path}", file=sys.stderr)
        return 1
    
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
    
    # Parse PDF with Gemini VLM
    print(f"Parsing PDF: {pdf_path.name}")
    try:
        result = parse_pdf(
            pdf_path,
            output_format="raw",
            llm_model="gemini-2.0-flash",  # Gemini VLM model with vision support
            api_keys=api_keys,
            require_llm=True,
            use_deep_learning=True
        )
        
        # Write output to JSON file
        output_file = output_dir / f"{pdf_path.stem}.json"
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
        print(f"Error parsing PDF: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

