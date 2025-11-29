"""
File routing logic for the intelligent parser.

Extracted and refactored from parse_pdf.py for use in the FastAPI application.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path to import from od_parse module
parent_dir = Path(__file__).parent.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import from existing parse_pdf.py functions
# We'll import the necessary functions from parse_pdf.py
# For now, we'll need to import the pipeline functions directly
from od_parse.excel import runExcelPipeline

# Try to import PyMuPDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


# We need to import functions from parse_pdf.py
# Since parse_pdf.py is in the parent directory, we'll import it
# But we need to be careful about circular imports
# Instead, we'll copy the necessary routing logic here

async def route_and_parse(
    file_path: Path,
    mech_mode: bool = False,
    output_dir: Optional[Path] = None,
    api_keys: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Route a file to the appropriate parsing pipeline and return results.
    
    This is the main routing function that determines which pipeline to use
    based on file extension and processes the file accordingly.
    
    Args:
        file_path: Path to the file to parse
        mech_mode: If True, use mechanical drawing pipeline for images/PDFs
        output_dir: Optional output directory for saving extracted images
        api_keys: Dictionary of API keys (must contain 'google' key for Gemini)
        
    Returns:
        Dictionary containing parsed results
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get API keys from parameter or environment
    if api_keys is None:
        api_keys = {
            "google": os.getenv("GOOGLE_API_KEY"),
            "roboflow": os.getenv("ROBOFLOW_API_KEY")
        }
    
    file_ext = file_path.suffix.lower()
    
    # Create args-like object for compatibility with existing functions
    class SimpleArgs:
        def __init__(self, mech):
            self.mech = mech
    
    args = SimpleArgs(mech_mode)
    
    # Route to appropriate pipeline
    if file_ext in ['.jpg', '.jpeg', '.png']:
        # Raster Pipeline - use image parser
        logger.info(f"Routing to: Raster Pipeline for {file_path.name}")
        return await asyncio.to_thread(
            _parse_image_file,
            file_path,
            api_keys,
            use_mechanical_drawing=mech_mode,
            page_number=1
        )
    
    elif file_ext == '.pdf':
        # PDF Pipeline - use intelligent triage
        logger.info(f"Routing to: PDF Pipeline for {file_path.name}")
        return await asyncio.to_thread(
            _run_pdf_pipeline,
            str(file_path),
            args,
            output_dir
        )
    
    elif file_ext in ['.dxf', '.dwg']:
        # Vector Pipeline - simulation
        logger.info(f"Routing to: Vector Pipeline (Simulation) for {file_path.name}")
        return await asyncio.to_thread(
            _parse_vector_file,
            file_path
        )
    
    elif file_ext in ['.xlsx', '.xls']:
        # Excel Pipeline
        logger.info(f"Routing to: Excel Pipeline for {file_path.name}")
        return await asyncio.to_thread(
            runExcelPipeline,
            str(file_path),
            args
        )
    
    else:
        # Delegate to the master router in parse_pdf.py for additional types
        # such as .docx and .pptx, and to keep supported-format logic in sync.
        logger.info(f"Routing to: Master router in parse_pdf.py for {file_path.name}")
        from parse_pdf import _route_file_by_type as master_route  # local import to avoid cycles
        return await asyncio.to_thread(
            master_route,
            file_path,
            api_keys,
            mech_mode,
            args,
            output_dir,
        )


def _parse_image_file(
    image_path: Path,
    api_keys: dict,
    use_mechanical_drawing: bool = False,
    page_number: int = 1
) -> dict:
    """
    Parse an image file using either the mechanical drawing pipeline or LLM-based parsing.
    
    This function is imported/synced from parse_pdf.py logic.
    """
    # Import here to avoid circular dependencies
    from parse_pdf import _parse_image_file as parse_image_file_impl
    return parse_image_file_impl(image_path, api_keys, use_mechanical_drawing, page_number)


def _run_pdf_pipeline(
    pdf_path: str,
    args,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Run PDF pipeline with intelligent triage.
    
    This function is imported/synced from parse_pdf.py logic.
    """
    # Import here to avoid circular dependencies
    from parse_pdf import runPdfPipeline
    return runPdfPipeline(pdf_path, args, output_dir)


def _parse_vector_file(vector_path: Path) -> dict:
    """
    Parse a vector file (DXF, DWG).
    
    This function is imported/synced from parse_pdf.py logic.
    """
    # Import here to avoid circular dependencies
    from parse_pdf import _parse_vector_file as parse_vector_file_impl
    return parse_vector_file_impl(vector_path)

