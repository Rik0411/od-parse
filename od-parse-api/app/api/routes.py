"""
FastAPI route handlers for the parsing API.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import get_settings_dep as get_settings, get_temp_dir, get_api_keys
from app.core.config import Settings
from app.core.router import route_and_parse

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@router.get("/info")
async def get_info(settings: Settings = Depends(get_settings)):
    """Get API information and supported file types."""
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "supported_extensions": settings.allowed_extensions,
        "max_file_size_mb": settings.max_file_size_mb
    }


@router.post("/parse")
async def parse_file(
    file: UploadFile = File(...),
    mech_mode: bool = Query(False, description="Use mechanical drawing pipeline"),
    settings: Settings = Depends(get_settings),
    temp_dir: Path = Depends(get_temp_dir),
    api_keys: Dict[str, str] = Depends(get_api_keys)
) -> Dict[str, Any]:
    """
    Parse an uploaded file using the intelligent parser.
    
    Args:
        file: Uploaded file to parse
        mech_mode: If True, use mechanical drawing pipeline for images/PDFs
        settings: Application settings
        temp_dir: Temporary directory for file storage
        api_keys: API keys dictionary
        
    Returns:
        Parsed result dictionary
        
    Raises:
        HTTPException: For validation errors or processing failures
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported formats: {', '.join(settings.allowed_extensions)}"
        )
    
    # Validate file size
    file_content = await file.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size_mb:.2f}MB. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    # Create temporary file
    temp_file_path = None
    try:
        # Create unique filename
        unique_id = str(uuid.uuid4())
        temp_file_path = temp_dir / f"{unique_id}{file_ext}"
        
        # Ensure temp directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Write uploaded file to temporary location
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Create output directory for extracted images
        output_dir = temp_dir / f"{unique_id}_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Route and parse the file
        try:
            result = await route_and_parse(
                file_path=temp_file_path,
                mech_mode=mech_mode,
                output_dir=output_dir,
                api_keys=api_keys
            )
            
            return result
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            # Log the full error for debugging
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error processing file: {error_trace}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file: {str(e)}"
            )
    
    finally:
        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {temp_file_path}: {e}")
        
        # Clean up output directory (optional - you might want to keep images)
        # if output_dir and output_dir.exists():
        #     try:
        #         import shutil
        #         shutil.rmtree(output_dir)
        #     except Exception as e:
        #         print(f"Warning: Failed to delete output directory {output_dir}: {e}")

