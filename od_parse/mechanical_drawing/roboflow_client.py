"""
Roboflow REST API Client for Mechanical Drawing Parser

This module provides helper functions for making API calls to the local Roboflow
inference server for annotation detection.

Supports two modes:
1. Local inference server (http://localhost:9001) - requires inference-cli
2. Roboflow Python SDK - direct API calls (fallback if server unavailable)
"""

import json
import requests
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from PIL import Image

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Roboflow Configuration
ROBOFLOW_SERVER_URL = "http://localhost:9001"
PROJECT_ID = "eng-drawing-ukrvj"
MODEL_VERSION = 3
DEFAULT_API_KEY = "ub3sg9EEXSEZhVGZL4JD"
DEFAULT_CONFIDENCE = 0.05


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """
    Get the dimensions (width, height) of an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Tuple of (width, height) in pixels
    """
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        logger.error(f"Failed to get image dimensions: {e}")
        raise


def call_roboflow_detection(
    image_path: Path,
    api_key: str = DEFAULT_API_KEY,
    confidence: float = DEFAULT_CONFIDENCE,
    use_sdk: bool = False
) -> Dict[str, Any]:
    """
    Call Roboflow to detect annotations.
    
    Supports two modes:
    1. Local inference server (default) - requires inference-cli running
    2. Roboflow Python SDK (fallback) - direct API calls
    
    Args:
        image_path: Path to the image file
        api_key: Roboflow API key (default: ub3sg9EEXSEZhVGZL4JD)
        confidence: Confidence threshold (default: 0.05 for 5%)
        use_sdk: If True, use Python SDK instead of local server
    
    Returns:
        Dictionary containing:
        - predictions: List of detection objects with pixel coordinates
        - originalWidth: Image width in pixels
        - originalHeight: Image height in pixels
    """
    # Get image dimensions first
    try:
        original_width, original_height = get_image_dimensions(image_path)
    except Exception as e:
        logger.error(f"Failed to get image dimensions: {e}")
        return {
            'predictions': [],
            'originalWidth': 0,
            'originalHeight': 0
        }
    
    # Try Python SDK first if requested, or as fallback
    if use_sdk:
        return _call_roboflow_sdk(image_path, api_key, confidence, original_width, original_height)
    
    # Try local inference server first
    logger.info("Calling Roboflow inference server for detection...")
    
    # Construct the inference URL
    # Try standard /infer endpoint first (common inference server pattern)
    infer_url = f"{ROBOFLOW_SERVER_URL}/infer"
    
    # Prepare parameters - project_id, model_version, api_key as query params
    params = {
        'api_key': api_key,
        'project_id': PROJECT_ID,
        'model_version': MODEL_VERSION,
        'confidence': confidence
    }
    
    # Prepare the image file for upload
    try:
        with open(image_path, 'rb') as image_file:
            files = {
                'image': (image_path.name, image_file, 'image/jpeg' if image_path.suffix.lower() in ['.jpg', '.jpeg'] else 'image/png')
            }
            
            # Make POST request to Roboflow server
            response = requests.post(
                infer_url,
                params=params,
                files=files,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
    except requests.exceptions.ConnectionError:
        logger.warning("Failed to connect to Roboflow inference server.")
        logger.warning("💡 To start the server, run: pip install inference-cli && inference server start")
        logger.warning("💡 Note: inference-cli requires Python <3.13. If you have Python 3.13+, falling back to Roboflow SDK...")
        # Fallback to SDK
        return _call_roboflow_sdk(image_path, api_key, confidence, original_width, original_height)
    except requests.exceptions.HTTPError as e:
        # If /infer endpoint doesn't work, try alternative format
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
            logger.warning("Standard /infer endpoint not found, trying alternative format...")
            # Try alternative: POST to /{project_id}/{version}
            try:
                alt_url = f"{ROBOFLOW_SERVER_URL}/{PROJECT_ID}/{MODEL_VERSION}"
                with open(image_path, 'rb') as image_file:
                    files = {
                        'file': (image_path.name, image_file, 'image/jpeg' if image_path.suffix.lower() in ['.jpg', '.jpeg'] else 'image/png')
                    }
                    alt_params = {
                        'api_key': api_key,
                        'confidence': confidence
                    }
                    response = requests.post(alt_url, params=alt_params, files=files, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    logger.info("Successfully connected using alternative endpoint format.")
            except Exception as alt_e:
                logger.warning(f"Alternative endpoint also failed: {alt_e}")
                logger.warning("Falling back to Roboflow SDK...")
                return _call_roboflow_sdk(image_path, api_key, confidence, original_width, original_height)
        else:
            logger.warning(f"Roboflow server HTTP error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.warning(f"Response status: {e.response.status_code}, Response text: {e.response.text[:200]}")
            logger.warning("Falling back to Roboflow SDK...")
            return _call_roboflow_sdk(image_path, api_key, confidence, original_width, original_height)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Roboflow server error: {e}, falling back to SDK...")
        return _call_roboflow_sdk(image_path, api_key, confidence, original_width, original_height)
    except Exception as e:
        logger.warning(f"Unexpected error calling Roboflow server: {e}, falling back to SDK...")
        return _call_roboflow_sdk(image_path, api_key, confidence, original_width, original_height)
    
    # Process Roboflow predictions
    # Roboflow returns predictions with normalized coordinates (0.0-1.0)
    # We need to convert them to pixel coordinates
    predictions = []
    
    if 'predictions' in result:
        for pred in result['predictions']:
            # Roboflow prediction format: x, y are center points, width and height are normalized
            x_center_normalized = pred.get('x', 0)
            y_center_normalized = pred.get('y', 0)
            width_normalized = pred.get('width', 0)
            height_normalized = pred.get('height', 0)
            
            # Convert from normalized (0.0-1.0) to pixel coordinates
            x_center = x_center_normalized * original_width
            y_center = y_center_normalized * original_height
            w = width_normalized * original_width
            h = height_normalized * original_height
            
            # Convert from center-point to top-left corner
            x_tl = x_center - (w / 2)
            y_tl = y_center - (h / 2)
            
            # Ensure coordinates are within image bounds
            x_tl = max(0, min(x_tl, original_width))
            y_tl = max(0, min(y_tl, original_height))
            w = max(1, min(w, original_width - x_tl))
            h = max(1, min(h, original_height - y_tl))
            
            predictions.append({
                'x': int(x_tl),
                'y': int(y_tl),
                'width': int(w),
                'height': int(h),
                'class': pred.get('class', 'unknown'),
                'confidence': pred.get('confidence', 0.0)
            })
    
    logger.info(f"Roboflow detected {len(predictions)} potential annotations.")
    
    return {
        'predictions': predictions,
        'originalWidth': original_width,
        'originalHeight': original_height
    }


def _call_roboflow_sdk(
    image_path: Path,
    api_key: str,
    confidence: float,
    original_width: int,
    original_height: int
) -> Dict[str, Any]:
    """
    Fallback: Use Roboflow Python SDK for inference.
    
    This works without requiring the inference-cli or local server.
    Makes direct API calls to Roboflow cloud.
    
    Args:
        image_path: Path to the image file
        api_key: Roboflow API key
        confidence: Confidence threshold
        original_width: Image width
        original_height: Image height
    
    Returns:
        Dictionary with predictions and dimensions
    """
    logger.info("Using Roboflow Python SDK (cloud API) for detection...")
    
    try:
        # Try to import roboflow SDK
        try:
            from roboflow import Roboflow
        except ImportError:
            logger.error("Roboflow Python SDK not installed.")
            logger.error("💡 Install it with: pip install roboflow")
            logger.error("💡 Note: This requires Python <3.13. For Python 3.13+, you need to:")
            logger.error("   1. Use Python 3.12 or lower, OR")
            logger.error("   2. Use Docker to run the inference server")
            return {
                'predictions': [],
                'originalWidth': original_width,
                'originalHeight': original_height
            }
        
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(PROJECT_ID)
        model = project.version(MODEL_VERSION).model
        
        # Run inference
        result = model.predict(str(image_path), confidence=confidence)
        
        # Process predictions
        predictions = []
        if hasattr(result, 'json') and 'predictions' in result.json():
            for pred in result.json()['predictions']:
                # Roboflow SDK returns predictions with normalized coordinates
                x_center_normalized = pred.get('x', 0) / original_width if 'x' in pred else pred.get('x', 0)
                y_center_normalized = pred.get('y', 0) / original_height if 'y' in pred else pred.get('y', 0)
                width_normalized = pred.get('width', 0) / original_width if 'width' in pred else pred.get('width', 0)
                height_normalized = pred.get('height', 0) / original_height if 'height' in pred else pred.get('height', 0)
                
                # If coordinates are already in pixels (not normalized), use them directly
                if x_center_normalized > 1.0 or y_center_normalized > 1.0:
                    # Already in pixels
                    x_center = x_center_normalized
                    y_center = y_center_normalized
                    w = width_normalized
                    h = height_normalized
                else:
                    # Normalized, convert to pixels
                    x_center = x_center_normalized * original_width
                    y_center = y_center_normalized * original_height
                    w = width_normalized * original_width
                    h = height_normalized * original_height
                
                # Convert from center-point to top-left corner
                x_tl = x_center - (w / 2)
                y_tl = y_center - (h / 2)
                
                # Ensure coordinates are within image bounds
                x_tl = max(0, min(x_tl, original_width))
                y_tl = max(0, min(y_tl, original_height))
                w = max(1, min(w, original_width - x_tl))
                h = max(1, min(h, original_height - y_tl))
                
                predictions.append({
                    'x': int(x_tl),
                    'y': int(y_tl),
                    'width': int(w),
                    'height': int(h),
                    'class': pred.get('class', 'unknown'),
                    'confidence': pred.get('confidence', 0.0)
                })
        
        logger.info(f"Roboflow SDK detected {len(predictions)} potential annotations.")
        
        return {
            'predictions': predictions,
            'originalWidth': original_width,
            'originalHeight': original_height
        }
        
    except Exception as e:
        logger.error(f"Error using Roboflow SDK: {e}")
        logger.error("💡 Solutions:")
        logger.error("   1. Install: pip install roboflow (requires Python <3.13)")
        logger.error("   2. Use Python 3.12 or lower")
        logger.error("   3. Use Docker to run inference server")
        return {
            'predictions': [],
            'originalWidth': original_width,
            'originalHeight': original_height
        }

