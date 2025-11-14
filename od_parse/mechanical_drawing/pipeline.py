"""
Hybrid Pipeline for Mechanical Drawing Parsing

This module implements a two-stage hybrid pipeline architecture with batch processing
to prevent API rate limiting (429 errors):
- Stage 1: Detection (Roboflow local server) - detects all potential annotations
- Stage 2: Verification & Parsing (Gemini multimodal API) - verifies and parses all patches in a single batch API call

Based on the research paper "Automated Parsing of Engineering Drawings..."
"""

import asyncio
import io
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image

from od_parse.mechanical_drawing.gemini_client import (
    image_to_base64,
    call_gemini_multimodal,
    stage2_run_batch_verification
)
from od_parse.mechanical_drawing.roboflow_client import call_roboflow_detection
from od_parse.mechanical_drawing.image_utils import crop_image_patch
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


async def stage1_run_roboflow_detection(image_path: Path, roboflow_api_key: str) -> Dict[str, Any]:
    """
    STAGE 1 (Roboflow Detection)
    
    Uses local Roboflow inference server to detect all potential annotations.
    This specialized detector (like YOLO) finds annotations and returns their
    class and bounding boxes.
    
    Args:
        image_path: Path to the mechanical drawing image
        roboflow_api_key: Roboflow API key
    
    Returns:
        Dictionary containing:
        - predictions: List of detection objects with pixel coordinates, class, confidence
        - originalWidth: Image width in pixels
        - originalHeight: Image height in pixels
    """
    logger.info("Stage 1: Detecting annotations via Roboflow...")
    
    try:
        # Call Roboflow detection (synchronous call, run in thread)
        result = await asyncio.to_thread(
            call_roboflow_detection,
            image_path,
            roboflow_api_key,
            0.05  # 5% confidence threshold
        )
        
        predictions = result.get('predictions', [])
        logger.info(f"Stage 1: Detected {len(predictions)} potential annotations.")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Stage 1 (Roboflow Detection): {e}")
        return {
            'predictions': [],
            'originalWidth': 0,
            'originalHeight': 0
        }


async def run_full_hybrid_pipeline(
    image_path: Path,
    roboflow_api_key: str,
    gemini_api_key: str,
    page_number: int = 1
) -> Dict[str, Any]:
    """
    MAIN HYBRID PARSING PIPELINE (Batch Processing - Rate-Limit Safe)
    
    Orchestrates a two-stage hybrid pipeline with batch processing to prevent 429 errors:
    1. Stage 1: Roboflow detection (local server) - detects all potential annotations
    2. Stage 2: Gemini verification & parsing (batch processing) - verifies and parses all patches in a single API call
    
    Args:
        image_path: Path to the mechanical drawing image
        roboflow_api_key: Roboflow API key
        gemini_api_key: Google API key for Gemini
        page_number: The page number this image corresponds to (for labeling, default: 1)
    
    Returns:
        Final structured JSON dictionary organized by category
    """
    logger.info(f"Starting 2-Stage HYBRID pipeline (Page {page_number}, BATCHED, Rate-Limit Safe)...")
    start_time = time.time()
    
    try:
        # --- STAGE 1: ROBOFLOW DETECTIONS ---
        logger.info("=" * 60)
        logger.info("STAGE 1: Annotation Detection (Roboflow)")
        logger.info("=" * 60)
        roboflow_result = await stage1_run_roboflow_detection(image_path, roboflow_api_key)
        
        predictions = roboflow_result.get('predictions', [])
        original_width = roboflow_result.get('originalWidth', 0)
        original_height = roboflow_result.get('originalHeight', 0)
        
        if not predictions or len(predictions) == 0:
            logger.warning("Stage 1 failed to find any annotations.")
            return {
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
                    'processing_time': time.time() - start_time,
                    'annotations_detected': 0,
                    'annotations_verified': 0,
                    'pipeline_stages': 2,
                    'stage1_detections': 0,
                    'stage2_verified': 0,
                    'page_number': page_number
                }
            }
        
        # --- PREPARE PATCHES FOR BATCHING ---
        logger.info("=" * 60)
        logger.info(f"STAGE 2: Preparing {len(predictions)} patches for batch processing...")
        logger.info("=" * 60)
        
        patches_to_process: List[Tuple[bytes, str]] = []
        
        # Load image once for all crops
        with Image.open(image_path) as img:
            # Ensure we're working with RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            for pred in predictions:
                try:
                    # Add margin to crop for better context
                    margin = 10  # 10px margin
                    crop_x = max(0, pred['x'] - margin)
                    crop_y = max(0, pred['y'] - margin)
                    crop_w = min(original_width - crop_x, pred['width'] + (margin * 2))
                    crop_h = min(original_height - crop_y, pred['height'] + (margin * 2))
                    
                    # Crop box is (left, upper, right, lower)
                    box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
                    patch_img = img.crop(box)
                    
                    # Convert PIL Image to bytes
                    with io.BytesIO() as output:
                        patch_img.save(output, format="PNG")
                        patch_bytes = output.getvalue()
                    
                    patches_to_process.append((patch_bytes, pred['class']))
                    
                except Exception as crop_error:
                    logger.error(f"Error cropping patch for class {pred.get('class', 'unknown')}: {crop_error}")
                    # Add empty bytes as placeholder - batch function will handle it
                    patches_to_process.append((b'', pred.get('class', 'unknown')))
        
        # --- STAGE 2: GEMINI BATCH VERIFICATION & PARSING (ONE CALL) ---
        logger.info("=" * 60)
        logger.info(f"STAGE 2: Verification & Parsing (Gemini) - Processing {len(patches_to_process)} patches in ONE batch")
        logger.info("=" * 60)
        
        # Make single batch API call
        verified_results = await asyncio.to_thread(
            stage2_run_batch_verification,
            patches_to_process,
            gemini_api_key
        )
        
        if not verified_results:
            logger.error("Stage 2 (Gemini Batch) returned no results.")
            verified_results = []
        
        # Ensure we have the same number of results as predictions
        if len(verified_results) != len(predictions):
            logger.warning(f"Batch result count mismatch: {len(verified_results)} results for {len(predictions)} predictions")
            # Pad with None if needed
            while len(verified_results) < len(predictions):
                verified_results.append(None)
            # Truncate if too many
            verified_results = verified_results[:len(predictions)]
        
        # Add source prediction info to each result
        for i, result in enumerate(verified_results):
            if result is not None and isinstance(result, dict):
                result['_source'] = predictions[i] if i < len(predictions) else {}
            elif result is None:
                # False positive - create placeholder
                verified_results[i] = {
                    'type': None,
                    '_source': predictions[i] if i < len(predictions) else {},
                    '_false_positive': True
                }
        
        # --- AGGREGATION ---
        logger.info("Aggregating verified results by category...")
        
        final_json = {
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
            '_FalsePositives': []
        }
        
        # Categorize results
        for result in verified_results:
            # Handle false positives and errors
            if result.get('_false_positive'):
                final_json['_FalsePositives'].append(result)
                continue
            
            if result.get('type') == 'Error':
                final_json['_Errors'].append(result)
                continue
            
            # Skip None or invalid results
            if result is None or not result.get('type'):
                final_json['_FalsePositives'].append(result)
                continue
            
            result_type = result.get('type', '')
            
            # Add source info for tracking (remove internal fields)
            clean_result = {k: v for k, v in result.items() if k not in ['_source', '_false_positive', 'error']}
            clean_result['_source'] = result.get('_source', {})
            
            # Categorize by type
            if result_type == 'LinearDimension':
                final_json['Measures'].append(clean_result)
            elif result_type == 'Radius':
                final_json['Radii'].append(clean_result)
            elif result_type in ['ViewLabel', 'ViewCallout']:
                final_json['Views'].append(clean_result)
            elif result_type == 'GD_T':
                final_json['GD_T'].append(clean_result)
            elif result_type == 'Material':
                final_json['Materials'].append(clean_result)
            elif result_type == 'Note':
                final_json['Notes'].append(clean_result)
            elif result_type == 'Thread':
                final_json['Threads'].append(clean_result)
            elif result_type == 'SurfaceRoughness':
                final_json['SurfaceRoughness'].append(clean_result)
            elif result_type == 'GeneralTolerance':
                final_json['GeneralTolerances'].append(clean_result)
            elif result_type == 'TitleBlock':
                final_json['TitleBlock'].append(clean_result)
            else:
                final_json['Other'].append(clean_result)
        
        # Add metadata
        processing_time = time.time() - start_time
        verified_count = sum(1 for r in verified_results if r is not None and r.get('type') and not r.get('_false_positive') and r.get('type') != 'Error')
        final_json['metadata'] = {
            'processing_time': processing_time,
            'annotations_detected': len(predictions),
            'annotations_verified': verified_count,
            'pipeline_stages': 2,
            'stage1_detections': len(predictions),
            'stage2_verified': verified_count,
            'stage2_false_positives': len(final_json['_FalsePositives']),
            'stage2_errors': len(final_json['_Errors']),
            'page_number': page_number
        }
        
        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info(f"Processing time: {processing_time:.3f}s")
        logger.info(f"Detected: {len(predictions)} annotations")
        logger.info(f"Verified: {verified_count} annotations")
        logger.info(f"False positives: {len(final_json['_FalsePositives'])}")
        logger.info(f"Errors: {len(final_json['_Errors'])}")
        logger.info("=" * 60)
        
        return final_json
        
    except Exception as e:
        logger.error(f"Error in main hybrid pipeline: {e}")
        # Return empty structure on error
        return {
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
                'processing_time': time.time() - start_time,
                'error': str(e),
                'page_number': page_number
            }
        }


def run_full_hybrid_pipeline_sync(
    image_path: Path,
    roboflow_api_key: str,
    gemini_api_key: str,
    page_number: int = 1
) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_full_hybrid_pipeline.
    
    For compatibility with existing code that doesn't use async/await.
    
    Args:
        image_path: Path to the mechanical drawing image
        roboflow_api_key: Roboflow API key
        gemini_api_key: Google API key for Gemini
        page_number: The page number this image corresponds to (for labeling, default: 1)
    
    Returns:
        Final structured JSON dictionary organized by category
    """
    return asyncio.run(run_full_hybrid_pipeline(image_path, roboflow_api_key, gemini_api_key, page_number))


async def run_full_parsing_pipeline(image_path: Path, api_key: str) -> Dict[str, Any]:
    """
    MAIN PARSING PIPELINE
    
    Orchestrates the complete two-stage pipeline:
    1. Stage 1: Detect all annotations (Gemini multimodal API)
    2. Stage 2: Parse each annotation (Gemini text API with JSON schema, in parallel)
    3. Aggregate results into final structured JSON
    
    This replaces the old unoptimized parser function.
    
    Args:
        image_path: Path to the mechanical drawing image
        api_key: Google API key for Gemini
    
    Returns:
        Final structured JSON dictionary organized by category
    """
    logger.info("Starting full parsing pipeline...")
    start_time = time.time()
    
    # --- STAGE 1: Detection ---
    logger.info("=" * 60)
    logger.info("STAGE 1: Annotation Detection (Gemini Vision)")
    logger.info("=" * 60)
    all_annotations = await stage1_run_ocr_detection(image_path, api_key)
    
    if not all_annotations or len(all_annotations) == 0:
        logger.warning("Stage 1 failed to find any annotations.")
        return {
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
            'metadata': {
                'processing_time': time.time() - start_time,
                'annotations_detected': 0
            }
        }
    
    # --- STAGE 2: Parsing (in parallel) ---
    logger.info("=" * 60)
    logger.info(f"STAGE 2: Annotation Parsing (Gemini Text API) - Processing {len(all_annotations)} annotations in parallel")
    logger.info("=" * 60)
    
    # Process all annotations in parallel
    parsing_tasks = [stage2_run_llm_parsing(annotation, api_key) for annotation in all_annotations]
    parsed_results = await asyncio.gather(*parsing_tasks)
    
    # --- AGGREGATION ---
    logger.info("Aggregating results by category...")
    
    # Initialize final JSON structure (matching paper's output format)
    final_json = {
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
        'Other': []
    }
    
    # Categorize and aggregate results based on parsed type field
    for result in parsed_results:
        if 'error' in result:
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['Other'].append(clean_result)
            continue
        
        # Key change: Infer category from parsed "type" field, not hardcoded category
        result_type = result.get('type', '')
        
        if result_type == 'LinearDimension':
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['Measures'].append(clean_result)
        elif result_type == 'Radius':
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['Radii'].append(clean_result)
        elif result_type in ['ViewLabel', 'ViewCallout']:
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['Views'].append(clean_result)
        elif result_type == 'GD&T':
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['GD_T'].append(clean_result)
        elif result_type == 'Material':
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['Materials'].append(clean_result)
        elif result_type == 'Note':
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['Notes'].append(clean_result)
        elif result_type == 'Thread':
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['Threads'].append(clean_result)
        elif result_type == 'SurfaceRoughness':
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['SurfaceRoughness'].append(clean_result)
        elif result_type == 'GeneralTolerance':
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['GeneralTolerances'].append(clean_result)
        elif result_type == 'TitleBlock':
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['TitleBlock'].append(clean_result)
        else:
            # Unknown type or parsing failed
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            final_json['Other'].append(clean_result)
    
    # Add metadata
    processing_time = time.time() - start_time
    final_json['metadata'] = {
        'processing_time': processing_time,
        'annotations_detected': len(all_annotations),
        'annotations_parsed': len(parsed_results),
        'pipeline_stages': 2,
        'stage1_detections': len(all_annotations),
        'stage2_parsed': len([r for r in parsed_results if 'error' not in r and r.get('type') != 'Unknown'])
    }
    
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"Processing time: {processing_time:.3f}s")
    logger.info(f"Detected: {len(all_annotations)} annotations")
    logger.info(f"Parsed: {len([r for r in parsed_results if 'error' not in r and r.get('type') != 'Unknown'])} annotations")
    logger.info("=" * 60)
    
    return final_json


def run_full_parsing_pipeline_sync(
    image_path: Path,
    roboflow_api_key: str = None,
    gemini_api_key: str = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_full_hybrid_pipeline.
    
    For compatibility with existing code that doesn't use async/await.
    This now uses the hybrid pipeline by default.
    
    Args:
        image_path: Path to the mechanical drawing image
        roboflow_api_key: Roboflow API key (defaults to ub3sg9EEXSEZhVGZL4JD if None)
        gemini_api_key: Google API key for Gemini (required)
    
    Returns:
        Final structured JSON dictionary organized by category
    """
    from od_parse.mechanical_drawing.roboflow_client import DEFAULT_API_KEY
    
    # Use default Roboflow API key if not provided
    if roboflow_api_key is None:
        roboflow_api_key = DEFAULT_API_KEY
    
    if gemini_api_key is None:
        raise ValueError("gemini_api_key is required")
    
    return run_full_hybrid_pipeline_sync(image_path, roboflow_api_key, gemini_api_key)

