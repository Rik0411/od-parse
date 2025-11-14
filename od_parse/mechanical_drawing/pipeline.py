"""
Hybrid Pipeline for Mechanical Drawing Parsing

This module implements a three-stage hybrid pipeline architecture:
- Stage 1: Detection (Roboflow local server) - detects all potential annotations
- Stage 2: Verification & Parsing (Gemini multimodal API) - verifies patches and parses values
- Stage 3: Missing Items Scan (Gemini multimodal API) - finds annotations missed by Stage 1

Based on the research paper "Automated Parsing of Engineering Drawings..."
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from od_parse.mechanical_drawing.gemini_client import (
    image_to_base64,
    call_gemini_multimodal
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


async def stage2_run_gemini_verification(
    image_path: Path,
    patch_base64: str,
    detection_class: str,
    api_key: str
) -> Optional[Dict[str, Any]]:
    """
    STAGE 2 (Gemini Verification & Parsing)
    
    Uses Gemini 2.5 Flash multimodal API to verify a patch and parse its content.
    This acts as the "Donut" parser layer, verifying if detection is correct and
    parsing the value from the patch.
    
    Args:
        image_path: Path to the full image (for context, not used directly)
        patch_base64: Base64-encoded image patch
        detection_class: The class predicted by Roboflow (e.g., "dimension", "radius")
        api_key: Google API key for Gemini
    
    Returns:
        Structured JSON object if verification succeeds, None if false positive
    """
    logger.debug(f"Stage 2: Verifying patch for class: \"{detection_class}\"")
    
    try:
        # Add rate limiting delay to avoid hitting API limits
        await asyncio.sleep(0.05)
        
        # System prompt for verification and parsing
        system_prompt = f"""You are an expert mechanical drawing verifier and parser.
Another AI model has provided this image patch and believes it contains a: "{detection_class}".

Your tasks are:
1. **VERIFY:** Is this correct? Does this patch *actually* contain a "{detection_class}"?
2. **PARSE:** If yes, parse the annotation into a structured JSON object.

Use the following schemas:
- For "dimension": {{"type": "LinearDimension", "value": 10}}
- For "radius": {{"type": "Radius", "value": 10, "count": 1}}
- For "gdt": {{"type": "GD_T", "symbol": "...", "value": 0.1, "datums": ["A", "B"]}}
- For "view": {{"type": "ViewLabel", "name": "Detail View A"}}

**CRITICAL:** If this is a false positive, or you cannot read the value,
or the patch does not contain the expected object, you MUST return **null**."""
        
        # Define response schema with oneOf to allow null or structured object
        response_schema = {
            "oneOf": [
                {"type": "NULL"},
                {
                    "type": "OBJECT",
                    "properties": {
                        "type": {"type": "STRING"},
                        "value": {"type": "NUMBER"},
                        "count": {"type": "NUMBER"},
                        "name": {"type": "STRING"},
                        "symbol": {"type": "STRING"},
                        "datums": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["type"]
                }
            ]
        }
        
        # Call Gemini multimodal API with patch
        result = await asyncio.to_thread(
            call_gemini_multimodal,
            api_key,
            patch_base64,
            "Verify and parse this patch.",
            response_schema,
            "image/png"  # Patches are saved as PNG
        )
        
        # If result is null or None, return None (false positive)
        if result is None:
            return None
        
        # Check if result is a dict with type field (valid parsing)
        if isinstance(result, dict) and result.get('type'):
            return result
        
        # Otherwise, treat as false positive
        return None
        
    except Exception as e:
        logger.error(f"Error in Stage 2 (Verification \"{detection_class}\"): {e}")
        return None  # Treat errors as false positives


async def stage3_find_missing_annotations(
    image_path: Path,
    verified_annotations: Dict[str, Any],
    api_key: str
) -> List[Dict[str, Any]]:
    """
    STAGE 3 (Gemini Missing Annotations Scan)
    
    Scans the full image and compares against the list of found items
    to find any annotations that were missed by Stage 1.
    
    Args:
        image_path: Path to the full mechanical drawing image
        verified_annotations: The final JSON object from Stage 2 (verified annotations)
        api_key: Google API key for Gemini
    
    Returns:
        List of newly found structured JSON annotations
    """
    logger.info("Stage 3: Running final 'missing values' scan...")
    
    try:
        # Convert full image to base64
        base64_image = await asyncio.to_thread(image_to_base64, image_path)
        
        # Determine MIME type from file extension
        mime_type = "image/png"
        if image_path.suffix.lower() in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        
        # Convert verified annotations to JSON string
        import json
        annotations_json_string = json.dumps(verified_annotations, indent=2)
        
        # System prompt for missing items scan
        system_prompt = """You are a Quality Assurance inspector for a drawing parser.
A primary system has already scanned the attached image and found
the annotations in the following JSON block.

**YOUR TASK:**
Carefully scan the **entire image** and find any text-based annotations
(dimensions, radii, view labels, GD&T) that are **MISSING** from the JSON.

- Do NOT return items that are already in the JSON.
- For each *new* item you find, parse it using the same schemas as before
  (e.g., {"type": "LinearDimension", "value": 10}).
- If you find no missing items, return an empty array []."""
        
        # Define response schema
        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "type": {"type": "STRING"},
                    "value": {"type": "NUMBER"},
                    "count": {"type": "NUMBER"},
                    "name": {"type": "STRING"},
                    "symbol": {"type": "STRING"},
                    "datums": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["type"]
            }
        }
        
        # Construct prompt with annotations list
        prompt = f"Here are the annotations we found so far:\n{annotations_json_string}\n\nNow, please find any that are missing from the image."
        
        # Call Gemini multimodal API
        missing_items = await asyncio.to_thread(
            call_gemini_multimodal,
            api_key,
            base64_image,
            prompt,
            response_schema,
            mime_type
        )
        
        # Ensure we have a list
        if not isinstance(missing_items, list):
            logger.warning(f"Expected list from API, got {type(missing_items)}. Wrapping in list.")
            missing_items = [missing_items] if missing_items else []
        
        logger.info(f"Stage 3: Found {len(missing_items)} missing annotations.")
        return missing_items
        
    except Exception as e:
        logger.error(f"Error in Stage 3 (Missing Values Scan): {e}")
        return []  # Return empty array on failure


async def run_full_hybrid_pipeline(
    image_path: Path,
    roboflow_api_key: str,
    gemini_api_key: str
) -> Dict[str, Any]:
    """
    MAIN HYBRID PARSING PIPELINE
    
    Orchestrates the complete three-stage hybrid pipeline:
    1. Stage 1: Roboflow detection (local server)
    2. Stage 2: Gemini verification & parsing (parallel patch processing)
    3. Stage 3: Gemini missing items scan (full image QA)
    
    Args:
        image_path: Path to the mechanical drawing image
        roboflow_api_key: Roboflow API key
        gemini_api_key: Google API key for Gemini
    
    Returns:
        Final structured JSON dictionary organized by category
    """
    logger.info("Starting full HYBRID parsing pipeline...")
    start_time = time.time()
    
    try:
        # --- STAGE 1: ROBOFLOW DETECTIONS ---
        logger.info("=" * 60)
        logger.info("STAGE 1: Annotation Detection (Roboflow)")
        logger.info("=" * 60)
        roboflow_result = await stage1_run_roboflow_detection(image_path, roboflow_api_key)
        
        predictions = roboflow_result.get('predictions', [])
        
        if not predictions or len(predictions) == 0:
            logger.warning("Stage 1 failed to find any annotations. Proceeding to Stage 3 as fallback.")
        
        # --- STAGE 2: GEMINI VERIFICATION & PARSING (in parallel) ---
        logger.info("=" * 60)
        logger.info(f"STAGE 2: Verification & Parsing (Gemini) - Processing {len(predictions)} patches in parallel")
        logger.info("=" * 60)
        
        verification_tasks = []
        for pred in predictions:
            # Crop the patch from the original image
            patch_base64 = await asyncio.to_thread(
                crop_image_patch,
                image_path,
                pred['x'],
                pred['y'],
                pred['width'],
                pred['height']
            )
            
            # Create verification task
            verification_tasks.append(
                stage2_run_gemini_verification(
                    image_path,
                    patch_base64,
                    pred['class'],
                    gemini_api_key
                )
            )
        
        # Process all patches in parallel
        verified_results = await asyncio.gather(*verification_tasks)
        
        # --- AGGREGATION (PRE-SCAN) ---
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
            'Other': []
        }
        
        # Filter out None values (false positives) and categorize
        for i, result in enumerate(verified_results):
            if result is None or not result.get('type'):
                continue  # Skip false positives
            
            result_type = result.get('type', '')
            
            # Add source info for tracking
            clean_result = {k: v for k, v in result.items() if k != '_source'}
            clean_result['_source'] = predictions[i] if i < len(predictions) else {}
            
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
        
        logger.info(f"Stage 2 complete. Verified {sum(len(v) for v in final_json.values())} annotations.")
        
        # --- STAGE 3: GEMINI "MISSING VALUES" SCAN ---
        logger.info("=" * 60)
        logger.info("STAGE 3: Missing Annotations Scan (Gemini)")
        logger.info("=" * 60)
        missing_items = await stage3_find_missing_annotations(image_path, final_json, gemini_api_key)
        
        # --- FINAL AGGREGATION ---
        # Add the newly found missing items to our final JSON
        for item in missing_items:
            item_type = item.get('type', '')
            
            if item_type == 'LinearDimension':
                final_json['Measures'].append(item)
            elif item_type == 'Radius':
                final_json['Radii'].append(item)
            elif item_type in ['ViewLabel', 'ViewCallout']:
                final_json['Views'].append(item)
            elif item_type == 'GD_T':
                final_json['GD_T'].append(item)
            elif item_type == 'Material':
                final_json['Materials'].append(item)
            elif item_type == 'Note':
                final_json['Notes'].append(item)
            elif item_type == 'Thread':
                final_json['Threads'].append(item)
            elif item_type == 'SurfaceRoughness':
                final_json['SurfaceRoughness'].append(item)
            elif item_type == 'GeneralTolerance':
                final_json['GeneralTolerances'].append(item)
            elif item_type == 'TitleBlock':
                final_json['TitleBlock'].append(item)
            else:
                final_json['Other'].append(item)
        
        # Add metadata
        processing_time = time.time() - start_time
        final_json['metadata'] = {
            'processing_time': processing_time,
            'annotations_detected': len(predictions),
            'annotations_verified': sum(len(v) for v in final_json.values()) - len(missing_items),
            'annotations_missing_found': len(missing_items),
            'pipeline_stages': 3,
            'stage1_detections': len(predictions),
            'stage2_verified': sum(1 for r in verified_results if r is not None),
            'stage3_missing_found': len(missing_items)
        }
        
        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info(f"Processing time: {processing_time:.3f}s")
        logger.info(f"Detected: {len(predictions)} annotations")
        logger.info(f"Verified: {sum(1 for r in verified_results if r is not None)} annotations")
        logger.info(f"Missing found: {len(missing_items)} annotations")
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
            'metadata': {
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
        }


def run_full_hybrid_pipeline_sync(
    image_path: Path,
    roboflow_api_key: str,
    gemini_api_key: str
) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_full_hybrid_pipeline.
    
    For compatibility with existing code that doesn't use async/await.
    
    Args:
        image_path: Path to the mechanical drawing image
        roboflow_api_key: Roboflow API key
        gemini_api_key: Google API key for Gemini
    
    Returns:
        Final structured JSON dictionary organized by category
    """
    return asyncio.run(run_full_hybrid_pipeline(image_path, roboflow_api_key, gemini_api_key))


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

