"""
Mechanical Drawing Parser Module

This module implements a hybrid three-stage pipeline for parsing engineering drawings:
- Stage 1: Detection (Roboflow local server) - detects all potential annotations
- Stage 2: Verification & Parsing (Gemini multimodal API) - verifies patches and parses values
- Stage 3: Missing Items Scan (Gemini multimodal API) - finds annotations missed by Stage 1
"""

from od_parse.mechanical_drawing.pipeline import (
    run_full_parsing_pipeline,
    run_full_parsing_pipeline_sync,
    run_full_hybrid_pipeline,
    run_full_hybrid_pipeline_sync,
    stage1_run_roboflow_detection,
    stage2_run_gemini_verification,
    stage3_find_missing_annotations
)
from od_parse.mechanical_drawing.gemini_client import (
    image_to_base64,
    call_gemini_multimodal,
    call_gemini_text
)
from od_parse.mechanical_drawing.roboflow_client import (
    call_roboflow_detection,
    get_image_dimensions
)
from od_parse.mechanical_drawing.image_utils import (
    crop_image_patch
)

__all__ = [
    'run_full_parsing_pipeline',
    'run_full_parsing_pipeline_sync',
    'run_full_hybrid_pipeline',
    'run_full_hybrid_pipeline_sync',
    'stage1_run_roboflow_detection',
    'stage2_run_gemini_verification',
    'stage3_find_missing_annotations',
    'image_to_base64',
    'call_gemini_multimodal',
    'call_gemini_text',
    'call_roboflow_detection',
    'get_image_dimensions',
    'crop_image_patch'
]

