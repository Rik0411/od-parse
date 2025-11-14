"""
Gemini REST API Client for Mechanical Drawing Parser

This module provides helper functions for making direct REST API calls to Google Gemini,
specifically for the two-stage mechanical drawing parsing pipeline.
"""

import base64
import json
import time
import io
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import requests
from PIL import Image

from od_parse.utils.logging_utils import get_logger

# Retry configuration for 429 errors
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff delays in seconds

logger = get_logger(__name__)

# Gemini API Configuration
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-2.0-flash"  # Using stable 2.0 flash model
GEMINI_MODEL_BATCH = "gemini-2.5-flash-preview-09-2025"  # Better for large context/batch processing


def image_to_base64(image_path: Path) -> str:
    """
    Convert an image file to base64 string.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Base64 encoded string (without data URI prefix)
    """
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return base64_data
    except Exception as e:
        logger.error(f"Failed to convert image to base64: {e}")
        raise


def call_gemini_multimodal(
    api_key: str,
    base64_image: str,
    prompt: str,
    response_schema: Optional[Dict[str, Any]] = None,
    mime_type: str = "image/png"
) -> Dict[str, Any]:
    """
    Call Gemini multimodal API (vision + text) with image input.
    
    Args:
        api_key: Google API key
        base64_image: Base64 encoded image data
        prompt: Text prompt for the model
        response_schema: Optional JSON schema for structured output
        mime_type: MIME type of the image (default: image/png)
    
    Returns:
        Parsed JSON response from the API
    """
    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
    
    # Build request payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_image
                        }
                    }
                ]
            }
        ]
    }
    
    # Add response schema if provided
    if response_schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    else:
        # Default to JSON response even without schema
        payload["generationConfig"] = {
            "responseMimeType": "application/json"
        }
    
    # Make API call with retry logic for 429 errors
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}
    
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, params=params, headers=headers, timeout=60)
            
            # Handle 429 errors with retry
            if response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"Rate limit hit (429). Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error("Rate limit exceeded after all retries")
                    response.raise_for_status()
            
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text from response
            text_response = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
            
            if not text_response:
                raise ValueError("Invalid API response structure from Gemini multimodal API")
            
            # Parse JSON response
            try:
                parsed_json = json.loads(text_response)
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {text_response[:200]}")
                raise ValueError(f"Invalid JSON response from API: {e}")
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < MAX_RETRIES - 1:
                last_exception = e
                continue
            logger.error(f"API request failed: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Gemini multimodal API call: {e}")
            raise
    
    # If we exhausted all retries
    if last_exception:
        raise last_exception


def call_gemini_text(
    api_key: str,
    text: str,
    system_prompt: str,
    response_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Call Gemini text API with JSON schema for structured output.
    
    Args:
        api_key: Google API key
        text: Input text to parse
        system_prompt: System instruction for the model
        response_schema: JSON schema defining the expected output structure
    
    Returns:
        Parsed JSON response matching the schema
    """
    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
    
    # Build request payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": text}]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }
    
    # Make API call with retry logic for 429 errors
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}
    
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, params=params, headers=headers, timeout=30)
            
            # Handle 429 errors with retry
            if response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"Rate limit hit (429). Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error("Rate limit exceeded after all retries")
                    response.raise_for_status()
            
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text from response
            json_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
            
            if not json_text:
                raise ValueError(f"Invalid API response structure from Gemini text API for: \"{text}\"")
            
            # Parse JSON response
            try:
                parsed_json = json.loads(json_text)
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {json_text[:200]}")
                raise ValueError(f"Invalid JSON response from API: {e}")
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < MAX_RETRIES - 1:
                last_exception = e
                continue
            logger.error(f"API request failed: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Gemini text API call: {e}")
            raise
    
    # If we exhausted all retries
    if last_exception:
        raise last_exception


def stage2_run_batch_verification(
    patches_with_class: List[Tuple[bytes, str]],
    api_key: str
) -> List[Optional[Dict[str, Any]]]:
    """
    STAGE 2 (Gemini Batch Verification & Parsing)
    
    Sends ALL patches in a single API call to Gemini 2.5 Flash.
    This eliminates 429 rate limit errors by making one call instead of many.
    
    Args:
        patches_with_class: A list of tuples, where each tuple is
                           (patch_image_bytes, detection_class)
        api_key: Google API key for Gemini
    
    Returns:
        A list of structured JSON objects (or None for false positives).
        The list length matches the input patches_with_class length.
    """
    logger.info(f"Stage 2: Batch processing {len(patches_with_class)} patches in a single API call...")
    
    if not patches_with_class:
        logger.warning("No patches provided for batch verification")
        return []
    
    # Use 2.5-flash for better large context handling
    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL_BATCH}:generateContent"
    
    # Build system prompt
    system_prompt = f"""You are an expert mechanical drawing parser. An AI model has provided
{len(patches_with_class)} image patches for verification and parsing.

You will receive a sequence of (image) and (text) parts.
The (text) part will state the detected class (e.g., "dimension").
Your task is to verify and parse EACH patch.

**YOUR TASKS:**
1. **VERIFY:** For each patch, is the detected class correct?
2. **PARSE:** If yes, parse the annotation into a structured JSON object
   based on the schemas below.
3. **RETURN:** You MUST return a single JSON array containing exactly
   {len(patches_with_class)} items.
   Each item in the array is either the parsed JSON object OR null
   if it was a false positive.

**SCHEMAS:**
- For dimensions: {{"id": "d1", "type": "LinearDimension", "value": 10.5, "units": "mm", "text": "10.5"}}
- For radii: {{"id": "r1", "type": "Radius", "value": 10, "count": 1, "text": "R10"}}
- For view labels: {{"id": "v1", "type": "ViewLabel", "name": "Detail View A", "text": "A"}}
- For GD&T: {{"id": "g1", "type": "GD_T", "symbol": "...", "value": 0.1, "datums": ["A", "B"], "text": "..."}}

**CRITICAL:** Your output MUST be a JSON array with {len(patches_with_class)}
entries. Use 'null' for any patch you determine is a false positive."""
    
    # Define response schema for array of objects or nulls
    response_schema = {
        "type": "ARRAY",
        "items": {
            "oneOf": [
                {"type": "NULL"},
                {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "STRING"},
                        "type": {"type": "STRING"},
                        "value": {"type": "NUMBER"},
                        "units": {"type": "STRING"},
                        "text": {"type": "STRING"},
                        "count": {"type": "NUMBER"},
                        "name": {"type": "STRING"},
                        "symbol": {"type": "STRING"},
                        "datums": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["id", "type"]
                }
            ]
        }
    }
    
    # Build multimodal payload with alternating image/text parts
    parts = []
    parts.append({"text": "See system instructions for batch parsing."})
    
    for i, (patch_bytes, detection_class) in enumerate(patches_with_class):
        # Convert bytes to base64
        patch_base64 = base64.b64encode(patch_bytes).decode('utf-8')
        
        # Add image part
        parts.append({
            "inlineData": {
                "mimeType": "image/png",
                "data": patch_base64
            }
        })
        
        # Add text part with detection class
        parts.append({
            "text": f"Patch {i+1} / {len(patches_with_class)}. Detected class: \"{detection_class}\". Verify and parse."
        })
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }
    
    # Make API call with retry logic for 429 errors
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, params=params, headers=headers, timeout=120)
            
            # Handle 429 errors with retry
            if response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    wait = (attempt + 1) * 2  # Exponential backoff (2s, 4s)
                    logger.warning(f"Rate limit hit (429) on BATCH call. Retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue
                else:
                    logger.error("Rate limit exceeded after all retries on BATCH call")
                    response.raise_for_status()
            
            response.raise_for_status()
            
            result = response.json()
            
            # Extract JSON text from response
            json_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "[]")
            
            if not json_text:
                logger.error("Batch parse FAILED: Empty response from API")
                return []
            
            # Parse JSON array
            parsed_list = json.loads(json_text)
            
            # Validate array length matches input
            if not isinstance(parsed_list, list):
                logger.error(f"Batch parse FAILED: Expected array, got {type(parsed_list)}")
                return []
            
            if len(parsed_list) != len(patches_with_class):
                logger.error(f"Batch parse FAILED: Expected {len(patches_with_class)} results, got {len(parsed_list)}")
                return []
            
            logger.info(f"Stage 2: Batch processing complete. Parsed {len(parsed_list)} items.")
            return parsed_list
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < MAX_RETRIES - 1:
                continue
            logger.error(f"API request failed: {e}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing batch response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in batch verification: {e}")
            return []
    
    logger.error("Rate limit exceeded after all retries on BATCH call.")
    return []


def process_image_batch(
    images: List[Image.Image],
    prompt: str,
    api_key: str,
    response_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a batch of images using Gemini REST API with batch processing.
    
    This function is designed for general document/image parsing (not just
    mechanical drawing patches). It takes multiple images and processes them
    in a single API call to avoid rate limiting.
    
    Args:
        images: List of PIL Image objects to process
        prompt: Text prompt describing what to extract from the images
        api_key: Google API key for Gemini
        response_schema: Optional JSON schema for structured output
    
    Returns:
        Dictionary containing parsed JSON response from the API
    """
    logger.info(f"Processing {len(images)} images in batch...")
    
    if not images:
        logger.warning("No images provided for batch processing")
        return {}
    
    # Use 2.5-flash for better large context handling
    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL_BATCH}:generateContent"
    
    # Build system prompt for general document parsing
    system_prompt = f"""You are an expert document parser. Analyze the provided image(s) and extract structured information.

Your task is to:
1. Extract all text content from the images
2. Identify and extract tables (if any)
3. Identify and extract form fields (if any)
4. Identify handwritten content (if any)
5. Extract any other relevant structured information

Return your analysis as a structured JSON object with the following structure:
{{
    "text": "extracted text content",
    "tables": [{{"rows": [...], "columns": [...]}}],
    "forms": [{{"field": "...", "value": "..."}}],
    "handwritten_content": ["..."],
    "other_data": {{...}}
}}"""
    
    # Build multimodal payload with all images
    parts = []
    parts.append({"text": prompt})
    
    for i, image in enumerate(images):
        # Convert PIL Image to base64
        buffer = io.BytesIO()
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Add image part
        parts.append({
            "inlineData": {
                "mimeType": "image/png",
                "data": image_base64
            }
        })
    
    # Build payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        }
    }
    
    # Add response schema if provided
    if response_schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    else:
        # Default to JSON response
        payload["generationConfig"] = {
            "responseMimeType": "application/json"
        }
    
    # Make API call with retry logic for 429 errors
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, params=params, headers=headers, timeout=120)
            
            # Check for transient errors (429 or 5xx) that should be retried
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    wait = (attempt + 1) * 2  # Exponential backoff
                    logger.warning(f"API Error ({response.status_code}) on single-stage call. Retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue
                else:
                    logger.error(f"API error ({response.status_code}) exceeded after all retries on image batch call")
                    response.raise_for_status()
            
            response.raise_for_status()
            
            result = response.json()
            
            # Extract JSON text from response
            json_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
            
            if not json_text:
                logger.error("Image batch processing FAILED: Empty response from API")
                return {}
            
            # Parse JSON response
            try:
                parsed_json = json.loads(json_text)
                logger.info(f"Image batch processing complete. Processed {len(images)} images.")
                return parsed_json
            except json.JSONDecodeError as e:
                # Try to extract JSON from response if it's embedded in text
                try:
                    json_start = json_text.find('{')
                    json_end = json_text.rfind('}')
                    if json_start >= 0 and json_end >= 0 and json_end > json_start:
                        json_str = json_text[json_start:json_end+1]
                        parsed_json = json.loads(json_str)
                        logger.info(f"Image batch processing complete. Processed {len(images)} images.")
                        return parsed_json
                    else:
                        # No JSON found, wrap entire response
                        logger.warning("Could not parse JSON from response, wrapping as text")
                        return {"extracted_text": json_text}
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON response: {e}")
                    return {"extracted_text": json_text}
            
        except requests.exceptions.HTTPError as e:
            # Retry on 429 or 5xx errors
            if (e.response.status_code == 429 or e.response.status_code >= 500) and attempt < MAX_RETRIES - 1:
                continue
            logger.error(f"API request failed: {e}")
            return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in image batch processing: {e}")
            return {}
    
    logger.error("Rate limit exceeded after all retries on image batch call.")
    return {}

