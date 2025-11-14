"""
Gemini REST API Client for Mechanical Drawing Parser

This module provides helper functions for making direct REST API calls to Google Gemini,
specifically for the two-stage mechanical drawing parsing pipeline.
"""

import base64
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests

from od_parse.utils.logging_utils import get_logger

# Retry configuration for 429 errors
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff delays in seconds

logger = get_logger(__name__)

# Gemini API Configuration
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-2.0-flash"  # Using stable 2.0 flash model


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

