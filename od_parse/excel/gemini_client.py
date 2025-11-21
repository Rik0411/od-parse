"""
Gemini REST API Client for Excel BOM SQL Generation

This module provides helper functions for generating DuckDB SQL queries
using Google Gemini to map raw Excel columns to standard BOM schema.
"""

import json
import time
import requests
from typing import List

from od_parse.utils.logging_utils import get_logger

# Retry configuration for 429 errors
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff delays in seconds

logger = get_logger(__name__)

# Gemini API Configuration
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-2.0-flash"  # Using stable 2.0 flash model


def generate_duckdb_sql_for_bom(columns: List[str], api_key: str) -> str:
    """
    Generate a DuckDB SQL query to map raw Excel columns to standard BOM schema.
    
    Uses Gemini to intelligently map column names to:
    - part_number (String)
    - description (String)
    - quantity (Integer)
    - material (String)
    - vendor (String)
    
    Args:
        columns: List of column names from the Excel sheet
        api_key: Google API key for Gemini
        
    Returns:
        Raw SQL query string (no markdown, no explanations)
        
    Raises:
        ValueError: If API response is invalid or cannot be parsed
        requests.exceptions.RequestException: If API call fails
    """
    if not columns:
        raise ValueError("Column list cannot be empty")
    
    # Build system prompt
    columns_str = ", ".join(columns)
    system_prompt = f"""You are a SQL Expert for DuckDB. I have a table named raw_data with the following columns: {columns_str}.

Write a DuckDB SQL query to select data from raw_data and alias the columns to match this standard Bill of Materials (BOM) schema:

part_number (String)
description (String)
quantity (Integer)
material (String)
vendor (String)

If a matching column doesn't exist, select NULL as that column name.

Return ONLY the raw SQL query. No markdown, no explanations."""
    
    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
    
    # Build request payload - we want plain text response, not JSON
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Generate SQL query for columns: {columns_str}"}]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": 0.1,  # Low temperature for deterministic SQL generation
            "maxOutputTokens": 500
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
            sql_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
            
            if not sql_text:
                raise ValueError(f"Invalid API response structure from Gemini API")
            
            # Clean up the SQL query - remove markdown code blocks if present
            sql_text = sql_text.strip()
            if sql_text.startswith("```sql"):
                sql_text = sql_text[6:].strip()
            elif sql_text.startswith("```"):
                sql_text = sql_text[3:].strip()
            if sql_text.endswith("```"):
                sql_text = sql_text[:-3].strip()
            
            # Return the cleaned SQL query
            return sql_text.strip()
                
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
            logger.error(f"Unexpected error in Gemini API call: {e}")
            raise
    
    # If we exhausted all retries
    if last_exception:
        raise last_exception
    
    raise ValueError("Failed to generate SQL query after all retries")

