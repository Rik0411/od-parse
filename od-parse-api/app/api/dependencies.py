"""
FastAPI dependency injection functions.
"""

import tempfile
from pathlib import Path
from typing import Dict

from app.core.config import get_settings as get_core_settings, Settings


def get_settings_dep() -> Settings:
    """Dependency to get application settings."""
    return get_core_settings()


def get_temp_dir() -> Path:
    """Dependency to get temporary directory for file uploads."""
    settings = get_core_settings()
    return settings.temp_dir


def get_api_keys() -> Dict[str, str]:
    """Dependency to get API keys dictionary."""
    settings = get_core_settings()
    api_keys = {
        "google": settings.google_api_key
    }
    if settings.roboflow_api_key:
        api_keys["roboflow"] = settings.roboflow_api_key
    return api_keys

