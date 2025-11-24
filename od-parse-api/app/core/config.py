"""
Configuration management for the FastAPI application.

Loads environment variables and provides centralized settings.
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    google_api_key: str
    roboflow_api_key: Optional[str] = None
    
    # API Configuration
    max_file_size_mb: int = 100
    temp_dir: Path = Path("/tmp/od-parse")
    allowed_extensions: List[str] = [
        '.pdf', '.png', '.jpg', '.jpeg', '.xlsx', '.xls', '.dxf', '.dwg'
    ]
    
    # Application Info
    app_name: str = "Intelligent File Parser API"
    app_version: str = "1.0.0"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env file


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        # Ensure temp directory exists
        _settings.temp_dir.mkdir(parents=True, exist_ok=True)
    return _settings

