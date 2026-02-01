"""
Configuration Settings

Environment variables and application configuration.
"""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App settings
    app_name: str = "Pocket Planner API"
    app_version: str = "0.1.0"
    debug: bool = True
    
    # API settings
    api_prefix: str = "/api/v1"
    
    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Optimization defaults
    default_max_iterations: int = 5
    default_door_clearance: float = 60.0
    default_walking_path_width: float = 45.0
    
    # Google AI (for Developer A)
    google_api_key: str = ""
    gemini_model: str = "gemini-1.5-pro"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
