"""
Application configuration management.
"""

import os
from typing import Optional

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Application settings
    APP_NAME: str = "AI Personality Drift Simulation"
    VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    
    # Database settings
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    QDRANT_URL: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    
    # Model settings
    MODEL_PATH: str = Field(default="./models", env="MODEL_PATH")
    MODEL_NAME: str = Field(default="llama-3.1-8b-instruct", env="MODEL_NAME")
    
    # Simulation settings
    SIMULATION_DURATION_DAYS: int = Field(default=30, env="SIMULATION_DURATION_DAYS")
    TIME_COMPRESSION_FACTOR: int = Field(default=24, env="TIME_COMPRESSION_FACTOR")
    
    # Data storage
    DATA_DIR: str = Field(default="./data", env="DATA_DIR")
    RESULTS_DIR: str = Field(default="./data/results", env="RESULTS_DIR")
    
    # API settings
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = Field(default=["*"], env="CORS_ORIGINS")


# Global settings instance
settings = Settings() 