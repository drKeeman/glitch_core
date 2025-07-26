"""
Application settings and configuration management.
"""

import os
from typing import List, Union
from pydantic import Field, ConfigDict, field_validator, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from pydantic_settings import BaseSettings


class CommaSeparatedList:
    """Custom type for handling comma-separated strings from environment variables."""
    
    def __init__(self, value: Union[str, List[str]]):
        if isinstance(value, str):
            self.items = [item.strip() for item in value.split(',')]
        else:
            self.items = value
    
    def __iter__(self):
        return iter(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __len__(self):
        return len(self.items)
    
    def __repr__(self):
        return f"CommaSeparatedList({self.items})"
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: type,
        _handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.union_schema([
            core_schema.str_schema(),
            core_schema.list_schema(core_schema.str_schema()),
        ])


class Settings(BaseSettings):
    """Application settings."""
    
    # Environment
    ENV: str = Field(default="staging")
    LOG_LEVEL: str = Field(default="INFO")
    
    # API Configuration
    API_PREFIX: str = Field(default="/api/v1")
    ALLOWED_ORIGINS: CommaSeparatedList = Field(
        default=CommaSeparatedList(["http://localhost:3000", "https://cognitive-drift.app"])
    )
    
    # Database URLs
    QDRANT_URL: str = Field(default="http://localhost:6333")
    REDIS_URL: str = Field(default="redis://localhost:6379")
    
    # LLM Configuration
    OLLAMA_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="llama3.2:3b")
    
    # WebSocket Configuration
    WS_HEARTBEAT_INTERVAL: int = Field(default=30)
    
    # Simulation Configuration
    DEFAULT_EPOCHS: int = Field(default=100)
    DEFAULT_EVENTS_PER_EPOCH: int = Field(default=10)
    
    @field_validator('ALLOWED_ORIGINS', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v):
        """Parse ALLOWED_ORIGINS from comma-separated string or list."""
        return CommaSeparatedList(v)
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_parse_none_str=None,
        extra="ignore"
    )


# Global settings instance
_settings: Settings = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings 