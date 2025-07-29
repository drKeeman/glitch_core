"""
Tests for configuration management.
"""

import pytest
from pydantic import ValidationError

from src.core.config import Settings


def test_settings_defaults():
    """Test that settings have correct default values."""
    settings = Settings()
    
    assert settings.APP_NAME == "AI Personality Drift Simulation"
    assert settings.VERSION == "0.1.0"
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000
    assert settings.REDIS_URL == "redis://localhost:6379"
    assert settings.QDRANT_URL == "http://localhost:6333"


def test_settings_environment_override(monkeypatch):
    """Test that environment variables override defaults."""
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("REDIS_URL", "redis://test:6379")
    
    settings = Settings()
    
    assert settings.DEBUG is True
    assert settings.PORT == 9000
    assert settings.REDIS_URL == "redis://test:6379"


def test_settings_invalid_port(monkeypatch):
    """Test that invalid port raises validation error."""
    monkeypatch.setenv("PORT", "invalid")
    
    with pytest.raises(ValidationError):
        Settings() 