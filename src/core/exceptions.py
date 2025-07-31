"""
Custom exceptions for the application.
"""

from typing import Any, Dict, Optional


class GlitchCoreException(Exception):
    """Base exception for all application errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(GlitchCoreException):
    """Raised when there's a configuration error."""
    pass


class DatabaseConnectionError(GlitchCoreException):
    """Raised when database connection fails."""
    pass


class ModelLoadError(GlitchCoreException):
    """Raised when model loading fails."""
    pass


class SimulationError(GlitchCoreException):
    """Raised when simulation encounters an error."""
    pass


class AssessmentError(GlitchCoreException):
    """Raised when assessment processing fails."""
    pass


class MechanisticAnalysisError(GlitchCoreException):
    """Raised when mechanistic analysis fails."""
    pass 