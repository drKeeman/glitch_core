"""
Custom exceptions for Glitch Core.
"""

from typing import Optional, Dict, Any


class GlitchCoreError(Exception):
    """Base exception for Glitch Core."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ConfigurationError(GlitchCoreError):
    """Raised when configuration is invalid or missing."""
    pass


class SimulationError(GlitchCoreError):
    """Raised when simulation fails."""
    pass


class LLMConnectionError(GlitchCoreError):
    """Raised when LLM service is unavailable."""
    pass


class MemoryStorageError(GlitchCoreError):
    """Raised when memory storage fails."""
    pass


class DatabaseConnectionError(GlitchCoreError):
    """Raised when database connection fails."""
    pass


class ValidationError(GlitchCoreError):
    """Raised when input validation fails."""
    pass


class InterventionError(GlitchCoreError):
    """Raised when intervention application fails."""
    pass


class AnalysisError(GlitchCoreError):
    """Raised when analysis operations fail."""
    pass


class WebSocketError(GlitchCoreError):
    """Raised when WebSocket operations fail."""
    pass


class AuthenticationError(GlitchCoreError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(GlitchCoreError):
    """Raised when authorization fails."""
    pass


class RateLimitError(GlitchCoreError):
    """Raised when rate limits are exceeded."""
    pass


class ResourceNotFoundError(GlitchCoreError):
    """Raised when a requested resource is not found."""
    pass


class ServiceUnavailableError(GlitchCoreError):
    """Raised when a required service is unavailable."""
    pass


class TimeoutError(GlitchCoreError):
    """Raised when operations timeout."""
    pass


class DataIntegrityError(GlitchCoreError):
    """Raised when data integrity is compromised."""
    pass


class ExternalServiceError(GlitchCoreError):
    """Raised when external service calls fail."""
    pass


class CircuitBreakerError(GlitchCoreError):
    """Raised when circuit breaker is open."""
    pass


class RetryableError(GlitchCoreError):
    """Base class for errors that can be retried."""
    pass


class NonRetryableError(GlitchCoreError):
    """Base class for errors that should not be retried."""
    pass 