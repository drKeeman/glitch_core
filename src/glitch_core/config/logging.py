"""
Structured logging configuration for Glitch Core.
"""

import sys
import time
import logging
from typing import Any, Dict
from contextlib import contextmanager

import structlog
from structlog.types import Processor


def setup_logging(log_level: str = "INFO") -> None:
    """Setup structured logging with structlog."""
    
    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    
    # Configure structlog processors
    processors: list[Processor] = [
        # Add timestamp
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        
        # Add call site info
        structlog.processors.CallsiteParameterAdder(
            parameters={
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        
        # Format the exception info
        structlog.processors.format_exc_info,
        
        # Add stack info
        structlog.processors.StackInfoRenderer(),
        
        # JSON output for production, console for development
        structlog.processors.JSONRenderer() if log_level == "PRODUCTION" else structlog.dev.ConsoleRenderer()
    ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


@contextmanager
def log_context(**context: Any):
    """Context manager for adding structured context to logs."""
    logger = get_logger()
    bound_logger = logger.bind(**context)
    try:
        yield bound_logger
    finally:
        # Clean up if needed
        pass


class DriftLogger:
    """Specialized logger for drift simulation events."""
    
    def __init__(self, experiment_id: str = None):
        self.logger = get_logger("drift_engine")
        self.experiment_id = experiment_id
        if experiment_id:
            self.logger = self.logger.bind(experiment_id=experiment_id)
    
    def simulation_start(self, persona_type: str, drift_profile: str, epochs: int, events_per_epoch: int):
        """Log simulation start."""
        self.logger.info(
            "simulation_started",
            persona_type=persona_type,
            drift_profile=drift_profile,
            epochs=epochs,
            events_per_epoch=events_per_epoch,
            event_type="simulation_start"
        )
    
    def epoch_completed(self, epoch: int, total_epochs: int, emotional_state: Dict[str, float]):
        """Log epoch completion."""
        self.logger.info(
            "epoch_completed",
            epoch=epoch,
            total_epochs=total_epochs,
            emotional_state=emotional_state,
            progress=f"{epoch}/{total_epochs}",
            event_type="epoch_completed"
        )
    
    def pattern_emerged(self, pattern_type: str, emotions: list, epoch: int, confidence: float):
        """Log pattern emergence."""
        self.logger.warning(
            "pattern_emerged",
            pattern_type=pattern_type,
            emotions=emotions,
            epoch=epoch,
            confidence=confidence,
            event_type="pattern_emerged"
        )
    
    def stability_warning(self, warning_type: str, emotions: list, risk_level: str):
        """Log stability warning."""
        self.logger.error(
            "stability_warning",
            warning_type=warning_type,
            emotions=emotions,
            risk_level=risk_level,
            event_type="stability_warning"
        )
    
    def intervention_injected(self, intervention_type: str, impact_score: float):
        """Log intervention injection."""
        self.logger.info(
            "intervention_injected",
            intervention_type=intervention_type,
            impact_score=impact_score,
            event_type="intervention_injected"
        )
    
    def simulation_completed(self, duration_seconds: float, patterns_detected: int, warnings: int):
        """Log simulation completion."""
        self.logger.info(
            "simulation_completed",
            duration_seconds=duration_seconds,
            patterns_detected=patterns_detected,
            warnings=warnings,
            event_type="simulation_completed"
        )


class PersonalityLogger:
    """Specialized logger for personality-related events."""
    
    def __init__(self):
        self.logger = get_logger("personality")
    
    def profile_loaded(self, profile_name: str, traits: Dict[str, float]):
        """Log personality profile loading."""
        self.logger.info(
            "profile_loaded",
            profile_name=profile_name,
            traits=traits,
            event_type="profile_loaded"
        )
    
    def emotional_shift(self, emotion: str, old_value: float, new_value: float, trigger: str):
        """Log emotional state changes."""
        self.logger.debug(
            "emotional_shift",
            emotion=emotion,
            old_value=old_value,
            new_value=new_value,
            delta=new_value - old_value,
            trigger=trigger,
            event_type="emotional_shift"
        )
    
    def trait_activation(self, trait: str, activation_level: float, context: str):
        """Log trait activation."""
        self.logger.debug(
            "trait_activation",
            trait=trait,
            activation_level=activation_level,
            context=context,
            event_type="trait_activation"
        )


class APILogger:
    """Specialized logger for API events."""
    
    def __init__(self):
        self.logger = get_logger("api")
    
    def request_start(self, method: str, path: str, client_ip: str = None):
        """Log API request start."""
        self.logger.info(
            "request_start",
            method=method,
            path=path,
            client_ip=client_ip,
            event_type="request_start"
        )
    
    def request_complete(self, method: str, path: str, status_code: int, duration_ms: float):
        """Log API request completion."""
        self.logger.info(
            "request_complete",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            event_type="request_complete"
        )
    
    def websocket_connected(self, client_id: str):
        """Log WebSocket connection."""
        self.logger.info(
            "websocket_connected",
            client_id=client_id,
            event_type="websocket_connected"
        )
    
    def websocket_disconnected(self, client_id: str):
        """Log WebSocket disconnection."""
        self.logger.info(
            "websocket_disconnected",
            client_id=client_id,
            event_type="websocket_disconnected"
        )


# Global logger instances
drift_logger = DriftLogger()
personality_logger = PersonalityLogger()
api_logger = APILogger() 