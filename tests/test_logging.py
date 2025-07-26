"""
Tests for the structured logging system.
"""

import pytest
from glitch_core.config.logging import setup_logging, get_logger, DriftLogger, PersonalityLogger, APILogger


def test_logging_setup():
    """Test that logging setup works correctly."""
    setup_logging("DEBUG")
    logger = get_logger("test")
    assert logger is not None


def test_drift_logger():
    """Test drift logger functionality."""
    setup_logging("INFO")
    drift_logger = DriftLogger("test_exp_123")
    
    # Test simulation start logging
    drift_logger.simulation_start(
        persona_type="resilient_optimist",
        drift_profile="resilient_optimist",
        epochs=10,
        events_per_epoch=5
    )
    
    # Test epoch completion logging
    drift_logger.epoch_completed(
        epoch=5,
        total_epochs=10,
        emotional_state={"joy": 0.7, "anxiety": 0.2}
    )
    
    # Test pattern emergence logging
    drift_logger.pattern_emerged(
        pattern_type="emotional_amplification",
        emotions=["joy", "social_energy"],
        epoch=5,
        confidence=0.8
    )
    
    # Test stability warning logging
    drift_logger.stability_warning(
        warning_type="extreme_emotion_warning",
        emotions=["anxiety"],
        risk_level="medium"
    )


def test_personality_logger():
    """Test personality logger functionality."""
    setup_logging("DEBUG")
    personality_logger = PersonalityLogger()
    
    # Test profile loading
    personality_logger.profile_loaded(
        profile_name="resilient_optimist",
        traits={"openness": 0.7, "neuroticism": 0.3}
    )
    
    # Test emotional shift
    personality_logger.emotional_shift(
        emotion="joy",
        old_value=0.5,
        new_value=0.7,
        trigger="personal_achievement"
    )
    
    # Test trait activation
    personality_logger.trait_activation(
        trait="extraversion",
        activation_level=0.8,
        context="social_interaction"
    )


def test_api_logger():
    """Test API logger functionality."""
    setup_logging("INFO")
    api_logger = APILogger()
    
    # Test request start
    api_logger.request_start(
        method="GET",
        path="/health",
        client_ip="127.0.0.1"
    )
    
    # Test request complete
    api_logger.request_complete(
        method="GET",
        path="/health",
        status_code=200,
        duration_ms=15.5
    )
    
    # Test WebSocket events
    api_logger.websocket_connected("client_123")
    api_logger.websocket_disconnected("client_123")


def test_log_context():
    """Test log context manager."""
    setup_logging("INFO")
    logger = get_logger("test")
    
    from glitch_core.config.logging import log_context
    
    with log_context(experiment_id="test_123", user_id="user_456") as ctx_logger:
        ctx_logger.info("test_message", additional_data="test_value") 