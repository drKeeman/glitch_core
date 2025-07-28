"""
Tests for the drift engine.
"""

import pytest
from glitch_core.config import get_drift_engine, get_settings
from glitch_core.core.personality.profiles import get_persona_config, get_drift_profile


@pytest.mark.asyncio
async def test_drift_engine_initialization():
    """Test drift engine initialization."""
    engine = get_drift_engine()
    assert engine is not None
    assert engine.settings is not None


@pytest.mark.asyncio
async def test_basic_simulation():
    """Test a basic simulation run."""
    engine = get_drift_engine()
    
    # Get a persona config
    persona_config = get_persona_config("resilient_optimist")
    drift_profile = get_drift_profile("resilient_optimist")
    
    # Convert to dict format for the engine
    persona_dict = {
        "traits": persona_config.traits,
        "cognitive_biases": persona_config.cognitive_biases,
        "emotional_baselines": persona_config.emotional_baselines,
        "memory_patterns": persona_config.memory_patterns
    }
    
    drift_dict = {
        "evolution_rules": drift_profile.evolution_rules,
        "stability_metrics": drift_profile.stability_metrics,
        "breakdown_conditions": drift_profile.breakdown_conditions
    }
    
    # Run a short simulation
    result = await engine.run_simulation(
        persona_config=persona_dict,
        drift_profile=drift_dict,
        epochs=5,
        events_per_epoch=3,
        seed=42
    )
    
    assert result is not None
    assert result.experiment_id is not None
    assert result.epochs == 5
    assert result.events_per_epoch == 3
    assert len(result.emotional_states) == 5
    assert result.start_time is not None
    assert result.end_time is not None


@pytest.mark.asyncio
async def test_personality_profiles():
    """Test personality profile loading."""
    profiles = ["resilient_optimist", "anxious_overthinker", "stoic_philosopher", "creative_volatile"]
    
    for profile_name in profiles:
        persona_config = get_persona_config(profile_name)
        drift_profile = get_drift_profile(profile_name)
        
        assert persona_config is not None
        assert drift_profile is not None
        assert persona_config.traits is not None
        assert persona_config.emotional_baselines is not None
        assert drift_profile.evolution_rules is not None
        assert drift_profile.stability_metrics is not None


def test_invalid_profile():
    """Test error handling for invalid profiles."""
    with pytest.raises(ValueError):
        get_persona_config("invalid_profile")
    
    with pytest.raises(ValueError):
        get_drift_profile("invalid_profile") 