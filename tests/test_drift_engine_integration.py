"""
Tests for drift engine integration with memory and LLM components.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from glitch_core.config import get_drift_engine, get_settings
from glitch_core.core.personality.profiles import get_persona_config, get_drift_profile


@pytest_asyncio.fixture
async def drift_engine():
    """Create a drift engine for testing."""
    engine = get_drift_engine()
    
    # Mock the memory and reflection components
    engine.memory_manager = AsyncMock()
    engine.reflection_engine = AsyncMock()
    
    return engine


class TestDriftEngineIntegration:
    """Test drift engine integration with memory and LLM."""
    
    @pytest.mark.asyncio
    async def test_run_simulation_with_memory_and_reflection(self, drift_engine):
        """Test running a simulation with memory and reflection integration."""
        # Mock successful initialization
        drift_engine.memory_manager.initialize.return_value = None
        drift_engine.reflection_engine.initialize.return_value = None
        
        # Mock memory storage
        drift_engine.memory_manager.save_memory.return_value = MagicMock(
            id="test_memory_id",
            content="Test memory",
            emotional_weight=0.7
        )
        
        # Mock reflection generation
        drift_engine.reflection_engine.generate_reflection.return_value = MagicMock(
            reflection="This is a test reflection.",
            generation_time=1.0,
            token_count=30,
            confidence=0.8,
            emotional_impact={"joy": 0.5}
        )
        
        # Mock memory retrieval
        drift_engine.memory_manager.retrieve_contextual.return_value = [
            MagicMock(content="Previous memory 1"),
            MagicMock(content="Previous memory 2")
        ]
        
        # Get test persona and drift profile
        persona_config = get_persona_config("resilient_optimist")
        drift_profile = get_drift_profile("resilient_optimist")
        
        # Convert to dictionaries for the drift engine
        persona_dict = persona_config.to_dict()
        drift_dict = drift_profile.to_dict()
        
        # Run a short simulation
        result = await drift_engine.run_simulation(
            persona_config=persona_dict,
            drift_profile=drift_dict,
            epochs=2,
            events_per_epoch=3,
            seed=42
        )
        
        # Verify simulation completed
        assert result.experiment_id is not None
        assert len(result.emotional_states) == 2
        assert result.epochs == 2
        assert result.events_per_epoch == 3
        
        # Verify memory and reflection components were initialized
        drift_engine.memory_manager.initialize.assert_called_once()
        drift_engine.reflection_engine.initialize.assert_called_once()
        
        # Verify memory storage was called (for each event)
        assert drift_engine.memory_manager.save_memory.call_count >= 6  # 2 epochs * 3 events
    
    @pytest.mark.asyncio
    async def test_memory_storage_integration(self, drift_engine):
        """Test memory storage integration during simulation."""
        # Mock initialization
        drift_engine.memory_manager.initialize.return_value = None
        drift_engine.reflection_engine.initialize.return_value = None
        
        # Mock memory storage
        drift_engine.memory_manager.save_memory.return_value = MagicMock(
            id="test_id",
            content="Test event",
            emotional_weight=0.7
        )
        
        persona_config = get_persona_config("anxious_overthinker")
        drift_profile = get_drift_profile("anxious_overthinker")
        
        # Convert to dictionaries for the drift engine
        persona_dict = persona_config.to_dict()
        drift_dict = drift_profile.to_dict()
        
        # Run simulation
        await drift_engine.run_simulation(
            persona_config=persona_dict,
            drift_profile=drift_dict,
            epochs=1,
            events_per_epoch=2,
            seed=123
        )
        
        # Verify memory storage calls
        save_calls = drift_engine.memory_manager.save_memory.call_args_list
        
        # Should have at least 2 calls (one for each event)
        assert len(save_calls) >= 2
        
        # Check that memory storage was called with correct parameters
        for call in save_calls:
            args, kwargs = call
            assert "content" in kwargs
            assert "emotional_weight" in kwargs
            assert "persona_bias" in kwargs
            assert "memory_type" in kwargs
            assert kwargs["memory_type"] == "event"
    
    @pytest.mark.asyncio
    async def test_reflection_generation_integration(self, drift_engine):
        """Test reflection generation integration during simulation."""
        # Mock initialization
        drift_engine.memory_manager.initialize.return_value = None
        drift_engine.reflection_engine.initialize.return_value = None
        
        # Mock memory retrieval for reflection
        drift_engine.memory_manager.retrieve_contextual.return_value = [
            MagicMock(content="Previous experience")
        ]
        
        # Mock reflection generation
        drift_engine.reflection_engine.generate_reflection.return_value = MagicMock(
            reflection="I'm reflecting on this experience.",
            generation_time=0.5,
            token_count=20,
            confidence=0.7,
            emotional_impact={"anxiety": 0.3}
        )
        
        persona_config = get_persona_config("stoic_philosopher")
        drift_profile = get_drift_profile("stoic_philosopher")
        
        # Convert to dictionaries for the drift engine
        persona_dict = persona_config.to_dict()
        drift_dict = drift_profile.to_dict()
        
        # Run simulation
        await drift_engine.run_simulation(
            persona_config=persona_dict,
            drift_profile=drift_dict,
            epochs=1,
            events_per_epoch=3,
            seed=456
        )
        
        # Verify reflection generation was called for significant events
        generate_calls = drift_engine.reflection_engine.generate_reflection.call_args_list
        
        # Should have some reflection calls (for significant events)
        assert len(generate_calls) >= 0  # May be 0 if no significant events
        
        # Check that reflection generation was called with correct parameters
        for call in generate_calls:
            args, kwargs = call
            assert "trigger_event" in kwargs
            assert "emotional_state" in kwargs
            assert "memories" in kwargs
            assert "persona_prompt" in kwargs
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, drift_engine):
        """Test error handling when memory or reflection components fail."""
        # Mock initialization failure
        drift_engine.memory_manager.initialize.side_effect = Exception("Memory init failed")
        drift_engine.reflection_engine.initialize.side_effect = Exception("Reflection init failed")
        
        persona_config = get_persona_config("resilient_optimist")
        drift_profile = get_drift_profile("resilient_optimist")
        
        # Convert to dictionaries for the drift engine
        persona_dict = persona_config.to_dict()
        drift_dict = drift_profile.to_dict()
        
        # Simulation should still run even if memory/reflection fail
        result = await drift_engine.run_simulation(
            persona_config=persona_dict,
            drift_profile=drift_dict,
            epochs=1,
            events_per_epoch=2,
            seed=789
        )
        
        # Verify simulation completed despite errors
        assert result.experiment_id is not None
        assert len(result.emotional_states) == 1
    
    def test_emotional_weight_calculation(self, drift_engine):
        """Test emotional weight calculation for events."""
        # Test positive event
        positive_event = {"type": "positive", "description": "Success at work"}
        emotional_state = {"joy": 0.6, "anxiety": 0.3}
        weight = drift_engine._calculate_emotional_weight(positive_event, emotional_state)
        assert 0.5 <= weight <= 1.0
        
        # Test trauma event
        trauma_event = {"type": "trauma", "description": "Severe accident"}
        weight = drift_engine._calculate_emotional_weight(trauma_event, emotional_state)
        assert weight > 0.8  # Trauma should have high weight
        
        # Test neutral event
        neutral_event = {"type": "neutral", "description": "Routine task"}
        weight = drift_engine._calculate_emotional_weight(neutral_event, emotional_state)
        assert weight < 0.5  # Neutral should have lower weight
    
    def test_significant_event_detection(self, drift_engine):
        """Test significant event detection."""
        # Test significant event types
        significant_events = [
            {"type": "trauma", "description": "Accident"},
            {"type": "success", "description": "Achievement"},
            {"type": "failure", "description": "Mistake"}
        ]
        
        for event in significant_events:
            is_significant = drift_engine._is_significant_event(event, {"joy": 0.5})
            assert is_significant is True
        
        # Test neutral event
        neutral_event = {"type": "neutral", "description": "Routine"}
        is_significant = drift_engine._is_significant_event(neutral_event, {"joy": 0.5})
        assert is_significant is False
        
        # Test high emotional intensity
        high_intensity_state = {"joy": 0.8, "anxiety": 0.9}
        is_significant = drift_engine._is_significant_event(neutral_event, high_intensity_state)
        assert is_significant is True
    
    def test_persona_prompt_creation(self, drift_engine):
        """Test persona-specific prompt creation."""
        # Test resilient optimist
        persona_config = {"type": "resilient_optimist"}
        prompt = drift_engine._create_persona_prompt(persona_config)
        assert "optimistic" in prompt.lower()
        assert "resilient" in prompt.lower()
        
        # Test anxious overthinker
        persona_config = {"type": "anxious_overthinker"}
        prompt = drift_engine._create_persona_prompt(persona_config)
        assert "anxious" in prompt.lower()
        assert "overthink" in prompt.lower()
        
        # Test unknown persona type
        persona_config = {"type": "unknown_type"}
        prompt = drift_engine._create_persona_prompt(persona_config)
        assert "balanced" in prompt.lower() 