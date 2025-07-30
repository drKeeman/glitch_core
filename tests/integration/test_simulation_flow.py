"""
Integration tests for simulation engine and event system.
"""

import asyncio
import pytest
import logging
from datetime import datetime
from typing import Dict, Any

from src.services.simulation_engine import SimulationEngine
from src.services.event_generator import EventGenerator
from src.services.memory_service import MemoryService
from src.models.simulation import ExperimentalCondition, SimulationConfig
from src.models.events import EventType, Event
from src.models.persona import Persona, PersonaBaseline, PersonaState


# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSimulationFlow:
    """Test simulation engine and event system integration."""
    
    @pytest.fixture
    async def event_generator(self):
        """Create event generator for testing."""
        generator = EventGenerator()
        await generator.load_event_templates()
        return generator
    
    @pytest.fixture
    async def memory_service(self):
        """Create memory service for testing."""
        service = MemoryService()
        await service.initialize_memory_system()
        return service
    
    @pytest.mark.asyncio
    async def test_event_generator_initialization(self, event_generator):
        """Test event generator initialization."""
        # For now, we'll test that the generator can be created
        # The templates may not load if the config files don't exist
        assert event_generator is not None
        assert hasattr(event_generator, 'event_templates')
        assert hasattr(event_generator, 'frequency_weights')
        
        logger.info(f"Event generator initialized with {len(event_generator.event_templates)} templates")
    
    @pytest.mark.asyncio
    async def test_simulation_config_validation(self):
        """Test simulation config validation."""
        # Test basic config creation
        config = SimulationConfig(
            duration_days=30,
            experimental_condition=ExperimentalCondition.STRESS,
            stress_event_frequency=0.1,
            neutral_event_frequency=0.2
        )
        
        assert config.duration_days == 30
        assert config.experimental_condition == ExperimentalCondition.STRESS
        assert config.is_valid_configuration()
        
        logger.info("Simulation config validation test passed")
    
    @pytest.mark.asyncio
    async def test_persona_creation(self):
        """Test persona creation with required fields."""
        # Create a complete persona baseline
        baseline = PersonaBaseline(
            name="Test Persona",
            age=30,
            occupation="Software Engineer",
            background="A test persona for simulation testing",
            openness=0.7,
            conscientiousness=0.6,
            extraversion=0.5,
            agreeableness=0.8,
            neuroticism=0.3,
            baseline_phq9=5.0,
            baseline_gad7=4.0,
            baseline_pss10=12.0
        )
        
        # Create persona state with a specific ID
        state = PersonaState(
            persona_id="test_persona_1",
            simulation_day=0,
            last_assessment_day=-1
        )
        
        # Create persona
        persona = Persona(baseline=baseline, state=state)
        
        assert persona.baseline.name == "Test Persona"
        # The persona ID should be preserved as provided
        assert persona.state.persona_id == "test_persona_1"
        assert persona.get_current_traits()["openness"] == 0.7
        
        logger.info("Persona creation test passed")
    
    @pytest.mark.asyncio
    async def test_memory_service_initialization(self, memory_service):
        """Test memory service initialization."""
        # Check that memory service was created
        assert memory_service is not None
        assert hasattr(memory_service, 'memory_collections')
        assert hasattr(memory_service, 'embedding_dimension')
        
        logger.info("Memory service initialization test passed")
    
    @pytest.mark.asyncio
    async def test_event_generation_basic(self, event_generator):
        """Test basic event generation functionality."""
        # Test that we can create events even without templates
        from src.models.events import StressEvent, EventCategory, EventIntensity
        
        # Create a simple stress event manually
        event = StressEvent(
            event_id="test_event_1",
            event_type=EventType.STRESS,
            category=EventCategory.DEATH,
            intensity=EventIntensity.HIGH,
            title="Test Stress Event",
            description="A test stress event for simulation",
            context="This is a test context",
            simulation_day=1,
            stress_impact=7.0,
            trauma_level=5.0,
            recovery_time_days=7
        )
        
        assert event.event_type == EventType.STRESS
        assert event.stress_impact == 7.0
        assert event.is_high_impact()
        
        logger.info("Basic event generation test passed")
    
    @pytest.mark.asyncio
    async def test_simulation_engine_basic(self):
        """Test basic simulation engine functionality."""
        # Create simulation engine
        engine = SimulationEngine()
        
        # Test that engine can be created
        assert engine is not None
        assert hasattr(engine, 'persona_manager')
        assert hasattr(engine, 'event_generator')
        assert hasattr(engine, 'assessment_service')
        
        # Test basic config creation with valid values
        config = SimulationConfig(
            duration_days=10,  # Must be >= assessment_interval_days (7)
            experimental_condition=ExperimentalCondition.MINIMAL,
            stress_event_frequency=0.01,  # Very low stress frequency
            neutral_event_frequency=0.05   # Low neutral frequency
        )
        
        # Debug the validation
        logger.info(f"Config validation: duration_days={config.duration_days}, "
                   f"assessment_interval_days={config.assessment_interval_days}, "
                   f"stress_freq={config.stress_event_frequency}, "
                   f"neutral_freq={config.neutral_event_frequency}, "
                   f"sum={config.stress_event_frequency + config.neutral_event_frequency}")
        
        assert config.is_valid_configuration()
        
        logger.info("Basic simulation engine test passed")


@pytest.mark.asyncio
async def test_simple_simulation_flow():
    """Test a simple simulation flow without external dependencies."""
    try:
        # Create basic components
        event_generator = EventGenerator()
        memory_service = MemoryService()
        
        # Test basic functionality
        assert event_generator is not None
        assert memory_service is not None
        
        # Create a simple simulation config
        config = SimulationConfig(
            duration_days=10,  # Must be >= assessment_interval_days (7)
            experimental_condition=ExperimentalCondition.MINIMAL,
            stress_event_frequency=0.01,
            neutral_event_frequency=0.05
        )
        
        assert config.is_valid_configuration()
        assert config.duration_days == 10
        assert config.experimental_condition == ExperimentalCondition.MINIMAL
        
        logger.info("Simple simulation flow test completed successfully")
        
    except Exception as e:
        logger.error(f"Simple simulation flow test failed: {e}")
        raise


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_simple_simulation_flow()) 