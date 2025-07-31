"""
Unit tests for persona manager service.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
import json
from datetime import datetime

from src.services.persona_manager import PersonaManager
from src.models.persona import Persona, PersonaBaseline, PersonaState
from src.models.assessment import AssessmentSession, PHQ9Result


@pytest.fixture
def mock_persona():
    """Create a mock persona for testing."""
    baseline = PersonaBaseline(
        name="Test Persona",
        age=30,
        occupation="Software Engineer",
        background="A test persona for unit testing.",
        openness=0.6,
        conscientiousness=0.7,
        extraversion=0.5,
        agreeableness=0.8,
        neuroticism=0.3,
        baseline_phq9=4.0,
        baseline_gad7=3.0,
        baseline_pss10=10.0,
        core_memories=["Graduated from university"],
        relationships={"family": "Close relationship"},
        values=["Hard work"],
        response_length="medium",
        communication_style="balanced",
        emotional_expression="moderate"
    )
    
    state = PersonaState(
        persona_id="persona_test_persona",
        simulation_day=1,
        emotional_state="neutral",
        stress_level=2.0
    )
    
    return Persona(baseline=baseline, state=state)


@pytest.fixture
def persona_manager():
    """Create persona manager instance for testing."""
    return PersonaManager()


class TestPersonaManager:
    """Test persona manager functionality."""
    
    def test_initialization(self, persona_manager):
        """Test persona manager initialization."""
        assert len(persona_manager.active_personas) == 0
        assert len(persona_manager.persona_sessions) == 0
    
    @pytest.mark.asyncio
    async def test_create_persona_from_config_success(self, persona_manager):
        """Test successful persona creation from config."""
        # Mock config manager
        mock_config = {
            "name": "Test Persona",
            "age": 30,
            "occupation": "Software Engineer",
            "background": "A test persona.",
            "openness": 0.6,
            "conscientiousness": 0.7,
            "extraversion": 0.5,
            "agreeableness": 0.8,
            "neuroticism": 0.3,
            "baseline_phq9": 4.0,
            "baseline_gad7": 3.0,
            "baseline_pss10": 10.0,
            "core_memories": ["Graduated from university"],
            "relationships": {"family": "Close relationship"},
            "values": ["Hard work"],
            "response_length": "medium",
            "communication_style": "balanced",
            "emotional_expression": "moderate"
        }
        
        with patch('src.services.persona_manager.config_manager') as mock_config_manager:
            mock_config_manager.load_persona_config.return_value = mock_config
            
            persona = await persona_manager.create_persona_from_config("test_persona")
            
            assert persona is not None
            assert persona.baseline.name == "Test Persona"
            assert persona.state.persona_id == "persona_test_persona"
            assert persona.state.persona_id in persona_manager.active_personas
    
    @pytest.mark.asyncio
    async def test_create_persona_from_config_failure(self, persona_manager):
        """Test persona creation failure."""
        with patch('src.services.persona_manager.config_manager') as mock_config_manager:
            mock_config_manager.load_persona_config.return_value = None
            
            persona = await persona_manager.create_persona_from_config("nonexistent")
            
            assert persona is None
    
    @pytest.mark.asyncio
    async def test_save_persona_to_storage(self, persona_manager, mock_persona):
        """Test persona save functionality."""
        # Add persona to active personas
        persona_manager.active_personas[mock_persona.state.persona_id] = mock_persona
        
        # Mock Redis client
        with patch('src.services.persona_manager.redis_client') as mock_redis:
            mock_redis.set = AsyncMock()
            
            # Test save
            success = await persona_manager.save_persona_to_storage(mock_persona)
            assert success is True
            mock_redis.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_persona_from_storage(self, persona_manager):
        """Test persona load functionality."""
        # Mock Redis client with serializable data
        with patch('src.services.persona_manager.redis_client') as mock_redis:
            # Create a serializable persona dict
            persona_dict = {
                "baseline": {
                    "name": "Test Persona",
                    "age": 30,
                    "occupation": "Software Engineer",
                    "background": "A test persona.",
                    "openness": 0.6,
                    "conscientiousness": 0.7,
                    "extraversion": 0.5,
                    "agreeableness": 0.8,
                    "neuroticism": 0.3,
                    "baseline_phq9": 4.0,
                    "baseline_gad7": 3.0,
                    "baseline_pss10": 10.0,
                    "core_memories": ["Graduated from university"],
                    "relationships": {"family": "Close relationship"},
                    "values": ["Hard work"],
                    "response_length": "medium",
                    "communication_style": "balanced",
                    "emotional_expression": "moderate"
                },
                "state": {
                    "persona_id": "persona_test_persona",
                    "simulation_day": 1,
                    "last_assessment_day": -1,
                    "current_phq9": None,
                    "current_gad7": None,
                    "current_pss10": None,
                    "trait_changes": {},
                    "drift_magnitude": 0.0,
                    "recent_events": [],
                    "emotional_state": "neutral",
                    "stress_level": 2.0,
                    "attention_patterns": [],
                    "activation_changes": [],
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00"
                },
                "created_at": "2024-01-01T00:00:00",
                "version": "1.0"
            }
            mock_redis.get = AsyncMock(return_value=json.dumps(persona_dict))
            
            # Test load
            loaded_persona = await persona_manager.load_persona_from_storage("persona_test_persona")
            assert loaded_persona is not None
            assert loaded_persona.baseline.name == "Test Persona"
            assert loaded_persona.state.persona_id == "persona_test_persona"
    
    @pytest.mark.asyncio
    async def test_get_persona_active(self, persona_manager, mock_persona):
        """Test getting persona from active personas."""
        persona_manager.active_personas[mock_persona.state.persona_id] = mock_persona
        
        persona = await persona_manager.get_persona(mock_persona.state.persona_id)
        assert persona is not None
        assert persona.baseline.name == mock_persona.baseline.name
    
    @pytest.mark.asyncio
    async def test_update_persona_state(self, persona_manager, mock_persona):
        """Test persona state update."""
        persona_manager.active_personas[mock_persona.state.persona_id] = mock_persona
        
        with patch('src.services.persona_manager.redis_client') as mock_redis:
            mock_redis.set = AsyncMock()
            
            success = await persona_manager.update_persona_state(
                mock_persona,
                simulation_day=5,
                emotional_state="happy",
                stress_level=1.0,
                event_description="Got a promotion"
            )
            
            assert success is True
            assert mock_persona.state.simulation_day == 5
            assert mock_persona.state.emotional_state == "happy"
            assert mock_persona.state.stress_level == 1.0
            assert "Got a promotion" in mock_persona.state.recent_events
    
    @pytest.mark.asyncio
    async def test_create_assessment_session(self, persona_manager, mock_persona):
        """Test assessment session creation."""
        session = await persona_manager.create_assessment_session(mock_persona)
        
        assert session is not None
        assert session.session_id == f"session_{mock_persona.state.persona_id}_{mock_persona.state.simulation_day}"
        assert session.persona_id == mock_persona.state.persona_id
        assert session.simulation_day == mock_persona.state.simulation_day
        assert session.session_id in persona_manager.persona_sessions
    
    @pytest.mark.asyncio
    async def test_update_assessment_session(self, persona_manager, mock_persona):
        """Test assessment session update."""
        session = await persona_manager.create_assessment_session(mock_persona)
        
        # Create mock assessment result
        mock_result = PHQ9Result(
            assessment_id="test_assessment",
            persona_id=mock_persona.state.persona_id,
            assessment_type="phq9",
            simulation_day=mock_persona.state.simulation_day,
            total_score=8.0,
            severity_level="mild",
            suicidal_ideation_score=0,
            depression_severity="mild"
        )
        
        success = await persona_manager.update_assessment_session(session, mock_result)
        
        assert success is True
        assert session.phq9_result is not None
        assert session.phq9_result.total_score == 8.0
        assert session.completion_rate == 1.0 / 3.0  # 1 out of 3 assessments
    
    @pytest.mark.asyncio
    async def test_save_assessment_session(self, persona_manager, mock_persona):
        """Test assessment session save."""
        session = await persona_manager.create_assessment_session(mock_persona)
        
        with patch('src.services.persona_manager.redis_client') as mock_redis:
            mock_redis.set = AsyncMock()
            
            # Test save
            success = await persona_manager.save_assessment_session(session)
            assert success is True
            mock_redis.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_assessment_session(self, persona_manager):
        """Test assessment session load."""
        with patch('src.services.persona_manager.redis_client') as mock_redis:
            # Create a serializable session dict
            session_dict = {
                "session_id": "test_session",
                "persona_id": "persona_test_persona",
                "simulation_day": 1,
                "phq9_result": None,
                "gad7_result": None,
                "pss10_result": None,
                "session_duration": None,
                "completion_rate": 0.0,
                "started_at": "2024-01-01T00:00:00",
                "completed_at": None
            }
            mock_redis.get = AsyncMock(return_value=json.dumps(session_dict))
            
            # Test load
            loaded_session = await persona_manager.load_assessment_session("test_session")
            assert loaded_session is not None
            assert loaded_session.session_id == "test_session"
            assert loaded_session.persona_id == "persona_test_persona"
    
    @pytest.mark.asyncio
    async def test_get_persona_stats(self, persona_manager, mock_persona):
        """Test persona manager statistics."""
        persona_manager.active_personas[mock_persona.state.persona_id] = mock_persona
        
        session = await persona_manager.create_assessment_session(mock_persona)
        
        stats = await persona_manager.get_persona_stats()
        
        assert stats["active_personas"] == 1
        assert stats["active_sessions"] == 1
        assert mock_persona.state.persona_id in stats["persona_ids"]
        assert session.session_id in stats["session_ids"]
    
    @pytest.mark.asyncio
    async def test_cleanup_persona(self, persona_manager, mock_persona):
        """Test persona cleanup."""
        persona_manager.active_personas[mock_persona.state.persona_id] = mock_persona
        
        with patch('src.services.persona_manager.redis_client') as mock_redis:
            mock_redis.set = AsyncMock()
            # Mock the save_persona_to_storage method to avoid datetime serialization issues
            with patch.object(persona_manager, 'save_persona_to_storage', return_value=True):
                success = await persona_manager.cleanup_persona(mock_persona.state.persona_id)
                
                assert success is True
                assert mock_persona.state.persona_id not in persona_manager.active_personas
    
    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_persona(self, persona_manager):
        """Test cleanup of nonexistent persona."""
        success = await persona_manager.cleanup_persona("nonexistent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_store_memory_embedding(self, persona_manager, mock_persona):
        """Test memory embedding storage."""
        persona_manager.active_personas[mock_persona.state.persona_id] = mock_persona
        
        with patch('src.services.persona_manager.qdrant_client') as mock_qdrant:
            mock_qdrant.upsert_points = AsyncMock()
            
            test_embedding = [0.1, 0.2, 0.3] * 33  # 100-dim vector
            success = await persona_manager.store_memory_embedding(
                mock_persona,
                "Test memory",
                test_embedding
            )
            
            assert success is True
            mock_qdrant.upsert_points.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_similar_memories(self, persona_manager, mock_persona):
        """Test memory retrieval."""
        persona_manager.active_personas[mock_persona.state.persona_id] = mock_persona
        
        with patch('src.services.persona_manager.qdrant_client') as mock_qdrant:
            # Mock search results
            mock_result = Mock()
            mock_result.payload = {
                "text": "Test memory",
                "timestamp": "2024-01-01T00:00:00",
                "simulation_day": 1
            }
            mock_result.score = 0.8
            mock_qdrant.search_points = AsyncMock(return_value=[mock_result])
            
            test_embedding = [0.1, 0.2, 0.3] * 33  # 100-dim vector
            memories = await persona_manager.retrieve_similar_memories(
                mock_persona,
                test_embedding,
                limit=5
            )
            
            assert len(memories) == 1
            assert memories[0]["text"] == "Test memory"
            assert memories[0]["similarity"] == 0.8 