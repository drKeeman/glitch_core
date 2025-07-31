"""
Unit tests for LLM service.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.services.llm_service import LLMService
from src.models.persona import Persona, PersonaBaseline, PersonaState


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
def llm_service():
    """Create LLM service instance for testing."""
    return LLMService()


class TestLLMService:
    """Test LLM service functionality."""
    
    def test_initialization(self, llm_service):
        """Test LLM service initialization."""
        assert llm_service.model is None
        assert llm_service.tokenizer is None
        assert llm_service.is_loaded is False
        assert len(llm_service.response_cache) == 0
        assert llm_service.total_tokens_processed == 0
        assert llm_service.total_inference_time == 0.0
    
    def test_create_persona_prompt(self, llm_service, mock_persona):
        """Test persona prompt creation."""
        context = "You are being asked about your day."
        instruction = "Describe how you feel about your work."
        
        prompt = llm_service._create_persona_prompt(mock_persona, context, instruction)
        
        assert "Test Persona" in prompt
        assert "30-year-old Software Engineer" in prompt
        assert "Openness (0.60):" in prompt
        assert "Conscientiousness (0.70):" in prompt
        assert "Extraversion (0.50):" in prompt
        assert "Agreeableness (0.80):" in prompt
        assert "Neuroticism (0.30):" in prompt
        assert context in prompt
        assert instruction in prompt
    
    def test_create_assessment_prompt(self, llm_service, mock_persona):
        """Test assessment prompt creation."""
        question = "Little interest or pleasure in doing things?"
        
        prompt = llm_service._create_assessment_prompt(mock_persona, "phq9", question)
        
        assert "PHQ9" in prompt  # Fixed: should be "PHQ9" not "PHQ-9"
        assert question in prompt
        assert "Respond with ONLY a number from 0-3" in prompt
        assert "Test Persona" in prompt
    
    @pytest.mark.asyncio
    async def test_parse_assessment_response_valid(self, llm_service):
        """Test parsing valid assessment responses."""
        # Test numeric responses
        assert await llm_service.parse_assessment_response("2", "phq9") == 2
        assert await llm_service.parse_assessment_response("0", "gad7") == 0
        assert await llm_service.parse_assessment_response("3", "pss10") == 3
        
        # Test text responses
        assert await llm_service.parse_assessment_response("Nearly every day", "phq9") == 3
        assert await llm_service.parse_assessment_response("Several days", "gad7") == 1
        assert await llm_service.parse_assessment_response("Very often", "pss10") == 4
    
    @pytest.mark.asyncio
    async def test_parse_assessment_response_invalid(self, llm_service):
        """Test parsing invalid assessment responses."""
        # Test invalid responses
        assert await llm_service.parse_assessment_response("", "phq9") is None
        assert await llm_service.parse_assessment_response("invalid", "gad7") is None
        assert await llm_service.parse_assessment_response("5", "pss10") is None  # Out of range
    
    def test_get_performance_stats(self, llm_service):
        """Test performance statistics."""
        # Set some mock performance data
        llm_service.total_tokens_processed = 1000
        llm_service.total_inference_time = 10.0
        llm_service.response_cache["test"] = "response"
        
        stats = llm_service.get_performance_stats()
        
        assert stats["total_tokens_processed"] == 1000
        assert stats["total_inference_time"] == 10.0
        assert stats["average_tokens_per_second"] == 100.0
        assert stats["cache_size"] == 1
        assert stats["model_loaded"] is False
    
    @pytest.mark.asyncio
    async def test_generate_response_not_loaded(self, llm_service, mock_persona):
        """Test response generation when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await llm_service.generate_response(mock_persona, "context", "instruction")
    
    @pytest.mark.asyncio
    async def test_conduct_assessment_not_loaded(self, llm_service, mock_persona):
        """Test assessment when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await llm_service.conduct_assessment(mock_persona, "phq9") 