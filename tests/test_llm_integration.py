"""
Tests for LLM integration functionality.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock

from glitch_core.config import get_reflection_engine, get_settings
from glitch_core.core.llm import ReflectionResponse


@pytest_asyncio.fixture
async def reflection_engine():
    """Create a reflection engine for testing."""
    engine = get_reflection_engine()
    # Mock the HTTP client
    engine.client = AsyncMock()
    return engine


class TestReflectionEngine:
    """Test reflection engine functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self, reflection_engine):
        """Test reflection engine initialization."""
        # Mock successful initialization
        reflection_engine.client.get.return_value = MagicMock(status_code=200)
        
        await reflection_engine.initialize()
        
        # Verify API call was made
        reflection_engine.client.get.assert_called_once_with("/api/tags")
    
    @pytest.mark.asyncio
    async def test_generate_reflection(self, reflection_engine):
        """Test reflection generation."""
        # Mock successful API response
        mock_response = {
            "response": "This is quite interesting. I'm feeling positive about this experience.",
            "eval_duration": 1000000000  # 1 second in nanoseconds
        }
        reflection_engine.client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: mock_response
        )
        
        response = await reflection_engine.generate_reflection(
            trigger_event="A positive social interaction",
            emotional_state={"joy": 0.7, "anxiety": 0.2},
            memories=["Previous positive interactions", "Good social experiences"],
            persona_prompt="You are an optimistic personality.",
            experiment_id="test_exp"
        )
        
        assert isinstance(response, ReflectionResponse)
        assert "positive" in response.reflection.lower()
        assert response.generation_time > 0
        assert response.token_count > 0
        assert 0 <= response.confidence <= 1
        
        # Verify API call was made
        reflection_engine.client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_reflection_fallback(self, reflection_engine):
        """Test fallback reflection when API fails."""
        # Mock API failure
        reflection_engine.client.post.side_effect = Exception("API Error")
        
        response = await reflection_engine.generate_reflection(
            trigger_event="A challenging situation",
            emotional_state={"anxiety": 0.8, "joy": 0.1},
            memories=[],
            persona_prompt="You are an anxious personality.",
            experiment_id="test_exp"
        )
        
        assert isinstance(response, ReflectionResponse)
        assert len(response.reflection) > 0
        assert response.generation_time > 0
        assert response.confidence == 0.3  # Fallback confidence
    
    def test_construct_prompt(self, reflection_engine):
        """Test prompt construction."""
        prompt = reflection_engine._construct_prompt(
            trigger_event="Test event",
            emotional_state={"joy": 0.7, "anxiety": 0.3},
            memories=["Memory 1", "Memory 2"],
            persona_prompt="You are optimistic."
        )
        
        assert "Test event" in prompt
        assert "joy: 0.70" in prompt
        assert "anxiety: 0.30" in prompt
        assert "Memory 1" in prompt
        assert "Memory 2" in prompt
        assert "You are optimistic" in prompt
    
    def test_parse_response(self, reflection_engine):
        """Test response parsing."""
        # Test normal response
        response_data = {"response": "This is my reflection on the event."}
        reflection = reflection_engine._parse_response(response_data)
        assert reflection == "This is my reflection on the event."
        
        # Test response with "Reflection:" prefix
        response_data = {"response": "Reflection: This is my reflection."}
        reflection = reflection_engine._parse_response(response_data)
        assert reflection == "This is my reflection."
        
        # Test empty response
        response_data = {"response": ""}
        reflection = reflection_engine._parse_response(response_data)
        assert "processing" in reflection.lower()
    
    def test_estimate_token_count(self, reflection_engine):
        """Test token count estimation."""
        text = "This is a test text with multiple words."
        token_count = reflection_engine._estimate_token_count(text)
        
        # Should be roughly text length / 4
        expected_count = len(text) // 4
        assert token_count == expected_count
    
    def test_calculate_confidence(self, reflection_engine):
        """Test confidence calculation."""
        # Fast response (high confidence)
        response_data = {"eval_duration": 1000000000}  # 1 second
        confidence = reflection_engine._calculate_confidence(response_data)
        assert confidence > 0.8
        
        # Slow response (low confidence)
        response_data = {"eval_duration": 8000000000}  # 8 seconds
        confidence = reflection_engine._calculate_confidence(response_data)
        assert confidence < 0.3
    
    def test_analyze_emotional_impact(self, reflection_engine):
        """Test emotional impact analysis."""
        reflection = "I'm feeling happy and excited about this wonderful experience!"
        emotional_state = {"joy": 0.5, "anxiety": 0.3}
        
        impact = reflection_engine._analyze_emotional_impact(reflection, emotional_state)
        
        assert "joy" in impact
        assert impact["joy"] > 0  # Should detect positive emotions
        assert 0 <= impact["joy"] <= 1
    
    def test_generate_fallback_reflection(self, reflection_engine):
        """Test fallback reflection generation."""
        # Test with dominant joy
        response = reflection_engine._generate_fallback_reflection(
            "Positive event",
            {"joy": 0.8, "anxiety": 0.2}
        )
        
        assert isinstance(response, ReflectionResponse)
        assert "pleasant" in response.reflection.lower()
        assert response.confidence == 0.3
        
        # Test with dominant anxiety
        response = reflection_engine._generate_fallback_reflection(
            "Challenging event",
            {"anxiety": 0.8, "joy": 0.2}
        )
        
        assert "uneasy" in response.reflection.lower() or "concerned" in response.reflection.lower()
    
    @pytest.mark.asyncio
    async def test_health_check(self, reflection_engine):
        """Test health check functionality."""
        # Mock healthy response
        reflection_engine.client.get.return_value = MagicMock(status_code=200)
        
        is_healthy = await reflection_engine.health_check()
        assert is_healthy is True
        
        # Mock unhealthy response
        reflection_engine.client.get.side_effect = Exception("Connection failed")
        
        is_healthy = await reflection_engine.health_check()
        assert is_healthy is False
    
    def test_get_usage_statistics(self, reflection_engine):
        """Test usage statistics."""
        # Set some usage data
        reflection_engine.total_requests = 10
        reflection_engine.total_tokens_used = 1000
        
        stats = reflection_engine.get_usage_statistics()
        
        assert stats["total_requests"] == 10
        assert stats["total_tokens_used"] == 1000
        assert stats["average_tokens_per_request"] == 100.0
        
        # Test with zero requests
        reflection_engine.total_requests = 0
        reflection_engine.total_tokens_used = 0
        
        stats = reflection_engine.get_usage_statistics()
        assert stats["average_tokens_per_request"] == 0


class TestReflectionResponse:
    """Test reflection response functionality."""
    
    def test_reflection_response_creation(self):
        """Test creating a reflection response."""
        response = ReflectionResponse(
            reflection="This is a test reflection.",
            generation_time=1.5,
            token_count=50,
            confidence=0.8,
            emotional_impact={"joy": 0.6, "anxiety": 0.2}
        )
        
        assert response.reflection == "This is a test reflection."
        assert response.generation_time == 1.5
        assert response.token_count == 50
        assert response.confidence == 0.8
        assert response.emotional_impact["joy"] == 0.6
        assert response.emotional_impact["anxiety"] == 0.2 