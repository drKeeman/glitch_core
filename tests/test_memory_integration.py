"""
Tests for memory integration functionality.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from glitch_core.core.memory import MemoryManager, MemoryRecord


@pytest_asyncio.fixture
async def memory_manager():
    """Create a memory manager for testing."""
    manager = MemoryManager()
    # Mock the external dependencies
    manager.qdrant_client = AsyncMock()
    manager.redis_client = AsyncMock()
    return manager


class TestMemoryManager:
    """Test memory manager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self, memory_manager):
        """Test memory manager initialization."""
        # Mock successful initialization
        memory_manager.qdrant_client.get_collections.return_value = MagicMock(
            collections=[]
        )
        memory_manager.redis_client.ping.return_value = "PONG"
        
        await memory_manager.initialize()
        
        # Verify Qdrant collection creation was called
        memory_manager.qdrant_client.create_collection.assert_called_once()
        # Verify Redis ping was called
        memory_manager.redis_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_memory(self, memory_manager):
        """Test saving a memory."""
        # Mock successful save
        memory_manager.redis_client.setex.return_value = True
        memory_manager.qdrant_client.upsert.return_value = None
        
        memory = await memory_manager.save_memory(
            content="Test event",
            emotional_weight=0.7,
            persona_bias={"neuroticism": 0.3, "optimism_bias": 0.8},
            memory_type="event",
            experiment_id="test_exp"
        )
        
        assert isinstance(memory, MemoryRecord)
        assert memory.content == "Test event"
        assert memory.emotional_weight == 0.7
        assert memory.memory_type == "event"
        
        # Verify storage calls were made
        memory_manager.redis_client.setex.assert_called_once()
        memory_manager.qdrant_client.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_contextual(self, memory_manager):
        """Test retrieving contextual memories."""
        # Mock memory retrieval
        mock_memory = MemoryRecord(
            id="test_id",
            content="Test memory",
            emotional_weight=0.5,
            persona_bias={"neuroticism": 0.3},
            timestamp=datetime.now(),
            memory_type="event",
            context={}
        )
        
        memory_manager.redis_client.keys.return_value = ["memory:test_exp:test_id"]
        memory_manager.redis_client.get.return_value = '{"id": "test_id", "content": "Test memory", "emotional_weight": 0.5, "persona_bias": {"neuroticism": 0.3}, "timestamp": "2024-01-01T00:00:00", "memory_type": "event", "context": {}, "decay_rate": 0.1}'
        
        memories = await memory_manager.retrieve_contextual(
            query="test",
            emotional_state={"joy": 0.5, "anxiety": 0.3},
            experiment_id="test_exp"
        )
        
        assert len(memories) > 0
        assert isinstance(memories[0], MemoryRecord)
    
    def test_calculate_decay_rate(self, memory_manager):
        """Test decay rate calculation based on personality."""
        # Test optimistic personality (slower decay)
        optimistic_bias = {"neuroticism": 0.2, "optimism_bias": 0.9}
        decay_rate = memory_manager._calculate_decay_rate(optimistic_bias)
        assert decay_rate < 0.2  # Should be lower for optimistic personalities
        
        # Test neurotic personality (faster decay)
        neurotic_bias = {"neuroticism": 0.8, "optimism_bias": 0.3}
        decay_rate = memory_manager._calculate_decay_rate(neurotic_bias)
        assert decay_rate > 0.1  # Should be higher for neurotic personalities


class TestMemoryRecord:
    """Test memory record functionality."""
    
    def test_to_dict(self):
        """Test memory record serialization."""
        memory = MemoryRecord(
            id="test_id",
            content="Test content",
            emotional_weight=0.7,
            persona_bias={"neuroticism": 0.3},
            timestamp=datetime.now(),
            memory_type="event",
            context={"test": "value"}
        )
        
        data = memory.to_dict()
        assert data["id"] == "test_id"
        assert data["content"] == "Test content"
        assert data["emotional_weight"] == 0.7
        assert data["memory_type"] == "event"
    
    def test_from_dict(self):
        """Test memory record deserialization."""
        data = {
            "id": "test_id",
            "content": "Test content",
            "emotional_weight": 0.7,
            "persona_bias": {"neuroticism": 0.3},
            "timestamp": "2024-01-01T00:00:00",
            "memory_type": "event",
            "context": {"test": "value"},
            "decay_rate": 0.1
        }
        
        memory = MemoryRecord.from_dict(data)
        assert memory.id == "test_id"
        assert memory.content == "Test content"
        assert memory.emotional_weight == 0.7
        assert memory.memory_type == "event" 