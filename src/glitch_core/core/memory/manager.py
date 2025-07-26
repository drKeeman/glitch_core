"""
Memory Manager: Temporal memory with personality-specific encoding.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)

from glitch_core.config.settings import get_settings
from glitch_core.config.logging import get_logger


@dataclass
class MemoryRecord:
    """A single memory record with metadata."""
    id: str
    content: str
    emotional_weight: float
    persona_bias: Dict[str, float]
    timestamp: datetime
    memory_type: str  # "event", "reflection", "intervention"
    context: Dict[str, Any]
    decay_rate: float = 0.1  # How quickly this memory fades
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "emotional_weight": self.emotional_weight,
            "persona_bias": self.persona_bias,
            "timestamp": self.timestamp.isoformat(),
            "memory_type": self.memory_type,
            "context": self.context,
            "decay_rate": self.decay_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            emotional_weight=data["emotional_weight"],
            persona_bias=data["persona_bias"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            memory_type=data["memory_type"],
            context=data["context"],
            decay_rate=data.get("decay_rate", 0.1)
        )


class MemoryManager:
    """
    Qdrant for semantic similarity
    Redis for active context window
    Personality-biased encoding/retrieval
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("memory_manager")
        
        # Qdrant client for vector storage
        self.qdrant_client = AsyncQdrantClient(
            url=self.settings.QDRANT_URL,
            timeout=self.settings.QDRANT_TIMEOUT
        )
        
        # Redis client for active context
        self.redis_client = redis.from_url(
            self.settings.REDIS_URL,
            decode_responses=True
        )
        
        # Collection name for this experiment
        self.collection_name = "glitch_memories"
        
    async def initialize(self):
        """Initialize Qdrant collection and Redis connections."""
        try:
            # Create Qdrant collection if it doesn't exist
            collections = await self.qdrant_client.get_collections()
            if self.collection_name not in [c.name for c in collections.collections]:
                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Standard embedding size
                        distance=Distance.COSINE
                    )
                )
                self.logger.info("qdrant_collection_created", collection=self.collection_name)
            
            # Test Redis connection
            await self.redis_client.ping()
            self.logger.info("memory_manager_initialized")
            
        except Exception as e:
            self.logger.error("memory_manager_init_failed", error=str(e))
            raise
    
    async def save_memory(
        self,
        content: str,
        emotional_weight: float,
        persona_bias: Dict[str, float],
        memory_type: str = "event",
        context: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None
    ) -> MemoryRecord:
        """
        Save a memory with personality-biased encoding.
        
        Args:
            content: The memory content
            emotional_weight: Emotional intensity (0-1)
            persona_bias: Personality-specific encoding biases
            memory_type: Type of memory ("event", "reflection", "intervention")
            context: Additional context metadata
            experiment_id: Associated experiment ID
            
        Returns:
            MemoryRecord with generated ID
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create memory record
        memory = MemoryRecord(
            id=memory_id,
            content=content,
            emotional_weight=emotional_weight,
            persona_bias=persona_bias,
            timestamp=timestamp,
            memory_type=memory_type,
            context=context or {},
            decay_rate=self._calculate_decay_rate(persona_bias)
        )
        
        try:
            # Store in Redis for active context
            await self._store_in_redis(memory, experiment_id)
            
            # Store in Qdrant for semantic search
            await self._store_in_qdrant(memory)
            
            self.logger.info(
                "memory_saved",
                memory_id=memory_id,
                memory_type=memory_type,
                emotional_weight=emotional_weight,
                experiment_id=experiment_id
            )
            
            return memory
            
        except Exception as e:
            self.logger.error(
                "memory_save_failed",
                memory_id=memory_id,
                error=str(e)
            )
            raise
    
    async def retrieve_contextual(
        self,
        query: str,
        emotional_state: Dict[str, float],
        limit: int = 5,
        experiment_id: Optional[str] = None
    ) -> List[MemoryRecord]:
        """
        Retrieve memories based on semantic similarity and emotional state.
        
        Args:
            query: Search query
            emotional_state: Current emotional state
            limit: Maximum number of memories to retrieve
            experiment_id: Filter by experiment ID
            
        Returns:
            List of relevant MemoryRecord objects
        """
        try:
            # Get recent memories from Redis first
            recent_memories = await self._get_recent_memories(experiment_id, limit=limit//2)
            
            # Get semantically similar memories from Qdrant
            similar_memories = await self._get_similar_memories(query, limit=limit//2)
            
            # Combine and rank by relevance
            all_memories = recent_memories + similar_memories
            ranked_memories = self._rank_by_relevance(all_memories, emotional_state)
            
            # Apply emotional filtering
            filtered_memories = self._filter_by_emotional_state(ranked_memories, emotional_state)
            
            self.logger.info(
                "memories_retrieved",
                query=query,
                count=len(filtered_memories),
                experiment_id=experiment_id
            )
            
            return filtered_memories[:limit]
            
        except Exception as e:
            self.logger.error("memory_retrieval_failed", error=str(e))
            return []
    
    async def get_memory_statistics(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics for monitoring."""
        try:
            # Get Redis stats
            redis_key_pattern = f"memory:{experiment_id}:*" if experiment_id else "memory:*"
            redis_keys = await self.redis_client.keys(redis_key_pattern)
            
            # Get Qdrant collection info
            collection_info = await self.qdrant_client.get_collection(self.collection_name)
            
            return {
                "redis_memories": len(redis_keys),
                "qdrant_points": collection_info.points_count,
                "experiment_id": experiment_id
            }
            
        except Exception as e:
            self.logger.error("memory_stats_failed", error=str(e))
            return {}
    
    def _calculate_decay_rate(self, persona_bias: Dict[str, float]) -> float:
        """Calculate memory decay rate based on personality biases."""
        # Higher neuroticism = faster decay of positive memories
        # Higher optimism = slower decay of positive memories
        neuroticism = persona_bias.get("neuroticism", 0.5)
        optimism = persona_bias.get("optimism_bias", 0.5)
        
        base_decay = 0.1
        neuroticism_factor = neuroticism * 0.2  # Faster decay
        optimism_factor = (1 - optimism) * 0.1  # Slower decay for optimists
        
        return max(0.01, min(0.5, base_decay + neuroticism_factor - optimism_factor))
    
    async def _store_in_redis(self, memory: MemoryRecord, experiment_id: Optional[str] = None):
        """Store memory in Redis for active context."""
        key = f"memory:{experiment_id or 'global'}:{memory.id}"
        data = memory.to_dict()
        
        # Set with TTL based on decay rate (inverse relationship)
        ttl_seconds = int(3600 / memory.decay_rate)  # 1 hour / decay_rate
        
        await self.redis_client.setex(
            key,
            ttl_seconds,
            json.dumps(data)
        )
    
    async def _store_in_qdrant(self, memory: MemoryRecord):
        """Store memory in Qdrant for semantic search."""
        # Simple embedding simulation (in production, use proper embedding model)
        embedding = self._generate_simple_embedding(memory.content)
        
        point = PointStruct(
            id=memory.id,
            vector=embedding,
            payload={
                "content": memory.content,
                "emotional_weight": memory.emotional_weight,
                "memory_type": memory.memory_type,
                "timestamp": memory.timestamp.isoformat(),
                "persona_bias": memory.persona_bias,
                "context": memory.context
            }
        )
        
        await self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
    
    async def _get_recent_memories(self, experiment_id: Optional[str], limit: int) -> List[MemoryRecord]:
        """Get recent memories from Redis."""
        pattern = f"memory:{experiment_id or 'global'}:*"
        keys = await self.redis_client.keys(pattern)
        
        memories = []
        for key in keys[:limit]:
            try:
                data = await self.redis_client.get(key)
                if data:
                    memory_dict = json.loads(data)
                    memories.append(MemoryRecord.from_dict(memory_dict))
            except Exception as e:
                self.logger.warning("redis_memory_parse_failed", key=key, error=str(e))
        
        return sorted(memories, key=lambda m: m.timestamp, reverse=True)
    
    async def _get_similar_memories(self, query: str, limit: int) -> List[MemoryRecord]:
        """Get semantically similar memories from Qdrant."""
        try:
            query_embedding = self._generate_simple_embedding(query)
            
            search_result = await self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            memories = []
            for point in search_result:
                payload = point.payload
                memory = MemoryRecord(
                    id=point.id,
                    content=payload["content"],
                    emotional_weight=payload["emotional_weight"],
                    persona_bias=payload["persona_bias"],
                    timestamp=datetime.fromisoformat(payload["timestamp"]),
                    memory_type=payload["memory_type"],
                    context=payload["context"]
                )
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            self.logger.error("qdrant_search_failed", error=str(e))
            return []
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate a simple embedding for semantic search."""
        # Simple hash-based embedding (replace with proper embedding model)
        import hashlib
        
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert to 384-dimensional vector
        embedding = []
        for i in range(0, len(hash_hex), 2):
            if len(embedding) >= 384:
                break
            val = int(hash_hex[i:i+2], 16) / 255.0
            embedding.append(val)
        
        # Pad to 384 dimensions
        while len(embedding) < 384:
            embedding.append(0.0)
        
        return embedding[:384]
    
    def _rank_by_relevance(self, memories: List[MemoryRecord], emotional_state: Dict[str, float]) -> List[MemoryRecord]:
        """Rank memories by emotional relevance."""
        def relevance_score(memory: MemoryRecord) -> float:
            # Emotional congruence score
            emotional_congruence = 0.0
            for emotion, intensity in emotional_state.items():
                if emotion in memory.persona_bias:
                    congruence = 1.0 - abs(intensity - memory.persona_bias[emotion])
                    emotional_congruence += congruence
            
            # Recency score
            time_diff = datetime.now() - memory.timestamp
            recency_score = 1.0 / (1.0 + time_diff.total_seconds() / 3600)  # Decay over hours
            
            # Emotional weight score
            weight_score = memory.emotional_weight
            
            return (emotional_congruence * 0.4 + recency_score * 0.3 + weight_score * 0.3)
        
        return sorted(memories, key=relevance_score, reverse=True)
    
    def _filter_by_emotional_state(self, memories: List[MemoryRecord], emotional_state: Dict[str, float]) -> List[MemoryRecord]:
        """Filter memories based on current emotional state."""
        filtered = []
        
        for memory in memories:
            # Skip memories that are too emotionally incongruent
            emotional_distance = 0.0
            for emotion, intensity in emotional_state.items():
                if emotion in memory.persona_bias:
                    distance = abs(intensity - memory.persona_bias[emotion])
                    emotional_distance += distance
            
            avg_distance = emotional_distance / len(emotional_state)
            
            # Keep memories that are reasonably emotionally congruent
            if avg_distance < 0.8:  # Threshold for emotional congruence
                filtered.append(memory)
        
        return filtered
    
    async def cleanup_old_memories(self, experiment_id: Optional[str] = None):
        """Clean up old memories to prevent storage bloat."""
        try:
            # Clean Redis (TTL handles most cleanup)
            pattern = f"memory:{experiment_id or 'global'}:*"
            keys = await self.redis_client.keys(pattern)
            
            # Clean Qdrant old memories (older than 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # This would require a more sophisticated cleanup in production
            # For now, we rely on TTL and manual cleanup
            
            self.logger.info("memory_cleanup_completed", keys_checked=len(keys))
            
        except Exception as e:
            self.logger.error("memory_cleanup_failed", error=str(e)) 