"""
Memory Manager: Temporal memory with personality-specific encoding.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TYPE_CHECKING

import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)

if TYPE_CHECKING:
    from glitch_core.config.settings import Settings

from glitch_core.config.logging import get_logger
from .models import MemoryRecord
from .compressor import MemoryCompressor, CompressedMemory
from .temporal_decay import TemporalDecay, DecayConfig
from .relevance_scorer import RelevanceScorer, RelevanceConfig, RelevanceScore
from .visualizer import MemoryVisualizer, MemoryVisualizationData


class MemoryManager:
    """
    Enhanced memory manager with Phase 2.1 optimizations:
    - Memory compression for long simulations
    - Temporal decay implementation
    - Relevance scoring for memory retrieval
    - Memory visualization tools
    """
    
    def __init__(self, settings: "Settings"):
        self.settings = settings
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
        
        # Phase 2.1 Components
        self.compressor = MemoryCompressor()
        self.temporal_decay = TemporalDecay()
        self.relevance_scorer = RelevanceScorer()
        self.visualizer = MemoryVisualizer()
        
        # Memory cache for optimization
        self._memory_cache: Dict[str, List[MemoryRecord]] = {}
        
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
                self.logger.info("qdrant_collection_created", collection_name=self.collection_name)
            
            # Test Redis connection
            await self.redis_client.ping()
            self.logger.info("redis_connection_established")
            
        except Exception as e:
            self.logger.error("memory_manager_initialization_failed", error=str(e))
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

    async def compress_memories_for_experiment(
        self, 
        experiment_id: str, 
        compression_threshold: float = 0.8
    ) -> List[CompressedMemory]:
        """
        Compress memories for a specific experiment to optimize storage.
        
        Args:
            experiment_id: ID of the experiment
            compression_threshold: Similarity threshold for compression
            
        Returns:
            List of compressed memories
        """
        try:
            # Get all memories for the experiment
            memories = await self._get_all_memories_for_experiment(experiment_id)
            
            if len(memories) < 10:
                self.logger.info("insufficient_memories_for_compression", count=len(memories))
                return []
            
            # Compress memories
            compressed_memories = self.compressor.compress_memories(
                memories, compression_threshold
            )
            
            # Store compressed memories
            await self._store_compressed_memories(experiment_id, compressed_memories)
            
            self.logger.info(
                "memories_compressed",
                experiment_id=experiment_id,
                original_count=len(memories),
                compressed_count=len(compressed_memories)
            )
            
            return compressed_memories
            
        except Exception as e:
            self.logger.error("memory_compression_failed", experiment_id=experiment_id, error=str(e))
            return []
    
    async def get_decay_analysis(
        self, 
        experiment_id: str,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get temporal decay analysis for experiment memories.
        
        Args:
            experiment_id: ID of the experiment
            current_time: Current time for decay calculation
            
        Returns:
            Dictionary with decay statistics
        """
        try:
            memories = await self._get_all_memories_for_experiment(experiment_id)
            current_time = current_time or datetime.utcnow()
            
            # Calculate decay statistics
            decay_stats = self.temporal_decay.get_decay_statistics(memories, current_time)
            
            # Find forgotten memories
            forgotten_ids = self.temporal_decay.find_forgotten_memories(
                memories, current_time
            )
            
            decay_stats["forgotten_memory_ids"] = forgotten_ids
            decay_stats["experiment_id"] = experiment_id
            
            return decay_stats
            
        except Exception as e:
            self.logger.error("decay_analysis_failed", experiment_id=experiment_id, error=str(e))
            return {}
    
    async def retrieve_with_relevance_scoring(
        self,
        query: str,
        emotional_state: Dict[str, float],
        experiment_id: str,
        limit: int = 5,
        min_score: float = 0.1,
        current_time: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
        persona_traits: Optional[Dict[str, float]] = None
    ) -> List[RelevanceScore]:
        """
        Retrieve memories with advanced relevance scoring.
        
        Args:
            query: Text query for content relevance
            emotional_state: Current emotional state
            experiment_id: ID of the experiment
            limit: Maximum number of memories to return
            min_score: Minimum relevance score threshold
            current_time: Current time for temporal relevance
            context: Current context for contextual relevance
            persona_traits: Current persona traits for persona relevance
            
        Returns:
            List of relevance scores sorted by total score
        """
        try:
            # Get all memories for the experiment
            memories = await self._get_all_memories_for_experiment(experiment_id)
            
            if not memories:
                return []
            
            # Score memories for relevance
            relevance_scores = self.relevance_scorer.score_memories(
                memories, query, emotional_state, current_time, context, persona_traits
            )
            
            # Filter by minimum score and limit
            filtered_scores = [
                score for score in relevance_scores 
                if score.total_score >= min_score
            ][:limit]
            
            self.logger.info(
                "relevance_scored_retrieval",
                experiment_id=experiment_id,
                total_memories=len(memories),
                returned_count=len(filtered_scores),
                top_score=filtered_scores[0].total_score if filtered_scores else 0
            )
            
            return filtered_scores
            
        except Exception as e:
            self.logger.error("relevance_scored_retrieval_failed", experiment_id=experiment_id, error=str(e))
            return []
    
    async def create_memory_visualizations(
        self,
        experiment_id: str,
        output_dir: str = "memory_visualizations"
    ) -> Dict[str, str]:
        """
        Create comprehensive memory visualizations for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping visualization type to file path
        """
        try:
            # Get experiment data
            memories = await self._get_all_memories_for_experiment(experiment_id)
            compressed_memories = await self._get_compressed_memories(experiment_id)
            
            if not memories:
                return {"error": "No memories found for experiment"}
            
            # Create visualization data
            current_time = datetime.utcnow()
            decay_strengths = self.temporal_decay.calculate_decay_batch(memories, current_time)
            
            # Sample relevance scores for visualization
            sample_memories = memories[:min(50, len(memories))]
            relevance_scores = self.relevance_scorer.score_memories(
                sample_memories, "sample query", {"joy": 0.5}, current_time
            )
            
            viz_data = MemoryVisualizationData(
                memories=memories,
                compressed_memories=compressed_memories,
                relevance_scores=relevance_scores,
                decay_strengths=decay_strengths,
                time_range=(min(m.timestamp for m in memories), max(m.timestamp for m in memories)),
                memory_types=list(set(m.memory_type for m in memories)),
                emotional_dimensions=list(set().union(*[m.persona_bias.keys() for m in memories]))
            )
            
            # Create visualizations
            visualizations = {}
            
            # Timeline visualization
            timeline_fig = self.visualizer.create_memory_timeline(memories)
            timeline_path = f"{output_dir}/timeline_{experiment_id}.png"
            self.visualizer.save_visualization(timeline_fig, timeline_path)
            visualizations["timeline"] = timeline_path
            plt.close(timeline_fig)
            
            # Emotional distribution
            emotional_fig = self.visualizer.create_emotional_distribution(memories)
            emotional_path = f"{output_dir}/emotional_distribution_{experiment_id}.png"
            self.visualizer.save_visualization(emotional_fig, emotional_path)
            visualizations["emotional_distribution"] = emotional_path
            plt.close(emotional_fig)
            
            # Decay analysis
            decay_fig = self.visualizer.create_decay_analysis(memories, current_time, self.temporal_decay)
            decay_path = f"{output_dir}/decay_analysis_{experiment_id}.png"
            self.visualizer.save_visualization(decay_fig, decay_path)
            visualizations["decay_analysis"] = decay_path
            plt.close(decay_fig)
            
            # Relevance analysis
            if relevance_scores:
                relevance_fig = self.visualizer.create_relevance_analysis(relevance_scores)
                relevance_path = f"{output_dir}/relevance_analysis_{experiment_id}.png"
                self.visualizer.save_visualization(relevance_fig, relevance_path)
                visualizations["relevance_analysis"] = relevance_path
                plt.close(relevance_fig)
            
            # Compression analysis
            if compressed_memories:
                compression_fig = self.visualizer.create_compression_analysis(memories, compressed_memories)
                compression_path = f"{output_dir}/compression_analysis_{experiment_id}.png"
                self.visualizer.save_visualization(compression_fig, compression_path)
                visualizations["compression_analysis"] = compression_path
                plt.close(compression_fig)
            
            # Summary report
            summary_report = self.visualizer.create_memory_summary_report(viz_data)
            summary_path = f"{output_dir}/summary_report_{experiment_id}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)
            visualizations["summary_report"] = summary_path
            
            self.logger.info(
                "memory_visualizations_created",
                experiment_id=experiment_id,
                visualization_count=len(visualizations)
            )
            
            return visualizations
            
        except Exception as e:
            self.logger.error("memory_visualization_failed", experiment_id=experiment_id, error=str(e))
            return {"error": f"Visualization failed: {str(e)}"}
    
    async def get_memory_performance_metrics(
        self, 
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for memory operations.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            memories = await self._get_all_memories_for_experiment(experiment_id)
            compressed_memories = await self._get_compressed_memories(experiment_id)
            
            # Basic metrics
            total_memories = len(memories)
            memory_types = list(set(m.memory_type for m in memories))
            
            # Compression metrics
            compression_efficiency = 0
            if total_memories > 0:
                compression_efficiency = len(compressed_memories) / total_memories
            
            # Decay metrics
            current_time = datetime.utcnow()
            decay_strengths = self.temporal_decay.calculate_decay_batch(memories, current_time)
            avg_decay_strength = sum(decay_strengths.values()) / len(decay_strengths) if decay_strengths else 0
            
            # Relevance metrics (sample)
            sample_memories = memories[:min(20, len(memories))]
            if sample_memories:
                relevance_scores = self.relevance_scorer.score_memories(
                    sample_memories, "performance test", {"joy": 0.5}, current_time
                )
                avg_relevance = sum(rs.total_score for rs in relevance_scores) / len(relevance_scores) if relevance_scores else 0
            else:
                avg_relevance = 0
            
            metrics = {
                "experiment_id": experiment_id,
                "total_memories": total_memories,
                "memory_types": memory_types,
                "compression_efficiency": compression_efficiency,
                "compressed_memory_count": len(compressed_memories),
                "average_decay_strength": avg_decay_strength,
                "average_relevance_score": avg_relevance,
                "forgotten_memories": len([s for s in decay_strengths.values() if s <= 0.01]),
                "strong_memories": len([s for s in decay_strengths.values() if s > 0.7]),
                "memory_age_hours": {
                    "min": min((current_time - m.timestamp).total_seconds() / 3600 for m in memories) if memories else 0,
                    "max": max((current_time - m.timestamp).total_seconds() / 3600 for m in memories) if memories else 0,
                    "avg": sum((current_time - m.timestamp).total_seconds() / 3600 for m in memories) / len(memories) if memories else 0
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error("performance_metrics_failed", experiment_id=experiment_id, error=str(e))
            return {"error": f"Performance metrics failed: {str(e)}"}
    
    async def _get_all_memories_for_experiment(self, experiment_id: str) -> List[MemoryRecord]:
        """Get all memories for an experiment from cache or storage."""
        if experiment_id in self._memory_cache:
            return self._memory_cache[experiment_id]
        
        # Get from Redis (recent memories)
        recent_memories = await self._get_recent_memories(experiment_id, limit=1000)
        
        # Get from Qdrant (all memories)
        all_memories = await self._get_similar_memories("", limit=1000)
        
        # Combine and deduplicate
        all_memory_dict = {m.id: m for m in recent_memories + all_memories}
        memories = list(all_memory_dict.values())
        
        # Cache for performance
        self._memory_cache[experiment_id] = memories
        
        return memories
    
    async def _get_compressed_memories(self, experiment_id: str) -> List[CompressedMemory]:
        """Get compressed memories for an experiment."""
        try:
            compressed_data = await self.redis_client.hgetall(f"compressed_memories:{experiment_id}")
            compressed_memories = []
            
            for memory_id, data in compressed_data.items():
                try:
                    memory_dict = json.loads(data)
                    compressed_memory = CompressedMemory.from_dict(memory_dict)
                    compressed_memories.append(compressed_memory)
                except Exception as e:
                    self.logger.warning("failed_to_parse_compressed_memory", memory_id=memory_id, error=str(e))
            
            return compressed_memories
            
        except Exception as e:
            self.logger.error("failed_to_get_compressed_memories", experiment_id=experiment_id, error=str(e))
            return []
    
    async def _store_compressed_memories(
        self, 
        experiment_id: str, 
        compressed_memories: List[CompressedMemory]
    ):
        """Store compressed memories in Redis."""
        try:
            for compressed_memory in compressed_memories:
                memory_data = json.dumps(compressed_memory.to_dict())
                await self.redis_client.hset(
                    f"compressed_memories:{experiment_id}",
                    compressed_memory.id,
                    memory_data
                )
            
            self.logger.info(
                "compressed_memories_stored",
                experiment_id=experiment_id,
                count=len(compressed_memories)
            )
            
        except Exception as e:
            self.logger.error("failed_to_store_compressed_memories", experiment_id=experiment_id, error=str(e)) 