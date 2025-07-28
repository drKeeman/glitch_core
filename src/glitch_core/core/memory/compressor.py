"""
Memory compression algorithms for long simulations.
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from glitch_core.config.logging import get_logger
from .models import MemoryRecord


@dataclass
class CompressedMemory:
    """A compressed representation of multiple memories."""
    id: str
    content_summary: str
    emotional_centroid: Dict[str, float]
    persona_bias_centroid: Dict[str, float]
    memory_count: int
    time_span: Tuple[datetime, datetime]
    memory_types: List[str]
    context_summary: Dict[str, Any]
    compression_ratio: float
    original_memory_ids: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content_summary": self.content_summary,
            "emotional_centroid": self.emotional_centroid,
            "persona_bias_centroid": self.persona_bias_centroid,
            "memory_count": self.memory_count,
            "time_span": [self.time_span[0].isoformat(), self.time_span[1].isoformat()],
            "memory_types": self.memory_types,
            "context_summary": self.context_summary,
            "compression_ratio": self.compression_ratio,
            "original_memory_ids": self.original_memory_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressedMemory":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content_summary=data["content_summary"],
            emotional_centroid=data["emotional_centroid"],
            persona_bias_centroid=data["persona_bias_centroid"],
            memory_count=data["memory_count"],
            time_span=(
                datetime.fromisoformat(data["time_span"][0]),
                datetime.fromisoformat(data["time_span"][1])
            ),
            memory_types=data["memory_types"],
            context_summary=data["context_summary"],
            compression_ratio=data["compression_ratio"],
            original_memory_ids=data["original_memory_ids"]
        )


class MemoryCompressor:
    """
    Compresses memories for long simulations using clustering and summarization.
    """
    
    def __init__(self):
        self.logger = get_logger("memory_compressor")
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def compress_memories(
        self, 
        memories: List[MemoryRecord], 
        compression_threshold: float = 0.8
    ) -> List[CompressedMemory]:
        """
        Compress memories using clustering and summarization.
        
        Args:
            memories: List of memories to compress
            compression_threshold: Minimum similarity for clustering (0.0-1.0)
            
        Returns:
            List of compressed memory clusters
        """
        if len(memories) < 10:
            return []
        
        self.logger.info(
            "starting_memory_compression",
            memory_count=len(memories),
            compression_threshold=compression_threshold
        )
        
        try:
            # Extract features for clustering
            features = self._extract_clustering_features(memories)
            
            # Perform clustering
            clusters = self._cluster_memories(features, compression_threshold)
            
            # Compress each cluster
            compressed_memories = []
            for cluster_indices in clusters:
                cluster_memories = [memories[i] for i in cluster_indices]
                compressed = self._compress_cluster(cluster_memories)
                compressed_memories.append(compressed)
            
            # Calculate overall compression ratio
            total_original = len(memories)
            total_compressed = len(compressed_memories)
            overall_ratio = total_compressed / total_original if total_original > 0 else 0
            
            self.logger.info(
                "memory_compression_completed",
                original_count=total_original,
                compressed_count=total_compressed,
                compression_ratio=overall_ratio
            )
            
            return compressed_memories
            
        except Exception as e:
            self.logger.error("memory_compression_failed", error=str(e))
            return []
    
    def _extract_clustering_features(self, memories: List[MemoryRecord]) -> np.ndarray:
        """Extract features for clustering from memories."""
        features = []
        
        for memory in memories:
            # Content features
            content_vector = self._vectorize_content(memory.content)
            
            # Emotional features
            emotional_vector = list(memory.persona_bias.values())
            
            # Temporal features (normalized timestamp)
            time_features = self._extract_temporal_features(memory.timestamp, memories)
            
            # Combine all features
            combined_features = np.concatenate([
                content_vector,
                emotional_vector,
                time_features
            ])
            
            features.append(combined_features)
        
        return np.array(features)
    
    def _vectorize_content(self, content: str) -> np.ndarray:
        """Convert content to TF-IDF vector."""
        try:
            # Fit vectorizer if not already fitted
            if not hasattr(self, '_fitted_vectorizer'):
                self.vectorizer.fit([content])
                self._fitted_vectorizer = True
            
            vector = self.vectorizer.transform([content]).toarray()[0]
            
            # Pad or truncate to fixed size
            if len(vector) < 50:
                vector = np.pad(vector, (0, 50 - len(vector)))
            else:
                vector = vector[:50]
            
            return vector
            
        except Exception:
            # Fallback to simple features
            return np.zeros(50)
    
    def _extract_temporal_features(self, timestamp: datetime, all_memories: List[MemoryRecord]) -> np.ndarray:
        """Extract temporal features for clustering."""
        timestamps = [m.timestamp for m in all_memories]
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        if max_time == min_time:
            return np.array([0.5])  # Neutral position
        
        # Normalize timestamp to 0-1 range
        normalized_time = (timestamp - min_time).total_seconds() / (max_time - min_time).total_seconds()
        
        # Add cyclical time features (hour of day, day of week)
        hour_feature = timestamp.hour / 24.0
        day_feature = timestamp.weekday() / 7.0
        
        return np.array([normalized_time, hour_feature, day_feature])
    
    def _cluster_memories(self, features: np.ndarray, threshold: float) -> List[List[int]]:
        """Cluster memories using DBSCAN."""
        # Convert similarity threshold to epsilon for DBSCAN
        # Lower threshold = higher similarity required = smaller epsilon
        epsilon = 1.0 - threshold
        
        clustering = DBSCAN(
            eps=epsilon,
            min_samples=2,  # At least 2 memories per cluster
            metric='cosine'
        ).fit(features)
        
        # Group indices by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label >= 0:  # Skip noise points
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
        
        return list(clusters.values())
    
    def _compress_cluster(self, cluster_memories: List[MemoryRecord]) -> CompressedMemory:
        """Compress a cluster of memories into a single compressed memory."""
        # Generate unique ID for compressed memory
        memory_ids = [m.id for m in cluster_memories]
        compressed_id = hashlib.md5(
            json.dumps(memory_ids, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # Summarize content
        content_summary = self._summarize_content(cluster_memories)
        
        # Calculate emotional centroid
        emotional_centroid = self._calculate_emotional_centroid(cluster_memories)
        
        # Calculate persona bias centroid
        persona_bias_centroid = self._calculate_persona_bias_centroid(cluster_memories)
        
        # Calculate time span
        timestamps = [m.timestamp for m in cluster_memories]
        time_span = (min(timestamps), max(timestamps))
        
        # Collect memory types
        memory_types = list(set(m.memory_type for m in cluster_memories))
        
        # Summarize context
        context_summary = self._summarize_context(cluster_memories)
        
        # Calculate compression ratio
        compression_ratio = 1.0 / len(cluster_memories)
        
        return CompressedMemory(
            id=compressed_id,
            content_summary=content_summary,
            emotional_centroid=emotional_centroid,
            persona_bias_centroid=persona_bias_centroid,
            memory_count=len(cluster_memories),
            time_span=time_span,
            memory_types=memory_types,
            context_summary=context_summary,
            compression_ratio=compression_ratio,
            original_memory_ids=memory_ids
        )
    
    def _summarize_content(self, memories: List[MemoryRecord]) -> str:
        """Create a summary of memory contents."""
        contents = [m.content for m in memories]
        
        # Simple summarization: most common words
        all_words = []
        for content in contents:
            words = content.lower().split()
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = {}
        for word in all_words:
            if len(word) > 3:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_words:
            summary = f"Memories about: {', '.join(word for word, _ in top_words)}"
        else:
            summary = "Compressed memory cluster"
        
        return summary
    
    def _calculate_emotional_centroid(self, memories: List[MemoryRecord]) -> Dict[str, float]:
        """Calculate the centroid of emotional weights."""
        if not memories:
            return {}
        
        # Get all emotional keys
        all_keys = set()
        for memory in memories:
            all_keys.update(memory.persona_bias.keys())
        
        # Calculate average for each emotion
        centroid = {}
        for key in all_keys:
            values = [m.persona_bias.get(key, 0.0) for m in memories]
            centroid[key] = sum(values) / len(values)
        
        return centroid
    
    def _calculate_persona_bias_centroid(self, memories: List[MemoryRecord]) -> Dict[str, float]:
        """Calculate the centroid of persona biases."""
        if not memories:
            return {}
        
        # Get all bias keys
        all_keys = set()
        for memory in memories:
            all_keys.update(memory.persona_bias.keys())
        
        # Calculate average for each bias
        centroid = {}
        for key in all_keys:
            values = [m.persona_bias.get(key, 0.0) for m in memories]
            centroid[key] = sum(values) / len(values)
        
        return centroid
    
    def _summarize_context(self, memories: List[MemoryRecord]) -> Dict[str, Any]:
        """Summarize context information from memories."""
        context_summary = {
            "total_memories": len(memories),
            "memory_types": list(set(m.memory_type for m in memories)),
            "average_emotional_weight": sum(m.emotional_weight for m in memories) / len(memories),
            "context_keys": set()
        }
        
        # Collect all context keys
        for memory in memories:
            context_summary["context_keys"].update(memory.context.keys())
        
        context_summary["context_keys"] = list(context_summary["context_keys"])
        
        return context_summary 