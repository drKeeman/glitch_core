"""
Relevance scoring for memory retrieval.
"""

import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from glitch_core.config.logging import get_logger
from .models import MemoryRecord


@dataclass
class RelevanceScore:
    """A memory with its relevance score and breakdown."""
    memory: MemoryRecord
    total_score: float
    content_relevance: float
    emotional_relevance: float
    temporal_relevance: float
    contextual_relevance: float
    persona_relevance: float
    explanation: str


@dataclass
class RelevanceConfig:
    """Configuration for relevance scoring."""
    content_weight: float = 0.3
    emotional_weight: float = 0.25
    temporal_weight: float = 0.2
    contextual_weight: float = 0.15
    persona_weight: float = 0.1
    
    # Content relevance settings
    min_content_similarity: float = 0.1
    max_content_similarity: float = 0.9
    
    # Emotional relevance settings
    emotional_congruence_threshold: float = 0.3
    emotional_intensity_weight: float = 0.5
    
    # Temporal relevance settings
    recent_memory_bonus: float = 0.2
    temporal_decay_factor: float = 0.1
    
    # Contextual relevance settings
    context_match_bonus: float = 0.3
    context_mismatch_penalty: float = 0.2


class RelevanceScorer:
    """
    Scores memory relevance for intelligent retrieval.
    """
    
    def __init__(self, config: Optional[RelevanceConfig] = None):
        self.config = config or RelevanceConfig()
        self.logger = get_logger("relevance_scorer")
        
        # TF-IDF vectorizer for content similarity
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self._fitted_vectorizer = False
    
    def score_memories(
        self,
        memories: List[MemoryRecord],
        query: str,
        emotional_state: Dict[str, float],
        current_time: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
        persona_traits: Optional[Dict[str, float]] = None
    ) -> List[RelevanceScore]:
        """
        Score memories for relevance to the current context.
        
        Args:
            memories: List of memories to score
            query: Text query for content relevance
            emotional_state: Current emotional state
            current_time: Current time for temporal relevance
            context: Current context for contextual relevance
            persona_traits: Current persona traits for persona relevance
            
        Returns:
            List of relevance scores sorted by total score
        """
        if not memories:
            return []
        
        self.logger.info(
            "scoring_memories",
            memory_count=len(memories),
            query_length=len(query)
        )
        
        try:
            # Prepare content vectors if needed
            if query and not self._fitted_vectorizer:
                self._fit_vectorizer([m.content for m in memories] + [query])
            
            # Score each memory
            scored_memories = []
            for memory in memories:
                score = self._calculate_relevance_score(
                    memory, query, emotional_state, current_time, context, persona_traits
                )
                scored_memories.append(score)
            
            # Sort by total score (descending)
            scored_memories.sort(key=lambda x: x.total_score, reverse=True)
            
            self.logger.info(
                "memory_scoring_completed",
                scored_count=len(scored_memories),
                top_score=scored_memories[0].total_score if scored_memories else 0
            )
            
            return scored_memories
            
        except Exception as e:
            self.logger.error("memory_scoring_failed", error=str(e))
            return []
    
    def get_top_relevant_memories(
        self,
        memories: List[MemoryRecord],
        query: str,
        emotional_state: Dict[str, float],
        limit: int = 5,
        min_score: float = 0.1,
        current_time: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
        persona_traits: Optional[Dict[str, float]] = None
    ) -> List[MemoryRecord]:
        """
        Get top relevant memories for a given context.
        
        Args:
            memories: List of memories to search
            query: Text query for content relevance
            emotional_state: Current emotional state
            limit: Maximum number of memories to return
            min_score: Minimum relevance score threshold
            current_time: Current time for temporal relevance
            context: Current context for contextual relevance
            persona_traits: Current persona traits for persona relevance
            
        Returns:
            List of top relevant memories
        """
        scored_memories = self.score_memories(
            memories, query, emotional_state, current_time, context, persona_traits
        )
        
        # Filter by minimum score and take top results
        relevant_memories = [
            score.memory for score in scored_memories 
            if score.total_score >= min_score
        ][:limit]
        
        return relevant_memories
    
    def _calculate_relevance_score(
        self,
        memory: MemoryRecord,
        query: str,
        emotional_state: Dict[str, float],
        current_time: Optional[datetime],
        context: Optional[Dict[str, Any]],
        persona_traits: Optional[Dict[str, float]]
    ) -> RelevanceScore:
        """Calculate comprehensive relevance score for a memory."""
        
        # Content relevance
        content_relevance = self._calculate_content_relevance(memory, query)
        
        # Emotional relevance
        emotional_relevance = self._calculate_emotional_relevance(memory, emotional_state)
        
        # Temporal relevance
        temporal_relevance = self._calculate_temporal_relevance(memory, current_time)
        
        # Contextual relevance
        contextual_relevance = self._calculate_contextual_relevance(memory, context)
        
        # Persona relevance
        persona_relevance = self._calculate_persona_relevance(memory, persona_traits)
        
        # Calculate weighted total score
        total_score = (
            self.config.content_weight * content_relevance +
            self.config.emotional_weight * emotional_relevance +
            self.config.temporal_weight * temporal_relevance +
            self.config.contextual_weight * contextual_relevance +
            self.config.persona_weight * persona_relevance
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            content_relevance, emotional_relevance, temporal_relevance,
            contextual_relevance, persona_relevance
        )
        
        return RelevanceScore(
            memory=memory,
            total_score=total_score,
            content_relevance=content_relevance,
            emotional_relevance=emotional_relevance,
            temporal_relevance=temporal_relevance,
            contextual_relevance=contextual_relevance,
            persona_relevance=persona_relevance,
            explanation=explanation
        )
    
    def _calculate_content_relevance(self, memory: MemoryRecord, query: str) -> float:
        """Calculate content relevance based on semantic similarity."""
        if not query or not memory.content:
            return 0.5  # Neutral score
        
        try:
            # Vectorize content and query
            content_vector = self.vectorizer.transform([memory.content]).toarray()
            query_vector = self.vectorizer.transform([query]).toarray()
            
            # Calculate cosine similarity
            similarity = cosine_similarity(content_vector, query_vector)[0][0]
            
            # Normalize to 0-1 range
            similarity = max(0.0, min(1.0, similarity))
            
            # Apply bounds
            if similarity < self.config.min_content_similarity:
                similarity = 0.0
            elif similarity > self.config.max_content_similarity:
                similarity = 1.0
            
            return similarity
            
        except Exception as e:
            self.logger.warning("content_relevance_calculation_failed", error=str(e))
            return 0.5
    
    def _calculate_emotional_relevance(
        self, 
        memory: MemoryRecord, 
        emotional_state: Dict[str, float]
    ) -> float:
        """Calculate emotional relevance based on emotional congruence."""
        if not emotional_state:
            return 0.5
        
        # Calculate emotional congruence
        memory_emotions = memory.persona_bias
        current_emotions = emotional_state
        
        # Find common emotions
        common_emotions = set(memory_emotions.keys()) & set(current_emotions.keys())
        
        if not common_emotions:
            return 0.3  # Low relevance for no emotional overlap
        
        # Calculate correlation between memory and current emotions
        memory_values = [memory_emotions.get(emotion, 0.0) for emotion in common_emotions]
        current_values = [current_emotions.get(emotion, 0.0) for emotion in common_emotions]
        
        try:
            correlation = np.corrcoef(memory_values, current_values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except Exception:
            correlation = 0.0
        
        # Convert correlation to 0-1 scale
        relevance = (correlation + 1.0) / 2.0
        
        # Apply emotional intensity weight
        emotional_intensity = memory.emotional_weight
        intensity_factor = 1.0 + (emotional_intensity - 0.5) * self.config.emotional_intensity_weight
        
        relevance *= intensity_factor
        
        # Apply threshold
        if relevance < self.config.emotional_congruence_threshold:
            relevance *= 0.5  # Penalty for low congruence
        
        return max(0.0, min(1.0, relevance))
    
    def _calculate_temporal_relevance(
        self, 
        memory: MemoryRecord, 
        current_time: Optional[datetime]
    ) -> float:
        """Calculate temporal relevance based on recency."""
        if not current_time:
            return 0.5
        
        # Calculate time difference
        time_diff = current_time - memory.timestamp
        hours_diff = time_diff.total_seconds() / 3600
        
        # Recent memories get bonus
        if hours_diff < 24:  # Last 24 hours
            relevance = 1.0 - (hours_diff / 24) * 0.3
        elif hours_diff < 168:  # Last week
            relevance = 0.7 - ((hours_diff - 24) / 144) * 0.3
        else:  # Older memories
            relevance = 0.4 * math.exp(-self.config.temporal_decay_factor * (hours_diff - 168) / 24)
        
        # Apply recent memory bonus
        if hours_diff < 1:  # Very recent (last hour)
            relevance += self.config.recent_memory_bonus
        
        return max(0.0, min(1.0, relevance))
    
    def _calculate_contextual_relevance(
        self, 
        memory: MemoryRecord, 
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate contextual relevance based on context matching."""
        if not context or not memory.context:
            return 0.5
        
        relevance = 0.5  # Base relevance
        
        # Check for context key matches
        memory_context = memory.context
        current_context = context
        
        # Count matching context keys
        matching_keys = 0
        total_keys = 0
        
        for key in current_context:
            if key in memory_context:
                total_keys += 1
                # Check if values are similar
                current_value = current_context[key]
                memory_value = memory_context[key]
                
                if isinstance(current_value, (int, float)) and isinstance(memory_value, (int, float)):
                    # Numeric comparison
                    if abs(current_value - memory_value) < 0.1:
                        matching_keys += 1
                elif isinstance(current_value, str) and isinstance(memory_value, str):
                    # String comparison
                    if current_value.lower() in memory_value.lower() or memory_value.lower() in current_value.lower():
                        matching_keys += 1
                elif current_value == memory_value:
                    # Exact match
                    matching_keys += 1
        
        if total_keys > 0:
            match_ratio = matching_keys / total_keys
            if match_ratio > 0.5:
                relevance += self.config.context_match_bonus
            elif match_ratio < 0.2:
                relevance -= self.config.context_mismatch_penalty
        
        return max(0.0, min(1.0, relevance))
    
    def _calculate_persona_relevance(
        self, 
        memory: MemoryRecord, 
        persona_traits: Optional[Dict[str, float]]
    ) -> float:
        """Calculate persona relevance based on personality trait alignment."""
        if not persona_traits:
            return 0.5
        
        # Calculate trait alignment
        memory_traits = memory.persona_bias
        current_traits = persona_traits
        
        # Find common traits
        common_traits = set(memory_traits.keys()) & set(current_traits.keys())
        
        if not common_traits:
            return 0.3
        
        # Calculate average trait similarity
        similarities = []
        for trait in common_traits:
            memory_value = memory_traits[trait]
            current_value = current_traits[trait]
            
            # Calculate similarity (1 - absolute difference)
            similarity = 1.0 - abs(memory_value - current_value)
            similarities.append(similarity)
        
        relevance = np.mean(similarities)
        
        # Apply personality-specific modifiers
        if "neuroticism" in common_traits:
            # High neuroticism = more sensitive to negative memories
            neuroticism = current_traits.get("neuroticism", 0.5)
            if memory.emotional_weight < 0.5:  # Negative memory
                relevance *= (1.0 + neuroticism * 0.2)
        
        if "extraversion" in common_traits:
            # High extraversion = preference for social memories
            extraversion = current_traits.get("extraversion", 0.5)
            if "social" in memory.context.get("source", "").lower():
                relevance *= (1.0 + extraversion * 0.2)
        
        return max(0.0, min(1.0, relevance))
    
    def _fit_vectorizer(self, texts: List[str]):
        """Fit the TF-IDF vectorizer with texts."""
        try:
            self.vectorizer.fit(texts)
            self._fitted_vectorizer = True
        except Exception as e:
            self.logger.warning("vectorizer_fitting_failed", error=str(e))
    
    def _generate_explanation(
        self,
        content_relevance: float,
        emotional_relevance: float,
        temporal_relevance: float,
        contextual_relevance: float,
        persona_relevance: float
    ) -> str:
        """Generate human-readable explanation of relevance score."""
        factors = []
        
        if content_relevance > 0.7:
            factors.append("high content similarity")
        elif content_relevance < 0.3:
            factors.append("low content similarity")
        
        if emotional_relevance > 0.7:
            factors.append("strong emotional congruence")
        elif emotional_relevance < 0.3:
            factors.append("emotional mismatch")
        
        if temporal_relevance > 0.7:
            factors.append("recent memory")
        elif temporal_relevance < 0.3:
            factors.append("old memory")
        
        if contextual_relevance > 0.7:
            factors.append("context match")
        elif contextual_relevance < 0.3:
            factors.append("context mismatch")
        
        if persona_relevance > 0.7:
            factors.append("persona alignment")
        elif persona_relevance < 0.3:
            factors.append("persona mismatch")
        
        if not factors:
            return "neutral relevance across all factors"
        
        return f"Relevant due to: {', '.join(factors)}"
    
    def get_relevance_statistics(
        self,
        scored_memories: List[RelevanceScore]
    ) -> Dict[str, Any]:
        """
        Get statistics about relevance scoring.
        
        Args:
            scored_memories: List of scored memories
            
        Returns:
            Dictionary with relevance statistics
        """
        if not scored_memories:
            return {}
        
        # Extract scores
        total_scores = [score.total_score for score in scored_memories]
        content_scores = [score.content_relevance for score in scored_memories]
        emotional_scores = [score.emotional_relevance for score in scored_memories]
        temporal_scores = [score.temporal_relevance for score in scored_memories]
        contextual_scores = [score.contextual_relevance for score in scored_memories]
        persona_scores = [score.persona_relevance for score in scored_memories]
        
        # Calculate statistics
        stats = {
            "total_memories": len(scored_memories),
            "average_total_score": np.mean(total_scores),
            "median_total_score": np.median(total_scores),
            "score_std": np.std(total_scores),
            "high_relevance_count": len([s for s in total_scores if s > 0.7]),
            "low_relevance_count": len([s for s in total_scores if s < 0.3]),
            "factor_averages": {
                "content": np.mean(content_scores),
                "emotional": np.mean(emotional_scores),
                "temporal": np.mean(temporal_scores),
                "contextual": np.mean(contextual_scores),
                "persona": np.mean(persona_scores)
            },
            "factor_std": {
                "content": np.std(content_scores),
                "emotional": np.std(emotional_scores),
                "temporal": np.std(temporal_scores),
                "contextual": np.std(contextual_scores),
                "persona": np.std(persona_scores)
            }
        }
        
        return stats 