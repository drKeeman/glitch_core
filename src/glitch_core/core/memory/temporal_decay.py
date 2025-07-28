"""
Temporal decay implementation for memory management.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np

from glitch_core.config.logging import get_logger
from .manager import MemoryRecord


@dataclass
class DecayConfig:
    """Configuration for temporal decay calculations."""
    base_decay_rate: float = 0.1  # Base decay rate per time unit
    emotional_modifier: float = 0.5  # How much emotional weight affects decay
    persona_modifier: float = 0.3  # How much persona bias affects decay
    context_modifier: float = 0.2  # How much context affects decay
    min_strength: float = 0.01  # Minimum memory strength before forgetting
    max_strength: float = 1.0  # Maximum memory strength
    time_unit: str = "hours"  # Time unit for decay calculations


class TemporalDecay:
    """
    Implements temporal decay for memories based on psychological research.
    """
    
    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()
        self.logger = get_logger("temporal_decay")
        
        # Decay functions for different memory types
        self.decay_functions = {
            "event": self._exponential_decay,
            "reflection": self._power_law_decay,
            "intervention": self._intervention_decay,
            "trauma": self._trauma_decay,
            "success": self._success_decay
        }
    
    def calculate_decay(
        self, 
        memory: MemoryRecord, 
        current_time: datetime,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate the current strength of a memory after temporal decay.
        
        Args:
            memory: Memory record to calculate decay for
            current_time: Current time for decay calculation
            emotional_state: Current emotional state (affects decay)
            
        Returns:
            Memory strength (0.0-1.0) after decay
        """
        try:
            # Calculate time difference
            time_diff = current_time - memory.timestamp
            time_units = self._convert_to_time_units(time_diff)
            
            # Get base decay rate for this memory type
            base_rate = memory.decay_rate
            
            # Apply emotional state modifier
            emotional_modifier = self._calculate_emotional_modifier(
                memory, emotional_state
            )
            
            # Apply persona bias modifier
            persona_modifier = self._calculate_persona_modifier(memory)
            
            # Apply context modifier
            context_modifier = self._calculate_context_modifier(memory)
            
            # Calculate effective decay rate
            effective_rate = base_rate * emotional_modifier * persona_modifier * context_modifier
            
            # Get appropriate decay function
            decay_func = self.decay_functions.get(
                memory.memory_type, 
                self._exponential_decay
            )
            
            # Calculate decayed strength
            strength = decay_func(time_units, effective_rate, memory.emotional_weight)
            
            # Apply bounds
            strength = max(self.config.min_strength, min(self.config.max_strength, strength))
            
            return strength
            
        except Exception as e:
            self.logger.error("decay_calculation_failed", error=str(e), memory_id=memory.id)
            return self.config.min_strength
    
    def calculate_decay_batch(
        self, 
        memories: List[MemoryRecord], 
        current_time: datetime,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate decay for multiple memories efficiently.
        
        Args:
            memories: List of memory records
            current_time: Current time for decay calculation
            emotional_state: Current emotional state
            
        Returns:
            Dictionary mapping memory ID to decayed strength
        """
        results = {}
        
        for memory in memories:
            strength = self.calculate_decay(memory, current_time, emotional_state)
            results[memory.id] = strength
        
        return results
    
    def find_forgotten_memories(
        self, 
        memories: List[MemoryRecord], 
        current_time: datetime,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Find memories that have decayed below the minimum threshold.
        
        Args:
            memories: List of memory records
            current_time: Current time for decay calculation
            emotional_state: Current emotional state
            
        Returns:
            List of memory IDs that should be forgotten
        """
        forgotten_ids = []
        
        for memory in memories:
            strength = self.calculate_decay(memory, current_time, emotional_state)
            if strength <= self.config.min_strength:
                forgotten_ids.append(memory.id)
        
        return forgotten_ids
    
    def _convert_to_time_units(self, time_diff: timedelta) -> float:
        """Convert timedelta to the configured time units."""
        if self.config.time_unit == "hours":
            return time_diff.total_seconds() / 3600
        elif self.config.time_unit == "days":
            return time_diff.total_seconds() / 86400
        elif self.config.time_unit == "minutes":
            return time_diff.total_seconds() / 60
        else:
            return time_diff.total_seconds() / 3600  # Default to hours
    
    def _calculate_emotional_modifier(
        self, 
        memory: MemoryRecord, 
        emotional_state: Optional[Dict[str, float]]
    ) -> float:
        """Calculate how emotional state affects decay."""
        if not emotional_state:
            return 1.0
        
        # Calculate emotional congruence
        memory_emotions = memory.persona_bias
        current_emotions = emotional_state
        
        # Find common emotions
        common_emotions = set(memory_emotions.keys()) & set(current_emotions.keys())
        
        if not common_emotions:
            return 1.0
        
        # Calculate correlation between memory and current emotions
        memory_values = [memory_emotions.get(emotion, 0.0) for emotion in common_emotions]
        current_values = [current_emotions.get(emotion, 0.0) for emotion in common_emotions]
        
        try:
            correlation = np.corrcoef(memory_values, current_values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except Exception:
            correlation = 0.0
        
        # Higher correlation = slower decay
        modifier = 1.0 - (self.config.emotional_modifier * (1.0 - correlation))
        return max(0.1, modifier)  # Ensure minimum decay rate
    
    def _calculate_persona_modifier(self, memory: MemoryRecord) -> float:
        """Calculate how persona bias affects decay."""
        # High neuroticism = faster decay for negative memories
        # High extraversion = slower decay for social memories
        # High conscientiousness = slower decay for structured memories
        
        neuroticism = memory.persona_bias.get("neuroticism", 0.5)
        extraversion = memory.persona_bias.get("extraversion", 0.5)
        conscientiousness = memory.persona_bias.get("conscientiousness", 0.5)
        
        # Calculate modifier based on personality traits
        modifier = 1.0
        
        # Neuroticism effect (faster decay for negative memories)
        if memory.emotional_weight < 0.5:  # Negative memory
            modifier *= (1.0 + neuroticism * 0.3)
        
        # Extraversion effect (slower decay for social memories)
        if "social" in memory.context.get("source", "").lower():
            modifier *= (1.0 - extraversion * 0.2)
        
        # Conscientiousness effect (slower decay for structured memories)
        if memory.memory_type == "reflection":
            modifier *= (1.0 - conscientiousness * 0.2)
        
        return max(0.1, modifier)
    
    def _calculate_context_modifier(self, memory: MemoryRecord) -> float:
        """Calculate how context affects decay."""
        context = memory.context
        modifier = 1.0
        
        # High significance = slower decay
        significance = context.get("significance", "medium")
        if significance == "high":
            modifier *= 0.7
        elif significance == "low":
            modifier *= 1.3
        
        # Novel events decay slower
        if context.get("novelty", False):
            modifier *= 0.8
        
        # Routine events decay faster
        if context.get("routine", False):
            modifier *= 1.2
        
        return max(0.1, modifier)
    
    def _exponential_decay(
        self, 
        time_units: float, 
        decay_rate: float, 
        emotional_weight: float
    ) -> float:
        """Exponential decay function (Ebbinghaus forgetting curve)."""
        # Emotional weight affects initial strength
        initial_strength = emotional_weight
        
        # Apply exponential decay
        strength = initial_strength * math.exp(-decay_rate * time_units)
        
        return strength
    
    def _power_law_decay(
        self, 
        time_units: float, 
        decay_rate: float, 
        emotional_weight: float
    ) -> float:
        """Power law decay for reflections and complex memories."""
        # Power law decay: strength = initial / (1 + rate * time)^alpha
        alpha = 0.5  # Decay exponent
        
        initial_strength = emotional_weight
        strength = initial_strength / (1 + decay_rate * time_units) ** alpha
        
        return strength
    
    def _intervention_decay(
        self, 
        time_units: float, 
        decay_rate: float, 
        emotional_weight: float
    ) -> float:
        """Specialized decay for intervention memories."""
        # Interventions have longer-lasting effects
        # Initial rapid decay, then slower decay
        
        if time_units < 24:  # First 24 hours
            # Rapid initial decay
            strength = emotional_weight * math.exp(-decay_rate * 2 * time_units)
        else:
            # Slower long-term decay
            remaining_time = time_units - 24
            initial_strength = emotional_weight * math.exp(-decay_rate * 2 * 24)
            strength = initial_strength * math.exp(-decay_rate * 0.5 * remaining_time)
        
        return strength
    
    def _trauma_decay(
        self, 
        time_units: float, 
        decay_rate: float, 
        emotional_weight: float
    ) -> float:
        """Specialized decay for traumatic memories."""
        # Traumatic memories have very slow decay
        # They may even strengthen over time (flashbulb memories)
        
        # Calculate base decay
        base_strength = emotional_weight * math.exp(-decay_rate * 0.1 * time_units)
        
        # Add flashbulb effect (memories can strengthen)
        flashbulb_effect = 0.1 * math.sin(time_units / 24)  # Daily cycle
        
        strength = base_strength + flashbulb_effect
        return min(1.0, strength)
    
    def _success_decay(
        self, 
        time_units: float, 
        decay_rate: float, 
        emotional_weight: float
    ) -> float:
        """Specialized decay for success memories."""
        # Success memories decay slowly and can strengthen
        # Positive reinforcement effect
        
        # Slower base decay
        base_strength = emotional_weight * math.exp(-decay_rate * 0.5 * time_units)
        
        # Positive reinforcement effect
        reinforcement = 0.05 * math.exp(-time_units / 168)  # Weekly cycle
        
        strength = base_strength + reinforcement
        return min(1.0, strength)
    
    def get_decay_statistics(
        self, 
        memories: List[MemoryRecord], 
        current_time: datetime
    ) -> Dict[str, Any]:
        """
        Get statistics about memory decay patterns.
        
        Args:
            memories: List of memory records
            current_time: Current time for calculation
            
        Returns:
            Dictionary with decay statistics
        """
        if not memories:
            return {}
        
        strengths = []
        ages = []
        types = {}
        
        for memory in memories:
            strength = self.calculate_decay(memory, current_time)
            age = (current_time - memory.timestamp).total_seconds() / 3600  # hours
            
            strengths.append(strength)
            ages.append(age)
            
            # Group by memory type
            memory_type = memory.memory_type
            if memory_type not in types:
                types[memory_type] = []
            types[memory_type].append(strength)
        
        # Calculate statistics
        stats = {
            "total_memories": len(memories),
            "average_strength": np.mean(strengths),
            "median_strength": np.median(strengths),
            "strength_std": np.std(strengths),
            "average_age_hours": np.mean(ages),
            "forgotten_count": len([s for s in strengths if s <= self.config.min_strength]),
            "strong_memories": len([s for s in strengths if s > 0.7]),
            "by_type": {}
        }
        
        # Statistics by memory type
        for memory_type, type_strengths in types.items():
            stats["by_type"][memory_type] = {
                "count": len(type_strengths),
                "average_strength": np.mean(type_strengths),
                "median_strength": np.median(type_strengths)
            }
        
        return stats 