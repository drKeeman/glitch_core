"""
Memory data models to avoid circular imports.
"""

from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass


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