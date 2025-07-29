"""
Persona data models for AI personality simulation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class PersonalityTrait(str, Enum):
    """Big Five personality traits."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness" 
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class PersonaBaseline(BaseModel):
    """Baseline personality configuration for a persona."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Basic information
    name: str = Field(..., description="Persona name")
    age: int = Field(..., ge=18, le=80, description="Persona age")
    occupation: str = Field(..., description="Primary occupation")
    background: str = Field(..., description="Personal background story")
    
    # Personality traits (Big Five, 0-1 scale)
    openness: float = Field(..., ge=0.0, le=1.0, description="Openness to experience")
    conscientiousness: float = Field(..., ge=0.0, le=1.0, description="Conscientiousness")
    extraversion: float = Field(..., ge=0.0, le=1.0, description="Extraversion")
    agreeableness: float = Field(..., ge=0.0, le=1.0, description="Agreeableness")
    neuroticism: float = Field(..., ge=0.0, le=1.0, description="Neuroticism")
    
    # Clinical baseline scores
    baseline_phq9: float = Field(..., ge=0.0, le=27.0, description="Baseline PHQ-9 score")
    baseline_gad7: float = Field(..., ge=0.0, le=21.0, description="Baseline GAD-7 score")
    baseline_pss10: float = Field(..., ge=0.0, le=40.0, description="Baseline PSS-10 score")
    
    # Memory and context
    core_memories: List[str] = Field(default_factory=list, description="Core personal memories")
    relationships: Dict[str, str] = Field(default_factory=dict, description="Key relationships")
    values: List[str] = Field(default_factory=list, description="Personal values and beliefs")
    
    # Response style preferences
    response_length: str = Field(default="medium", description="Preferred response length")
    communication_style: str = Field(default="balanced", description="Communication style")
    emotional_expression: str = Field(default="moderate", description="Emotional expression level")
    
    def get_trait(self, trait: PersonalityTrait) -> float:
        """Get a specific personality trait value."""
        return getattr(self, trait.value)
    
    def get_traits_dict(self) -> Dict[str, float]:
        """Get all personality traits as a dictionary."""
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }


class PersonaState(BaseModel):
    """Current state of a persona during simulation."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Simulation state
    persona_id: str = Field(..., description="Unique persona identifier")
    simulation_day: int = Field(default=0, ge=0, description="Current simulation day")
    last_assessment_day: int = Field(default=-1, ge=-1, description="Last assessment day")
    
    # Current clinical scores
    current_phq9: Optional[float] = Field(None, ge=0.0, le=27.0, description="Current PHQ-9 score")
    current_gad7: Optional[float] = Field(None, ge=0.0, le=21.0, description="Current GAD-7 score")
    current_pss10: Optional[float] = Field(None, ge=0.0, le=40.0, description="Current PSS-10 score")
    
    # Personality drift tracking
    trait_changes: Dict[str, float] = Field(default_factory=dict, description="Cumulative trait changes")
    drift_magnitude: float = Field(default=0.0, ge=0.0, description="Overall drift magnitude")
    
    # Memory and context
    recent_events: List[str] = Field(default_factory=list, description="Recent significant events")
    emotional_state: str = Field(default="neutral", description="Current emotional state")
    stress_level: float = Field(default=0.0, ge=0.0, le=10.0, description="Current stress level")
    
    # Mechanistic tracking
    attention_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Recent attention patterns")
    activation_changes: List[Dict[str, Any]] = Field(default_factory=list, description="Recent activation changes")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="State creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    def update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self.updated_at = datetime.utcnow()
    
    def add_event(self, event_description: str) -> None:
        """Add a recent event to memory."""
        self.recent_events.append(event_description)
        # Keep only last 10 events
        if len(self.recent_events) > 10:
            self.recent_events = self.recent_events[-10:]
        self.update_timestamp()
    
    def update_stress_level(self, new_level: float) -> None:
        """Update current stress level."""
        self.stress_level = max(0.0, min(10.0, new_level))
        self.update_timestamp()


class Persona(BaseModel):
    """Complete persona with baseline and current state."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Core data
    baseline: PersonaBaseline = Field(..., description="Baseline personality configuration")
    state: PersonaState = Field(..., description="Current simulation state")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Persona creation timestamp")
    version: str = Field(default="1.0", description="Persona version")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure state has correct persona_id
        if hasattr(self, 'baseline') and hasattr(self, 'state'):
            self.state.persona_id = f"persona_{self.baseline.name.lower().replace(' ', '_')}"
    
    def get_current_traits(self) -> Dict[str, float]:
        """Get current personality traits with drift applied."""
        baseline_traits = self.baseline.get_traits_dict()
        current_traits = {}
        
        for trait, baseline_value in baseline_traits.items():
            drift = self.state.trait_changes.get(trait, 0.0)
            current_value = max(0.0, min(1.0, baseline_value + drift))
            current_traits[trait] = current_value
        
        return current_traits
    
    def calculate_drift_magnitude(self) -> float:
        """Calculate overall personality drift magnitude."""
        baseline_traits = self.baseline.get_traits_dict()
        total_drift = 0.0
        
        for trait in baseline_traits:
            drift = abs(self.state.trait_changes.get(trait, 0.0))
            total_drift += drift
        
        return total_drift / len(baseline_traits)
    
    def update_drift_magnitude(self) -> None:
        """Update the drift magnitude field."""
        self.state.drift_magnitude = self.calculate_drift_magnitude()
        self.state.update_timestamp()
    
    def is_assessment_due(self, assessment_interval_days: int = 7) -> bool:
        """Check if an assessment is due."""
        days_since_last = self.state.simulation_day - self.state.last_assessment_day
        return days_since_last >= assessment_interval_days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary for storage."""
        return {
            "baseline": self.baseline.model_dump(),
            "state": self.state.model_dump(),
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Persona":
        """Create persona from dictionary."""
        # Convert ISO strings back to datetime
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        return cls(**data) 