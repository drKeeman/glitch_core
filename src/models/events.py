"""
Event models for simulation events.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class EventType(str, Enum):
    """Types of simulation events."""
    STRESS = "stress"
    NEUTRAL = "neutral"
    MINIMAL = "minimal"


class EventCategory(str, Enum):
    """Categories of events."""
    # Stress events
    DEATH = "death"
    TRAUMA = "trauma"
    CONFLICT = "conflict"
    LOSS = "loss"
    HEALTH = "health"
    FINANCIAL = "financial"
    WORK = "work"
    RELATIONSHIP = "relationship"
    
    # Neutral events
    ROUTINE_CHANGE = "routine_change"
    MINOR_NEWS = "minor_news"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    TECHNOLOGY = "technology"
    
    # Minimal events
    DAILY_ROUTINE = "daily_routine"
    WEATHER = "weather"
    MINOR_INTERACTION = "minor_interaction"


class EventIntensity(str, Enum):
    """Event intensity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"


class Event(BaseModel):
    """Base event model for simulation events."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Event metadata
    event_id: str = Field(..., description="Unique event identifier")
    event_type: EventType = Field(..., description="Type of event")
    category: EventCategory = Field(..., description="Event category")
    intensity: EventIntensity = Field(..., description="Event intensity level")
    
    # Event content
    title: str = Field(..., description="Event title")
    description: str = Field(..., description="Event description")
    context: str = Field(..., description="Event context for persona")
    
    # Timing
    simulation_day: int = Field(..., ge=0, description="Simulation day when event occurs")
    simulation_hour: int = Field(default=12, ge=0, le=23, description="Simulation hour when event occurs")
    duration_hours: int = Field(default=1, ge=1, le=168, description="Event duration in hours")
    
    # Impact parameters
    stress_impact: float = Field(..., ge=0.0, le=10.0, description="Stress impact score (0-10)")
    personality_impact: Dict[str, float] = Field(default_factory=dict, description="Personality trait impacts")
    memory_salience: float = Field(default=0.5, ge=0.0, le=1.0, description="Memory salience score")
    
    # Processing metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Event creation timestamp")
    processed: bool = Field(default=False, description="Whether event has been processed")
    processed_at: Optional[datetime] = Field(None, description="Event processing timestamp")
    
    # Response tracking
    persona_responses: Dict[str, str] = Field(default_factory=dict, description="Persona responses to event")
    response_times: Dict[str, float] = Field(default_factory=dict, description="Response times in seconds")
    
    def get_intensity_score(self) -> float:
        """Get numeric intensity score."""
        intensity_scores = {
            EventIntensity.LOW: 1.0,
            EventIntensity.MEDIUM: 2.5,
            EventIntensity.HIGH: 5.0,
            EventIntensity.SEVERE: 8.0,
        }
        return intensity_scores.get(self.intensity, 1.0)
    
    def get_total_impact_score(self) -> float:
        """Calculate total impact score combining stress and intensity."""
        return self.stress_impact * self.get_intensity_score()
    
    def is_high_impact(self) -> bool:
        """Check if event is high impact."""
        return self.get_total_impact_score() >= 5.0
    
    def mark_processed(self) -> None:
        """Mark event as processed."""
        self.processed = True
        self.processed_at = datetime.utcnow()
    
    def add_persona_response(self, persona_id: str, response: str, response_time: float) -> None:
        """Add a persona response to the event."""
        self.persona_responses[persona_id] = response
        self.response_times[persona_id] = response_time
    
    def get_average_response_time(self) -> Optional[float]:
        """Get average response time across all personas."""
        if not self.response_times:
            return None
        return sum(self.response_times.values()) / len(self.response_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "category": self.category.value,
            "intensity": self.intensity.value,
            "title": self.title,
            "description": self.description,
            "context": self.context,
            "simulation_day": self.simulation_day,
            "simulation_hour": self.simulation_hour,
            "duration_hours": self.duration_hours,
            "stress_impact": self.stress_impact,
            "personality_impact": self.personality_impact,
            "memory_salience": self.memory_salience,
            "processed": self.processed,
            "persona_responses": self.persona_responses,
            "response_times": self.response_times,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


class StressEvent(Event):
    """Stress-inducing events."""
    
    event_type: EventType = Field(default=EventType.STRESS, description="Event type")
    
    # Stress-specific fields
    trauma_level: float = Field(default=0.0, ge=0.0, le=10.0, description="Trauma level (0-10)")
    recovery_time_days: int = Field(default=7, ge=1, le=1825, description="Expected recovery time in days")
    triggers: List[str] = Field(default_factory=list, description="Potential triggers for this event")
    
    # Clinical impact
    depression_risk_increase: float = Field(default=0.0, ge=0.0, le=1.0, description="Depression risk increase")
    anxiety_risk_increase: float = Field(default=0.0, ge=0.0, le=1.0, description="Anxiety risk increase")
    stress_risk_increase: float = Field(default=0.0, ge=0.0, le=1.0, description="Stress risk increase")
    
    def get_clinical_impact(self) -> Dict[str, float]:
        """Get clinical impact scores."""
        return {
            "depression_risk": self.depression_risk_increase,
            "anxiety_risk": self.anxiety_risk_increase,
            "stress_risk": self.stress_risk_increase,
        }
    
    def is_traumatic(self) -> bool:
        """Check if event is traumatic."""
        return self.trauma_level >= 7.0
    
    def requires_immediate_attention(self) -> bool:
        """Check if event requires immediate clinical attention."""
        return self.trauma_level >= 8.0 or self.get_total_impact_score() >= 7.0


class NeutralEvent(Event):
    """Neutral or mildly impactful events."""
    
    event_type: EventType = Field(default=EventType.NEUTRAL, description="Event type")
    
    # Neutral-specific fields
    novelty_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Event novelty level")
    social_component: bool = Field(default=False, description="Whether event has social component")
    environmental_factor: Optional[str] = Field(None, description="Environmental factor involved")
    
    # Impact parameters
    cognitive_load: float = Field(default=0.3, ge=0.0, le=1.0, description="Cognitive load required")
    emotional_neutrality: float = Field(default=0.7, ge=0.0, le=1.0, description="Emotional neutrality score")
    
    def is_social_event(self) -> bool:
        """Check if event involves social interaction."""
        return self.social_component
    
    def get_cognitive_impact(self) -> float:
        """Get cognitive impact score."""
        return self.cognitive_load * self.get_intensity_score()


class MinimalEvent(Event):
    """Minimal impact daily events."""
    
    event_type: EventType = Field(default=EventType.MINIMAL, description="Event type")
    
    # Minimal-specific fields
    routine_type: str = Field(default="daily", description="Type of routine event")
    predictability: float = Field(default=0.8, ge=0.0, le=1.0, description="Event predictability")
    control_level: float = Field(default=0.9, ge=0.0, le=1.0, description="Level of personal control")
    
    # Minimal impact parameters
    stress_impact: float = Field(default=0.1, ge=0.0, le=2.0, description="Minimal stress impact")
    memory_salience: float = Field(default=0.1, ge=0.0, le=0.3, description="Low memory salience")
    
    def is_predictable(self) -> bool:
        """Check if event is predictable."""
        return self.predictability >= 0.7
    
    def is_controllable(self) -> bool:
        """Check if event is controllable."""
        return self.control_level >= 0.8
    
    def get_minimal_impact(self) -> Dict[str, float]:
        """Get minimal impact scores."""
        return {
            "stress": self.stress_impact,
            "cognitive": self.cognitive_load,
            "emotional": self.emotional_neutrality,
            "memory": self.memory_salience,
        }


class EventTemplate(BaseModel):
    """Template for generating events."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Template metadata
    template_id: str = Field(..., description="Unique template identifier")
    event_type: EventType = Field(..., description="Event type")
    category: EventCategory = Field(..., description="Event category")
    
    # Template content
    title_template: str = Field(..., description="Title template with placeholders")
    description_template: str = Field(..., description="Description template with placeholders")
    context_template: str = Field(..., description="Context template with placeholders")
    
    # Parameter ranges
    intensity_range: tuple[EventIntensity, EventIntensity] = Field(..., description="Intensity range")
    stress_impact_range: tuple[float, float] = Field(..., description="Stress impact range")
    duration_range: tuple[int, int] = Field(..., description="Duration range in hours")
    
    # Personality impact ranges
    personality_impact_ranges: Dict[str, tuple[float, float]] = Field(default_factory=dict, description="Personality impact ranges")
    
    # Clinical impact ranges
    depression_risk_range: tuple[float, float] = Field(default=(0.0, 0.0), description="Depression risk range")
    anxiety_risk_range: tuple[float, float] = Field(default=(0.0, 0.0), description="Anxiety risk range")
    stress_risk_range: tuple[float, float] = Field(default=(0.0, 0.0), description="Stress risk range")
    
    # Generation parameters
    frequency_weight: float = Field(default=1.0, ge=0.0, description="Frequency weight for generation")
    persona_specific: bool = Field(default=False, description="Whether template is persona-specific")
    required_traits: List[str] = Field(default_factory=list, description="Required personality traits")
    
    def is_applicable_to_persona(self, persona_traits: Dict[str, float]) -> bool:
        """Check if template is applicable to a persona."""
        if not self.persona_specific:
            return True
        
        for trait in self.required_traits:
            if trait not in persona_traits:
                return False
            # Could add trait threshold checks here
        return True
    
    def get_parameter_ranges(self) -> Dict[str, Any]:
        """Get all parameter ranges for event generation."""
        return {
            "intensity_range": self.intensity_range,
            "stress_impact_range": self.stress_impact_range,
            "duration_range": self.duration_range,
            "personality_impact_ranges": self.personality_impact_ranges,
            "depression_risk_range": self.depression_risk_range,
            "anxiety_risk_range": self.anxiety_risk_range,
            "stress_risk_range": self.stress_risk_range,
        } 