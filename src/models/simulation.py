"""
Simulation state and configuration models.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class SimulationStatus(str, Enum):
    """Simulation status states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class ExperimentalCondition(str, Enum):
    """Experimental conditions for simulation."""
    CONTROL = "control"
    STRESS = "stress"
    NEUTRAL = "neutral"
    MINIMAL = "minimal"


class SimulationConfig(BaseModel):
    """Configuration for simulation parameters."""
    
    model_config = ConfigDict(extra="allow")
    
    # Simulation parameters
    duration_days: int = Field(default=30, ge=1, le=365, description="Simulation duration in days")
    time_compression_factor: int = Field(default=24, ge=1, le=168, description="Time compression factor (hours per day)")
    assessment_interval_days: int = Field(default=7, ge=1, le=30, description="Days between assessments")
    
    # Experimental design
    experimental_condition: ExperimentalCondition = Field(default=ExperimentalCondition.CONTROL, description="Experimental condition")
    persona_count: int = Field(default=3, ge=1, le=10, description="Number of personas per condition")
    
    # Event parameters
    stress_event_frequency: float = Field(default=0.1, ge=0.0, le=1.0, description="Daily probability of stress events")
    neutral_event_frequency: float = Field(default=0.3, ge=0.0, le=1.0, description="Daily probability of neutral events")
    event_intensity_range: tuple[float, float] = Field(default=(0.5, 1.0), description="Event intensity range")
    
    # Mechanistic analysis
    capture_attention_patterns: bool = Field(default=True, description="Capture attention patterns during inference")
    capture_activation_changes: bool = Field(default=True, description="Capture activation changes during inference")
    attention_sampling_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Fraction of tokens to sample for attention")
    
    # Performance settings
    max_concurrent_personas: int = Field(default=3, ge=1, le=10, description="Maximum concurrent persona simulations")
    checkpoint_interval_hours: int = Field(default=6, ge=1, le=24, description="Checkpoint save interval")
    memory_cleanup_interval: int = Field(default=24, ge=1, le=168, description="Memory cleanup interval in hours")
    
    # Data collection
    save_raw_responses: bool = Field(default=True, description="Save raw LLM responses")
    save_mechanistic_data: bool = Field(default=True, description="Save mechanistic analysis data")
    save_memory_embeddings: bool = Field(default=True, description="Save memory embeddings")
    
    # Export settings
    export_format: str = Field(default="json", description="Data export format")
    export_compression: bool = Field(default=True, description="Compress exported data")
    
    def get_total_simulation_hours(self) -> int:
        """Calculate total simulation time in hours."""
        return self.duration_days * self.time_compression_factor
    
    def get_assessment_count(self) -> int:
        """Calculate total number of assessments per persona."""
        return self.duration_days // self.assessment_interval_days
    
    def is_valid_configuration(self) -> bool:
        """Validate configuration parameters."""
        if self.duration_days < self.assessment_interval_days:
            return False
        if self.stress_event_frequency + self.neutral_event_frequency > 1.0:
            return False
        if self.attention_sampling_rate <= 0.0 or self.attention_sampling_rate > 1.0:
            return False
        return True


class SimulationState(BaseModel):
    """Current state of the simulation."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Simulation metadata
    simulation_id: str = Field(..., description="Unique simulation identifier")
    config: SimulationConfig = Field(..., description="Simulation configuration")
    
    # Status and progress
    status: SimulationStatus = Field(default=SimulationStatus.IDLE, description="Current simulation status")
    current_day: int = Field(default=0, ge=0, description="Current simulation day")
    current_hour: int = Field(default=0, ge=0, le=23, description="Current simulation hour")
    
    # Progress tracking
    start_time: Optional[datetime] = Field(None, description="Simulation start timestamp")
    end_time: Optional[datetime] = Field(None, description="Simulation end timestamp")
    last_checkpoint: Optional[datetime] = Field(None, description="Last checkpoint timestamp")
    
    # Persona tracking
    active_personas: Set[str] = Field(default_factory=set, description="Currently active persona IDs")
    completed_personas: Set[str] = Field(default_factory=set, description="Completed persona IDs")
    failed_personas: Set[str] = Field(default_factory=set, description="Failed persona IDs")
    
    # Event tracking
    total_events_processed: int = Field(default=0, ge=0, description="Total events processed")
    stress_events_processed: int = Field(default=0, ge=0, description="Stress events processed")
    neutral_events_processed: int = Field(default=0, ge=0, description="Neutral events processed")
    
    # Assessment tracking
    total_assessments_completed: int = Field(default=0, ge=0, description="Total assessments completed")
    assessments_due_today: List[str] = Field(default_factory=list, description="Personas due for assessment today")
    
    # Performance metrics
    average_response_time: float = Field(default=0.0, ge=0.0, description="Average response time in seconds")
    memory_usage_mb: float = Field(default=0.0, ge=0.0, description="Current memory usage in MB")
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="Current CPU usage percentage")
    
    # Error tracking
    error_count: int = Field(default=0, ge=0, description="Total errors encountered")
    last_error: Optional[str] = Field(None, description="Last error message")
    
    # Data collection
    data_points_collected: int = Field(default=0, ge=0, description="Total data points collected")
    mechanistic_samples: int = Field(default=0, ge=0, description="Mechanistic analysis samples")
    
    def get_progress_percentage(self) -> float:
        """Calculate simulation progress percentage."""
        if self.config.duration_days == 0:
            return 0.0
        return min(100.0, (self.current_day / self.config.duration_days) * 100.0)
    
    def get_elapsed_time(self) -> Optional[float]:
        """Get elapsed simulation time in seconds."""
        if not self.start_time:
            return None
        end_time = self.end_time or datetime.utcnow()
        return (end_time - self.start_time).total_seconds()
    
    def get_estimated_completion_time(self) -> Optional[datetime]:
        """Estimate completion time based on current progress."""
        if self.get_progress_percentage() == 0:
            return None
        
        elapsed = self.get_elapsed_time()
        if not elapsed:
            return None
        
        progress = self.get_progress_percentage() / 100.0
        total_estimated_time = elapsed / progress
        remaining_time = total_estimated_time - elapsed
        
        return datetime.utcnow() + timedelta(seconds=remaining_time)
    
    def is_completed(self) -> bool:
        """Check if simulation is completed."""
        return self.status == SimulationStatus.COMPLETED
    
    def is_running(self) -> bool:
        """Check if simulation is currently running."""
        return self.status == SimulationStatus.RUNNING
    
    def can_start(self) -> bool:
        """Check if simulation can be started."""
        return self.status in [SimulationStatus.IDLE, SimulationStatus.PAUSED]
    
    def can_pause(self) -> bool:
        """Check if simulation can be paused."""
        return self.status == SimulationStatus.RUNNING
    
    def mark_started(self) -> None:
        """Mark simulation as started."""
        self.status = SimulationStatus.RUNNING
        self.start_time = datetime.utcnow()
        self.current_day = 0
        self.current_hour = 0
    
    def mark_completed(self) -> None:
        """Mark simulation as completed."""
        self.status = SimulationStatus.COMPLETED
        self.end_time = datetime.utcnow()
    
    def mark_paused(self) -> None:
        """Mark simulation as paused."""
        self.status = SimulationStatus.PAUSED
    
    def mark_error(self, error_message: str) -> None:
        """Mark simulation as error state."""
        self.status = SimulationStatus.ERROR
        self.last_error = error_message
        self.error_count += 1
    
    def advance_time(self, hours: int = 1) -> None:
        """Advance simulation time."""
        self.current_hour += hours
        
        # Advance to next day if needed
        while self.current_hour >= 24:
            self.current_hour -= 24
            self.current_day += 1
    
    def add_persona(self, persona_id: str) -> None:
        """Add a persona to active tracking."""
        self.active_personas.add(persona_id)
    
    def complete_persona(self, persona_id: str) -> None:
        """Mark a persona as completed."""
        self.active_personas.discard(persona_id)
        self.completed_personas.add(persona_id)
    
    def fail_persona(self, persona_id: str, error_message: str) -> None:
        """Mark a persona as failed."""
        self.active_personas.discard(persona_id)
        self.failed_personas.add(persona_id)
        self.mark_error(f"Persona {persona_id} failed: {error_message}")
    
    def update_performance_metrics(self, response_time: float, memory_mb: float, cpu_percent: float) -> None:
        """Update performance metrics."""
        self.average_response_time = response_time
        self.memory_usage_mb = memory_mb
        self.cpu_usage_percent = cpu_percent
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert simulation state to dictionary."""
        return {
            "simulation_id": self.simulation_id,
            "config": self.config.model_dump(),
            "status": self.status.value,
            "current_day": self.current_day,
            "current_hour": self.current_hour,
            "progress_percentage": self.get_progress_percentage(),
            "active_personas": list(self.active_personas),
            "completed_personas": list(self.completed_personas),
            "failed_personas": list(self.failed_personas),
            "total_events_processed": self.total_events_processed,
            "total_assessments_completed": self.total_assessments_completed,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        } 