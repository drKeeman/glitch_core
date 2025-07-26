"""
Pydantic models for API v1 endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field


class PersonaConfigRequest(BaseModel):
    """Request model for creating/updating persona configurations."""
    
    name: str = Field(..., description="Human-readable name for the persona")
    persona_type: str = Field(..., description="Type of persona (resilient_optimist, anxious_overthinker, etc.)")
    traits: Dict[str, float] = Field(default_factory=dict, description="Personality traits (Big 5)")
    cognitive_biases: Dict[str, float] = Field(default_factory=dict, description="Cognitive bias strengths")
    emotional_baselines: Dict[str, float] = Field(default_factory=dict, description="Default emotional states")
    memory_patterns: Dict[str, float] = Field(default_factory=dict, description="Memory encoding biases")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Test Optimist",
                "persona_type": "resilient_optimist",
                "traits": {"openness": 0.8, "conscientiousness": 0.7},
                "cognitive_biases": {"confirmation_bias": 0.3},
                "emotional_baselines": {"joy": 0.6, "anxiety": 0.2},
                "memory_patterns": {"positive_recall": 0.8}
            }
        }


class PersonaConfigResponse(BaseModel):
    """Response model for persona configurations."""
    
    id: UUID
    name: str
    persona_type: str
    traits: Dict[str, float]
    cognitive_biases: Dict[str, float]
    emotional_baselines: Dict[str, float]
    memory_patterns: Dict[str, float]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ExperimentRequest(BaseModel):
    """Request model for creating experiments."""
    
    persona_id: UUID = Field(..., description="ID of the persona to simulate")
    drift_profile: str = Field(..., description="Type of drift profile to apply")
    epochs: int = Field(100, ge=1, le=1000, description="Number of simulation epochs")
    events_per_epoch: int = Field(10, ge=1, le=50, description="Events per epoch")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    class Config:
        json_schema_extra = {
            "example": {
                "persona_id": "123e4567-e89b-12d3-a456-426614174000",
                "drift_profile": "gradual_deterioration",
                "epochs": 100,
                "events_per_epoch": 10,
                "seed": 42
            }
        }


class ExperimentResponse(BaseModel):
    """Response model for experiments."""
    
    id: UUID
    persona_id: UUID
    drift_profile: str
    epochs: int
    events_per_epoch: int
    seed: Optional[int]
    status: str = Field(..., description="Status: running, completed, failed, stopped")
    current_epoch: int = Field(0, description="Current epoch if running")
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class InterventionRequest(BaseModel):
    """Request model for injecting interventions."""
    
    experiment_id: UUID = Field(..., description="ID of the experiment to intervene in")
    event_type: str = Field(..., description="Type of intervention event")
    intensity: float = Field(1.0, ge=0.0, le=10.0, description="Intensity of the intervention")
    description: str = Field(..., description="Description of the intervention")
    
    class Config:
        json_schema_extra = {
            "example": {
                "experiment_id": "123e4567-e89b-12d3-a456-426614174000",
                "event_type": "trauma_injection",
                "intensity": 8.5,
                "description": "Severe social rejection event"
            }
        }


class InterventionResponse(BaseModel):
    """Response model for interventions."""
    
    id: UUID
    experiment_id: UUID
    event_type: str
    intensity: float
    description: str
    applied_at: datetime
    impact_score: Optional[float] = Field(None, description="Measured impact on personality")
    
    class Config:
        from_attributes = True


class AnalysisResult(BaseModel):
    """Response model for analysis results."""
    
    experiment_id: UUID
    emergence_points: List[Dict[str, Any]] = Field(..., description="Detected pattern emergence points")
    stability_boundaries: List[Dict[str, Any]] = Field(..., description="Stability boundary analysis")
    intervention_leverage: List[Dict[str, Any]] = Field(..., description="Intervention impact measurements")
    attention_evolution: List[Dict[str, Any]] = Field(..., description="Attention pattern evolution")
    drift_patterns: List[Dict[str, Any]] = Field(..., description="Overall drift pattern analysis")
    created_at: datetime
    
    class Config:
        from_attributes = True


class WebSocketEvent(BaseModel):
    """WebSocket event model."""
    
    event_type: str = Field(..., description="Type of event")
    experiment_id: UUID
    timestamp: datetime
    data: Dict[str, Any] = Field(..., description="Event-specific data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "epoch_completed",
                "experiment_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-01T12:00:00Z",
                "data": {
                    "epoch": 42,
                    "emotional_state": {"joy": 0.6, "anxiety": 0.3},
                    "stability_score": 0.85
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow) 