"""
Assessment models for clinical scales and results.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, ConfigDict

from src.core.experiment_config import experiment_config


class SeverityLevel(str, Enum):
    """Clinical severity levels."""
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


def get_clinical_thresholds():
    """Get clinical thresholds from configuration."""
    config = experiment_config.get_config("clinical_thresholds")
    return config


def get_phq9_thresholds():
    """Get PHQ-9 thresholds from configuration."""
    config = get_clinical_thresholds()
    return config.get("phq9", {})


def get_gad7_thresholds():
    """Get GAD-7 thresholds from configuration."""
    config = get_clinical_thresholds()
    return config.get("gad7", {})


def get_pss10_thresholds():
    """Get PSS-10 thresholds from configuration."""
    config = get_clinical_thresholds()
    return config.get("pss10", {})


class AssessmentResult(BaseModel):
    """Base class for assessment results."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Assessment metadata
    assessment_id: str = Field(..., description="Unique assessment identifier")
    persona_id: str = Field(..., description="Persona being assessed")
    assessment_type: str = Field(..., description="Type of assessment")
    simulation_day: int = Field(..., ge=0, description="Simulation day of assessment")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Assessment timestamp")
    
    # Raw responses
    raw_responses: List[str] = Field(default_factory=list, description="Raw text responses")
    parsed_scores: List[int] = Field(default_factory=list, description="Parsed numeric scores")
    
    # Results
    total_score: float = Field(..., description="Total assessment score")
    severity_level: SeverityLevel = Field(..., description="Clinical severity level")
    
    # Analysis
    response_consistency: float = Field(default=0.0, ge=0.0, le=1.0, description="Response consistency score")
    response_time_avg: Optional[float] = Field(None, ge=0.0, description="Average response time in seconds")
    
    # Clinical interpretation
    clinical_interpretation: Optional[Dict[str, Any]] = Field(None, description="Clinical interpretation and recommendations")
    
    def get_score_change(self, baseline_score: float) -> float:
        """Calculate score change from baseline."""
        return self.total_score - baseline_score
    
    def is_clinically_significant(self, baseline_score: float, threshold: float = 5.0) -> bool:
        """Check if score change is clinically significant."""
        change = abs(self.get_score_change(baseline_score))
        return change >= threshold


class PHQ9Result(AssessmentResult):
    """PHQ-9 (Patient Health Questionnaire-9) assessment result."""
    
    assessment_type: str = Field(default="phq9", description="Assessment type")
    
    # PHQ-9 specific fields
    depression_severity: SeverityLevel = Field(..., description="Depression severity level")
    suicidal_ideation_score: int = Field(..., ge=0, le=3, description="Suicidal ideation item score")
    
    @classmethod
    def calculate_severity(cls, total_score: float) -> SeverityLevel:
        """Calculate severity level based on PHQ-9 score."""
        thresholds = get_phq9_thresholds()
        minimal_threshold = thresholds.get("minimal_threshold", 5)
        mild_threshold = thresholds.get("mild_threshold", 10)
        moderate_threshold = thresholds.get("moderate_threshold", 15)
        
        if total_score < minimal_threshold:
            return SeverityLevel.MINIMAL
        elif total_score < mild_threshold:
            return SeverityLevel.MILD
        elif total_score < moderate_threshold:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.SEVERE
    
    def has_suicidal_ideation(self) -> bool:
        """Check if suicidal ideation is present."""
        return self.suicidal_ideation_score >= 2


class GAD7Result(AssessmentResult):
    """GAD-7 (Generalized Anxiety Disorder-7) assessment result."""
    
    assessment_type: str = Field(default="gad7", description="Assessment type")
    
    # GAD-7 specific fields
    anxiety_severity: SeverityLevel = Field(..., description="Anxiety severity level")
    worry_duration: str = Field(default="unknown", description="Duration of worry symptoms")
    
    @classmethod
    def calculate_severity(cls, total_score: float) -> SeverityLevel:
        """Calculate severity level based on GAD-7 score."""
        thresholds = get_gad7_thresholds()
        minimal_threshold = thresholds.get("minimal_threshold", 5)
        mild_threshold = thresholds.get("mild_threshold", 10)
        moderate_threshold = thresholds.get("moderate_threshold", 15)
        
        if total_score < minimal_threshold:
            return SeverityLevel.MINIMAL
        elif total_score < mild_threshold:
            return SeverityLevel.MILD
        elif total_score < moderate_threshold:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.SEVERE


class PSS10Result(AssessmentResult):
    """PSS-10 (Perceived Stress Scale-10) assessment result."""
    
    assessment_type: str = Field(default="pss10", description="Assessment type")
    
    # PSS-10 specific fields
    stress_severity: SeverityLevel = Field(..., description="Stress severity level")
    coping_effectiveness: float = Field(default=0.0, ge=0.0, le=1.0, description="Coping effectiveness score")
    
    @classmethod
    def calculate_severity(cls, total_score: float) -> SeverityLevel:
        """Calculate severity level based on PSS-10 score."""
        thresholds = get_pss10_thresholds()
        minimal_threshold = thresholds.get("minimal_threshold", 13)
        mild_threshold = thresholds.get("mild_threshold", 16)
        moderate_threshold = thresholds.get("moderate_threshold", 19)
        
        if total_score < minimal_threshold:
            return SeverityLevel.MINIMAL
        elif total_score < mild_threshold:
            return SeverityLevel.MILD
        elif total_score < moderate_threshold:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.SEVERE


class AssessmentSession(BaseModel):
    """Complete assessment session with multiple scales."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Session metadata
    session_id: str = Field(..., description="Unique session identifier")
    persona_id: str = Field(..., description="Persona being assessed")
    simulation_day: int = Field(..., ge=0, description="Simulation day")
    
    # Assessment results
    phq9_result: Optional[PHQ9Result] = Field(None, description="PHQ-9 assessment result")
    gad7_result: Optional[GAD7Result] = Field(None, description="GAD-7 assessment result")
    pss10_result: Optional[PSS10Result] = Field(None, description="PSS-10 assessment result")
    
    # Session metadata
    session_duration: Optional[float] = Field(None, ge=0.0, description="Total session duration in seconds")
    completion_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Assessment completion rate")
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Session start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Session completion timestamp")
    
    def get_all_results(self) -> List[AssessmentResult]:
        """Get all assessment results from session."""
        results = []
        if self.phq9_result:
            results.append(self.phq9_result)
        if self.gad7_result:
            results.append(self.gad7_result)
        if self.pss10_result:
            results.append(self.pss10_result)
        return results
    
    def get_composite_score(self) -> Dict[str, float]:
        """Get composite scores from all assessments."""
        composite = {}
        if self.phq9_result:
            composite["depression"] = self.phq9_result.total_score
        if self.gad7_result:
            composite["anxiety"] = self.gad7_result.total_score
        if self.pss10_result:
            composite["stress"] = self.pss10_result.total_score
        return composite
    
    def get_overall_severity(self) -> SeverityLevel:
        """Get overall severity level from all assessments."""
        results = self.get_all_results()
        if not results:
            return SeverityLevel.MINIMAL
        
        # Return the highest severity level (worst case)
        severity_scores = {
            SeverityLevel.MINIMAL: 0,
            SeverityLevel.MILD: 1,
            SeverityLevel.MODERATE: 2,
            SeverityLevel.SEVERE: 3
        }
        
        max_severity_score = max(severity_scores[result.severity_level] for result in results)
        
        if max_severity_score == 0:
            return SeverityLevel.MINIMAL
        elif max_severity_score == 1:
            return SeverityLevel.MILD
        elif max_severity_score == 2:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.SEVERE
    
    def is_complete(self) -> bool:
        """Check if session is complete."""
        return self.completed_at is not None
    
    def mark_completed(self) -> None:
        """Mark session as completed."""
        self.completed_at = datetime.utcnow()
        # Calculate session duration
        if self.started_at and self.completed_at:
            self.session_duration = (self.completed_at - self.started_at).total_seconds() 