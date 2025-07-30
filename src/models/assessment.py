"""
Assessment result models for psychiatric scales.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class SeverityLevel(str, Enum):
    """Clinical severity levels."""
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


# Threshold constants for severity calculation
# This is assumption, formore realistic and intelligent LLMs ned to 
# think about more relevant to 'borderline' disorders thresholds
# but for now lets stick with this
PHQ9_MINIMAL_THRESHOLD = 5
PHQ9_MILD_THRESHOLD = 10
PHQ9_MODERATE_THRESHOLD = 15
PHQ9_SEVERE_THRESHOLD = 20

GAD7_MINIMAL_THRESHOLD = 5
GAD7_MILD_THRESHOLD = 10
GAD7_MODERATE_THRESHOLD = 15
GAD7_SEVERE_THRESHOLD = 20

PSS10_MINIMAL_THRESHOLD = 13
PSS10_MILD_THRESHOLD = 16
PSS10_MODERATE_THRESHOLD = 19
PSS10_SEVERE_THRESHOLD = 22


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
        if total_score < PHQ9_MINIMAL_THRESHOLD:
            return SeverityLevel.MINIMAL
        elif total_score < PHQ9_MILD_THRESHOLD:
            return SeverityLevel.MILD
        elif total_score < PHQ9_MODERATE_THRESHOLD:
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
        if total_score < GAD7_MINIMAL_THRESHOLD:
            return SeverityLevel.MINIMAL
        elif total_score < GAD7_MILD_THRESHOLD:
            return SeverityLevel.MILD
        elif total_score < GAD7_MODERATE_THRESHOLD:
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
        if total_score < PSS10_MINIMAL_THRESHOLD:
            return SeverityLevel.MINIMAL
        elif total_score < PSS10_MILD_THRESHOLD:
            return SeverityLevel.MILD
        elif total_score < PSS10_MODERATE_THRESHOLD:
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
        """Get all assessment results in this session."""
        results = []
        if self.phq9_result:
            results.append(self.phq9_result)
        if self.gad7_result:
            results.append(self.gad7_result)
        if self.pss10_result:
            results.append(self.pss10_result)
        return results
    
    def get_composite_score(self) -> Dict[str, float]:
        """Get composite scores across all assessments."""
        scores = {}
        if self.phq9_result:
            scores["depression"] = self.phq9_result.total_score
        if self.gad7_result:
            scores["anxiety"] = self.gad7_result.total_score
        if self.pss10_result:
            scores["stress"] = self.pss10_result.total_score
        return scores
    
    def get_overall_severity(self) -> SeverityLevel:
        """Get overall clinical severity based on all assessments."""
        max_severity = SeverityLevel.MINIMAL
        
        for result in self.get_all_results():
            if result.severity_level == SeverityLevel.SEVERE:
                return SeverityLevel.SEVERE
            elif result.severity_level == SeverityLevel.MODERATE and max_severity in [SeverityLevel.MINIMAL, SeverityLevel.MILD]:
                max_severity = SeverityLevel.MODERATE
            elif result.severity_level == SeverityLevel.MILD and max_severity == SeverityLevel.MINIMAL:
                max_severity = SeverityLevel.MILD
        
        return max_severity
    
    def is_complete(self) -> bool:
        """Check if session is complete."""
        return self.completed_at is not None
    
    def mark_completed(self) -> None:
        """Mark session as completed."""
        self.completed_at = datetime.utcnow()
        if self.started_at and self.completed_at:
            self.session_duration = (self.completed_at - self.started_at).total_seconds() 