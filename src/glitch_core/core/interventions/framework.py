"""
Comprehensive intervention framework for personality research.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from scipy import stats

from glitch_core.config.logging import get_logger


class InterventionType(Enum):
    """Types of interventions available."""
    TRAUMA = "trauma"
    THERAPY = "therapy"
    SOCIAL = "social"
    COGNITIVE = "cognitive"
    ENVIRONMENTAL = "environmental"
    MEDICAL = "medical"
    SUCCESS = "success"
    FAILURE = "failure"
    STRESS = "stress"
    RECOVERY = "recovery"


class InterventionIntensity(Enum):
    """Intensity levels for interventions."""
    MINIMAL = 0.1
    MILD = 0.3
    MODERATE = 0.5
    SEVERE = 0.7
    EXTREME = 0.9


@dataclass
class InterventionTemplate:
    """Template for creating interventions."""
    name: str
    intervention_type: InterventionType
    description: str
    default_intensity: float
    target_traits: List[str]
    expected_effects: Dict[str, float]
    duration_epochs: int
    recovery_time: int
    contraindications: List[str]
    parameters: Dict[str, Any]


@dataclass
class Intervention:
    """A single intervention instance."""
    id: str
    experiment_id: str
    intervention_type: InterventionType
    intensity: float
    description: str
    applied_at_epoch: int
    target_traits: List[str]
    parameters: Dict[str, Any]
    status: str  # "applied", "active", "completed", "failed"
    impact_measurements: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "intervention_type": self.intervention_type.value,
            "intensity": self.intensity,
            "description": self.description,
            "applied_at_epoch": self.applied_at_epoch,
            "target_traits": self.target_traits,
            "parameters": self.parameters,
            "status": self.status,
            "impact_measurements": self.impact_measurements,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intervention":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            experiment_id=data["experiment_id"],
            intervention_type=InterventionType(data["intervention_type"]),
            intensity=data["intensity"],
            description=data["description"],
            applied_at_epoch=data["applied_at_epoch"],
            target_traits=data["target_traits"],
            parameters=data["parameters"],
            status=data["status"],
            impact_measurements=data["impact_measurements"],
            created_at=datetime.fromisoformat(data["created_at"])
        )


@dataclass
class InterventionImpact:
    """Measurement of intervention impact."""
    intervention_id: str
    experiment_id: str
    pre_intervention_metrics: Dict[str, float]
    post_intervention_metrics: Dict[str, float]
    impact_scores: Dict[str, float]
    recovery_time: Optional[int]
    permanent_changes: Dict[str, float]
    effectiveness_rating: float
    measured_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "intervention_id": self.intervention_id,
            "experiment_id": self.experiment_id,
            "pre_intervention_metrics": self.pre_intervention_metrics,
            "post_intervention_metrics": self.post_intervention_metrics,
            "impact_scores": self.impact_scores,
            "recovery_time": self.recovery_time,
            "permanent_changes": self.permanent_changes,
            "effectiveness_rating": self.effectiveness_rating,
            "measured_at": self.measured_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterventionImpact":
        """Create from dictionary."""
        return cls(
            intervention_id=data["intervention_id"],
            experiment_id=data["experiment_id"],
            pre_intervention_metrics=data["pre_intervention_metrics"],
            post_intervention_metrics=data["post_intervention_metrics"],
            impact_scores=data["impact_scores"],
            recovery_time=data.get("recovery_time"),
            permanent_changes=data["permanent_changes"],
            effectiveness_rating=data["effectiveness_rating"],
            measured_at=datetime.fromisoformat(data["measured_at"])
        )


class InterventionFramework:
    """
    Comprehensive framework for managing interventions in personality research.
    """
    
    def __init__(self):
        self.logger = get_logger("intervention_framework")
        
        # Intervention templates
        self.templates = self._create_intervention_templates()
        
        # Active interventions tracking
        self.active_interventions: Dict[str, Intervention] = {}
        
        # Impact measurement history
        self.impact_history: Dict[str, List[InterventionImpact]] = {}
        
    def create_intervention(
        self,
        experiment_id: str,
        intervention_type: Union[str, InterventionType],
        intensity: float,
        description: str,
        applied_at_epoch: int,
        target_traits: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Intervention:
        """
        Create a new intervention.
        
        Args:
            experiment_id: ID of the experiment
            intervention_type: Type of intervention
            intensity: Intensity level (0.0-1.0)
            description: Description of the intervention
            applied_at_epoch: Epoch when intervention was applied
            target_traits: Traits targeted by the intervention
            parameters: Additional parameters
            
        Returns:
            Created intervention
        """
        try:
            # Convert string to enum if needed
            if isinstance(intervention_type, str):
                intervention_type = InterventionType(intervention_type)
            
            # Generate unique ID
            intervention_id = str(uuid.uuid4())
            
            # Get template for this intervention type
            template = self._get_template_for_type(intervention_type)
            
            # Set default target traits if not provided
            if target_traits is None:
                target_traits = template.target_traits
            
            # Set default parameters if not provided
            if parameters is None:
                parameters = template.parameters.copy()
            
            # Create intervention
            intervention = Intervention(
                id=intervention_id,
                experiment_id=experiment_id,
                intervention_type=intervention_type,
                intensity=intensity,
                description=description,
                applied_at_epoch=applied_at_epoch,
                target_traits=target_traits,
                parameters=parameters,
                status="applied",
                impact_measurements={},
                created_at=datetime.utcnow()
            )
            
            # Store active intervention
            self.active_interventions[intervention_id] = intervention
            
            self.logger.info(
                "intervention_created",
                intervention_id=intervention_id,
                experiment_id=experiment_id,
                intervention_type=intervention_type.value,
                intensity=intensity
            )
            
            return intervention
            
        except Exception as e:
            self.logger.error("intervention_creation_failed", error=str(e))
            raise
    
    def measure_intervention_impact(
        self,
        intervention_id: str,
        pre_intervention_data: Dict[str, Any],
        post_intervention_data: Dict[str, Any],
        current_epoch: int
    ) -> InterventionImpact:
        """
        Measure the impact of an intervention.
        
        Args:
            intervention_id: ID of the intervention
            pre_intervention_data: Data before intervention
            post_intervention_data: Data after intervention
            current_epoch: Current epoch for timing calculations
            
        Returns:
            Intervention impact measurement
        """
        try:
            intervention = self.active_interventions.get(intervention_id)
            if not intervention:
                raise ValueError(f"Intervention {intervention_id} not found")
            
            # Extract metrics
            pre_metrics = self._extract_metrics(pre_intervention_data)
            post_metrics = self._extract_metrics(post_intervention_data)
            
            # Calculate impact scores
            impact_scores = self._calculate_impact_scores(pre_metrics, post_metrics)
            
            # Calculate recovery time
            recovery_time = self._calculate_recovery_time(
                intervention, post_intervention_data, current_epoch
            )
            
            # Calculate permanent changes
            permanent_changes = self._calculate_permanent_changes(pre_metrics, post_metrics)
            
            # Calculate effectiveness rating
            effectiveness_rating = self._calculate_effectiveness_rating(impact_scores)
            
            # Create impact measurement
            impact = InterventionImpact(
                intervention_id=intervention_id,
                experiment_id=intervention.experiment_id,
                pre_intervention_metrics=pre_metrics,
                post_intervention_metrics=post_metrics,
                impact_scores=impact_scores,
                recovery_time=recovery_time,
                permanent_changes=permanent_changes,
                effectiveness_rating=effectiveness_rating,
                measured_at=datetime.utcnow()
            )
            
            # Store in history
            if intervention_id not in self.impact_history:
                self.impact_history[intervention_id] = []
            self.impact_history[intervention_id].append(impact)
            
            # Update intervention status
            intervention.status = "completed"
            intervention.impact_measurements = impact_scores
            
            self.logger.info(
                "intervention_impact_measured",
                intervention_id=intervention_id,
                effectiveness_rating=effectiveness_rating
            )
            
            return impact
            
        except Exception as e:
            self.logger.error("impact_measurement_failed", intervention_id=intervention_id, error=str(e))
            raise
    
    def get_intervention_templates(self) -> Dict[str, InterventionTemplate]:
        """Get all available intervention templates."""
        return {template.name: template for template in self.templates}
    
    def get_intervention_history(self, experiment_id: str) -> List[Intervention]:
        """Get intervention history for an experiment."""
        return [
            intervention for intervention in self.active_interventions.values()
            if intervention.experiment_id == experiment_id
        ]
    
    def get_impact_history(self, intervention_id: str) -> List[InterventionImpact]:
        """Get impact history for an intervention."""
        return self.impact_history.get(intervention_id, [])
    
    def compare_interventions(
        self, 
        intervention_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple interventions.
        
        Args:
            intervention_ids: List of intervention IDs to compare
            
        Returns:
            Comparison results
        """
        try:
            if len(intervention_ids) < 2:
                return {"error": "Need at least 2 interventions for comparison"}
            
            comparison_results = {
                "intervention_count": len(intervention_ids),
                "comparisons": {},
                "effectiveness_rankings": [],
                "type_analysis": {}
            }
            
            # Get interventions
            interventions = []
            for intervention_id in intervention_ids:
                intervention = self.active_interventions.get(intervention_id)
                if intervention:
                    interventions.append(intervention)
            
            if len(interventions) < 2:
                return {"error": "Not enough valid interventions for comparison"}
            
            # Compare effectiveness
            effectiveness_data = []
            for intervention in interventions:
                impact_history = self.get_impact_history(intervention.id)
                if impact_history:
                    latest_impact = impact_history[-1]
                    effectiveness_data.append({
                        "intervention_id": intervention.id,
                        "intervention_type": intervention.intervention_type.value,
                        "intensity": intervention.intensity,
                        "effectiveness": latest_impact.effectiveness_rating
                    })
            
            # Sort by effectiveness
            effectiveness_data.sort(key=lambda x: x["effectiveness"], reverse=True)
            comparison_results["effectiveness_rankings"] = effectiveness_data
            
            # Analyze by intervention type
            type_analysis = {}
            for intervention in interventions:
                intervention_type = intervention.intervention_type.value
                if intervention_type not in type_analysis:
                    type_analysis[intervention_type] = []
                
                impact_history = self.get_impact_history(intervention.id)
                if impact_history:
                    type_analysis[intervention_type].append({
                        "intervention_id": intervention.id,
                        "intensity": intervention.intensity,
                        "effectiveness": impact_history[-1].effectiveness_rating
                    })
            
            comparison_results["type_analysis"] = type_analysis
            
            return comparison_results
            
        except Exception as e:
            self.logger.error("intervention_comparison_failed", error=str(e))
            return {"error": f"Comparison failed: {str(e)}"}
    
    def export_intervention_data(
        self, 
        experiment_id: str,
        format: str = "json"
    ) -> Union[str, bytes]:
        """
        Export intervention data for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            format: Export format ("json", "csv", "excel")
            
        Returns:
            Exported data
        """
        try:
            # Get intervention history
            interventions = self.get_intervention_history(experiment_id)
            
            # Get impact data
            impact_data = {}
            for intervention in interventions:
                impact_history = self.get_impact_history(intervention.id)
                if impact_history:
                    impact_data[intervention.id] = [impact.to_dict() for impact in impact_history]
            
            # Prepare export data
            export_data = {
                "experiment_id": experiment_id,
                "interventions": [intervention.to_dict() for intervention in interventions],
                "impact_data": impact_data,
                "exported_at": datetime.utcnow().isoformat()
            }
            
            if format.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            else:
                return f"Export format {format} not implemented"
                
        except Exception as e:
            self.logger.error("intervention_export_failed", experiment_id=experiment_id, error=str(e))
            return f"Export failed: {str(e)}"
    
    def _create_intervention_templates(self) -> List[InterventionTemplate]:
        """Create intervention templates."""
        templates = [
            # Trauma interventions
            InterventionTemplate(
                name="Social Rejection",
                intervention_type=InterventionType.TRAUMA,
                description="Simulate severe social rejection event",
                default_intensity=0.8,
                target_traits=["neuroticism", "extraversion", "agreeableness"],
                expected_effects={"stability": -0.3, "anxiety": 0.6, "depression": 0.4},
                duration_epochs=10,
                recovery_time=20,
                contraindications=["high_resilience", "low_social_sensitivity"],
                parameters={"rejection_type": "peer", "public": True, "unexpected": True}
            ),
            
            InterventionTemplate(
                name="Work Failure",
                intervention_type=InterventionType.TRAUMA,
                description="Simulate significant work failure",
                default_intensity=0.7,
                target_traits=["conscientiousness", "neuroticism"],
                expected_effects={"stability": -0.2, "anxiety": 0.5, "confidence": -0.4},
                duration_epochs=8,
                recovery_time=15,
                contraindications=["high_self_efficacy", "low_work_importance"],
                parameters={"failure_type": "project", "public": False, "career_impact": True}
            ),
            
            # Therapy interventions
            InterventionTemplate(
                name="Cognitive Behavioral Therapy",
                intervention_type=InterventionType.THERAPY,
                description="Simulate CBT session effects",
                default_intensity=0.6,
                target_traits=["neuroticism", "anxiety", "depression"],
                expected_effects={"stability": 0.3, "anxiety": -0.4, "depression": -0.3},
                duration_epochs=15,
                recovery_time=5,
                contraindications=["low_therapy_acceptance"],
                parameters={"session_type": "individual", "technique": "thought_restructuring"}
            ),
            
            InterventionTemplate(
                name="Mindfulness Meditation",
                intervention_type=InterventionType.THERAPY,
                description="Simulate mindfulness practice effects",
                default_intensity=0.4,
                target_traits=["neuroticism", "anxiety", "stress"],
                expected_effects={"stability": 0.2, "anxiety": -0.3, "stress": -0.4},
                duration_epochs=20,
                recovery_time=3,
                contraindications=["low_mindfulness_acceptance"],
                parameters={"practice_type": "breathing", "duration_minutes": 20}
            ),
            
            # Social interventions
            InterventionTemplate(
                name="Social Support",
                intervention_type=InterventionType.SOCIAL,
                description="Simulate receiving social support",
                default_intensity=0.5,
                target_traits=["extraversion", "agreeableness", "depression"],
                expected_effects={"stability": 0.2, "depression": -0.3, "belonging": 0.4},
                duration_epochs=12,
                recovery_time=2,
                contraindications=["social_avoidance"],
                parameters={"support_type": "emotional", "source": "friends", "public": False}
            ),
            
            # Success interventions
            InterventionTemplate(
                name="Major Achievement",
                intervention_type=InterventionType.SUCCESS,
                description="Simulate major personal achievement",
                default_intensity=0.7,
                target_traits=["extraversion", "conscientiousness", "confidence"],
                expected_effects={"stability": 0.3, "confidence": 0.5, "motivation": 0.4},
                duration_epochs=10,
                recovery_time=1,
                contraindications=["imposter_syndrome"],
                parameters={"achievement_type": "career", "recognition": True, "public": True}
            ),
            
            # Environmental interventions
            InterventionTemplate(
                name="Environmental Change",
                intervention_type=InterventionType.ENVIRONMENTAL,
                description="Simulate significant environmental change",
                default_intensity=0.6,
                target_traits=["openness", "neuroticism", "adaptability"],
                expected_effects={"stability": -0.1, "anxiety": 0.3, "adaptability": 0.2},
                duration_epochs=25,
                recovery_time=10,
                contraindications=["low_adaptability"],
                parameters={"change_type": "relocation", "voluntary": False, "support_available": True}
            )
        ]
        
        return templates
    
    def _get_template_for_type(self, intervention_type: InterventionType) -> InterventionTemplate:
        """Get template for intervention type."""
        for template in self.templates:
            if template.intervention_type == intervention_type:
                return template
        
        # Return default template if not found
        return self.templates[0]
    
    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics from simulation data."""
        metrics = {}
        
        # Extract emotional state metrics
        if "emotional_state" in data:
            emotional_state = data["emotional_state"]
            metrics.update(emotional_state)
        
        # Extract stability metrics
        if "stability_metrics" in data:
            stability_metrics = data["stability_metrics"]
            metrics.update(stability_metrics)
        
        # Extract personality trait metrics
        if "personality_traits" in data:
            personality_traits = data["personality_traits"]
            metrics.update(personality_traits)
        
        return metrics
    
    def _calculate_impact_scores(
        self, 
        pre_metrics: Dict[str, float], 
        post_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate impact scores for different metrics."""
        impact_scores = {}
        
        # Calculate changes for common metrics
        common_metrics = set(pre_metrics.keys()) & set(post_metrics.keys())
        
        for metric in common_metrics:
            pre_value = pre_metrics[metric]
            post_value = post_metrics[metric]
            
            # Calculate relative change
            if pre_value != 0:
                relative_change = (post_value - pre_value) / abs(pre_value)
            else:
                relative_change = post_value - pre_value
            
            impact_scores[metric] = relative_change
        
        # Calculate overall impact score
        if impact_scores:
            # Weight positive and negative changes
            positive_changes = [v for v in impact_scores.values() if v > 0]
            negative_changes = [abs(v) for v in impact_scores.values() if v < 0]
            
            positive_impact = np.mean(positive_changes) if positive_changes else 0
            negative_impact = np.mean(negative_changes) if negative_changes else 0
            
            overall_impact = positive_impact - negative_impact
            impact_scores["overall_impact"] = overall_impact
        
        return impact_scores
    
    def _calculate_recovery_time(
        self, 
        intervention: Intervention, 
        post_data: Dict[str, Any], 
        current_epoch: int
    ) -> Optional[int]:
        """Calculate recovery time for intervention."""
        # This is a simplified calculation
        # In practice, you'd track when metrics return to baseline
        
        template = self._get_template_for_type(intervention.intervention_type)
        expected_recovery = template.recovery_time
        
        # Adjust based on intensity
        intensity_factor = intervention.intensity / template.default_intensity
        adjusted_recovery = int(expected_recovery * intensity_factor)
        
        return adjusted_recovery
    
    def _calculate_permanent_changes(
        self, 
        pre_metrics: Dict[str, float], 
        post_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate permanent changes from intervention."""
        permanent_changes = {}
        
        # For now, assume small changes are permanent
        # In practice, you'd track long-term stability
        
        common_metrics = set(pre_metrics.keys()) & set(post_metrics.keys())
        
        for metric in common_metrics:
            change = post_metrics[metric] - pre_metrics[metric]
            
            # Assume small changes (< 10%) are permanent
            if abs(change) < 0.1:
                permanent_changes[metric] = change
        
        return permanent_changes
    
    def _calculate_effectiveness_rating(self, impact_scores: Dict[str, float]) -> float:
        """Calculate overall effectiveness rating."""
        if not impact_scores:
            return 0.0
        
        # Get overall impact
        overall_impact = impact_scores.get("overall_impact", 0.0)
        
        # Normalize to 0-1 scale
        effectiveness = (overall_impact + 1.0) / 2.0
        effectiveness = max(0.0, min(1.0, effectiveness))
        
        return effectiveness 