"""
Temporal analyzer for detecting interpretability patterns in personality evolution.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass

from glitch_core.config.logging import get_logger


@dataclass
class AnalysisResult:
    """Result of temporal analysis."""
    
    emergence_points: List[Dict[str, Any]]
    stability_boundaries: List[Dict[str, Any]]
    intervention_leverage: List[Dict[str, Any]]
    attention_evolution: List[Dict[str, Any]]
    drift_patterns: List[Dict[str, Any]]
    created_at: datetime


class TemporalAnalyzer:
    """
    Core interpretability research algorithms for temporal patterns.
    """
    
    def __init__(self):
        self.logger = get_logger("temporal_analyzer")
        self.pattern_detector = PatternDetector()
        self.stability_analyzer = StabilityAnalyzer()
    
    def analyze_drift_patterns(self, simulation_data: Dict[str, Any]) -> AnalysisResult:
        """
        Analyze drift patterns in simulation data.
        
        Args:
            simulation_data: Complete simulation result with epochs and metrics
            
        Returns:
            AnalysisResult with all interpretability insights
        """
        self.logger.info("analyzing_drift_patterns", epochs=len(simulation_data.get("epochs", [])))
        
        epochs = simulation_data.get("epochs", [])
        if not epochs:
            self.logger.warning("no_epochs_to_analyze")
            return self._empty_analysis_result()
        
        # Extract time series data
        emotional_states = [epoch.get("emotional_state", {}) for epoch in epochs]
        stability_scores = [epoch.get("stability_score", 0.0) for epoch in epochs]
        attention_metrics = [epoch.get("attention_metrics", {}) for epoch in epochs]
        interventions = simulation_data.get("interventions", [])
        
        # Run analysis components
        emergence_points = self.pattern_detector.detect_emergence_points(
            emotional_states, stability_scores
        )
        
        stability_boundaries = self.stability_analyzer.analyze_stability_boundaries(
            stability_scores, emotional_states
        )
        
        intervention_leverage = self._analyze_intervention_leverage(
            interventions, emotional_states, stability_scores
        )
        
        attention_evolution = self._analyze_attention_evolution(attention_metrics)
        
        drift_patterns = self._analyze_overall_drift_patterns(
            emotional_states, stability_scores, emergence_points
        )
        
        return AnalysisResult(
            emergence_points=emergence_points,
            stability_boundaries=stability_boundaries,
            intervention_leverage=intervention_leverage,
            attention_evolution=attention_evolution,
            drift_patterns=drift_patterns,
            created_at=datetime.utcnow()
        )
    
    def _analyze_intervention_leverage(
        self, 
        interventions: List[Dict[str, Any]], 
        emotional_states: List[Dict[str, float]],
        stability_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Analyze the impact of interventions on personality evolution."""
        leverage_points = []
        
        for intervention in interventions:
            intervention_epoch = intervention.get("epoch", 0)
            if intervention_epoch >= len(emotional_states):
                continue
                
            # Calculate pre-intervention baseline
            pre_baseline = self._calculate_baseline(
                emotional_states[:intervention_epoch], 
                stability_scores[:intervention_epoch]
            )
            
            # Calculate post-intervention changes
            post_changes = self._calculate_post_intervention_changes(
                emotional_states[intervention_epoch:],
                stability_scores[intervention_epoch:],
                pre_baseline
            )
            
            leverage_points.append({
                "intervention_id": intervention.get("id"),
                "epoch": intervention_epoch,
                "event_type": intervention.get("event_type"),
                "intensity": intervention.get("intensity"),
                "pre_baseline": pre_baseline,
                "post_changes": post_changes,
                "impact_score": self._calculate_impact_score(post_changes),
                "recovery_time": self._calculate_recovery_time(
                    emotional_states[intervention_epoch:],
                    pre_baseline
                )
            })
        
        return leverage_points
    
    def _analyze_attention_evolution(
        self, 
        attention_metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze how attention patterns evolve over time."""
        if not attention_metrics or not attention_metrics[0]:
            return []
        
        evolution_points = []
        
        # Track attention focus shifts
        for i, metrics in enumerate(attention_metrics):
            if i == 0:
                continue
                
            prev_metrics = attention_metrics[i-1]
            
            # Calculate attention drift
            focus_shifts = {}
            for key in metrics:
                if key in prev_metrics:
                    shift = metrics[key] - prev_metrics[key]
                    if abs(shift) > 0.1:  # Significant shift threshold
                        focus_shifts[key] = shift
            
            if focus_shifts:
                evolution_points.append({
                    "epoch": i,
                    "focus_shifts": focus_shifts,
                    "total_shift_magnitude": sum(abs(v) for v in focus_shifts.values()),
                    "primary_shift": max(focus_shifts.items(), key=lambda x: abs(x[1])) if focus_shifts else None
                })
        
        return evolution_points
    
    def _analyze_overall_drift_patterns(
        self,
        emotional_states: List[Dict[str, float]],
        stability_scores: List[float],
        emergence_points: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze overall drift patterns in the simulation."""
        patterns = []
        
        # Detect emotional drift patterns
        emotional_drift = self._detect_emotional_drift_patterns(emotional_states)
        if emotional_drift:
            patterns.append(emotional_drift)
        
        # Detect stability drift patterns
        stability_drift = self._detect_stability_drift_patterns(stability_scores)
        if stability_drift:
            patterns.append(stability_drift)
        
        # Detect emergence-driven patterns
        if emergence_points:
            emergence_pattern = self._analyze_emergence_patterns(emergence_points)
            patterns.append(emergence_pattern)
        
        return patterns
    
    def _detect_emotional_drift_patterns(
        self, 
        emotional_states: List[Dict[str, float]]
    ) -> Optional[Dict[str, Any]]:
        """Detect patterns in emotional state evolution."""
        if len(emotional_states) < 10:
            return None
        
        # Calculate emotional volatility
        volatility_scores = []
        for i in range(1, len(emotional_states)):
            prev_state = emotional_states[i-1]
            curr_state = emotional_states[i]
            
            # Calculate emotional distance
            distance = sum((curr_state.get(k, 0) - prev_state.get(k, 0))**2 
                         for k in set(prev_state) | set(curr_state))**0.5
            volatility_scores.append(distance)
        
        avg_volatility = np.mean(volatility_scores)
        volatility_trend = np.polyfit(range(len(volatility_scores)), volatility_scores, 1)[0]
        
        return {
            "pattern_type": "emotional_drift",
            "avg_volatility": avg_volatility,
            "volatility_trend": volatility_trend,
            "stability_classification": self._classify_emotional_stability(avg_volatility, volatility_trend)
        }
    
    def _detect_stability_drift_patterns(self, stability_scores: List[float]) -> Optional[Dict[str, Any]]:
        """Detect patterns in stability score evolution."""
        if len(stability_scores) < 10:
            return None
        
        # Calculate stability trends
        trend = np.polyfit(range(len(stability_scores)), stability_scores, 1)
        slope = trend[0]
        
        # Detect stability breakdown points
        breakdown_points = []
        for i in range(1, len(stability_scores)):
            if stability_scores[i] < 0.3 and stability_scores[i-1] >= 0.3:
                breakdown_points.append(i)
        
        return {
            "pattern_type": "stability_drift",
            "overall_trend": slope,
            "breakdown_points": breakdown_points,
            "final_stability": stability_scores[-1] if stability_scores else 0.0,
            "stability_classification": self._classify_stability_trend(slope, breakdown_points)
        }
    
    def _analyze_emergence_patterns(
        self, 
        emergence_points: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze patterns in emergence point distribution."""
        if not emergence_points:
            return {"pattern_type": "emergence", "count": 0}
        
        epochs = [point.get("epoch", 0) for point in emergence_points]
        confidence_scores = [point.get("confidence", 0.0) for point in emergence_points]
        
        return {
            "pattern_type": "emergence",
            "count": len(emergence_points),
            "distribution": {
                "early_emergence": len([e for e in epochs if e < len(epochs) * 0.3]),
                "mid_emergence": len([e for e in epochs if 0.3 <= e/len(epochs) <= 0.7]),
                "late_emergence": len([e for e in epochs if e > len(epochs) * 0.7])
            },
            "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0.0,
            "emergence_clusters": self._detect_emergence_clusters(epochs)
        }
    
    def _calculate_baseline(
        self, 
        emotional_states: List[Dict[str, float]], 
        stability_scores: List[float]
    ) -> Dict[str, float]:
        """Calculate baseline metrics from pre-intervention data."""
        if not emotional_states:
            return {}
        
        # Average emotional states
        avg_emotional = {}
        for state in emotional_states:
            for emotion, value in state.items():
                avg_emotional[emotion] = avg_emotional.get(emotion, 0) + value
        
        for emotion in avg_emotional:
            avg_emotional[emotion] /= len(emotional_states)
        
        # Average stability
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        return {
            "emotional_baseline": avg_emotional,
            "stability_baseline": avg_stability
        }
    
    def _calculate_post_intervention_changes(
        self,
        post_emotional_states: List[Dict[str, float]],
        post_stability_scores: List[float],
        pre_baseline: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate changes after intervention."""
        if not post_emotional_states:
            return {}
        
        # Calculate emotional changes
        emotional_changes = {}
        baseline_emotional = pre_baseline.get("emotional_baseline", {})
        
        for state in post_emotional_states:
            for emotion, value in state.items():
                baseline_value = baseline_emotional.get(emotion, 0)
                change = value - baseline_value
                emotional_changes[emotion] = emotional_changes.get(emotion, 0) + change
        
        # Average the changes
        for emotion in emotional_changes:
            emotional_changes[emotion] /= len(post_emotional_states)
        
        # Calculate stability change
        baseline_stability = pre_baseline.get("stability_baseline", 0.0)
        avg_post_stability = np.mean(post_stability_scores) if post_stability_scores else 0.0
        stability_change = avg_post_stability - baseline_stability
        
        return {
            "emotional_changes": emotional_changes,
            "stability_change": stability_change
        }
    
    def _calculate_impact_score(self, post_changes: Dict[str, Any]) -> float:
        """Calculate overall impact score from post-intervention changes."""
        emotional_changes = post_changes.get("emotional_changes", {})
        stability_change = post_changes.get("stability_change", 0.0)
        
        # Calculate emotional impact magnitude
        emotional_impact = sum(abs(v) for v in emotional_changes.values())
        
        # Combine with stability impact
        total_impact = emotional_impact + abs(stability_change)
        
        return min(total_impact, 10.0)  # Cap at 10.0
    
    def _calculate_recovery_time(
        self, 
        post_emotional_states: List[Dict[str, float]], 
        pre_baseline: Dict[str, float]
    ) -> Optional[int]:
        """Calculate time to return to baseline after intervention."""
        baseline_emotional = pre_baseline.get("emotional_baseline", {})
        
        for i, state in enumerate(post_emotional_states):
            # Check if emotional state is close to baseline
            total_diff = sum(abs(state.get(emotion, 0) - baseline_emotional.get(emotion, 0)) 
                           for emotion in set(state) | set(baseline_emotional))
            if total_diff < 0.1:  # Recovery threshold
                return i
        
        return None  # No recovery detected
    
    def _classify_emotional_stability(self, volatility: float, trend: float) -> str:
        """Classify emotional stability based on volatility and trend."""
        if volatility < 0.1 and abs(trend) < 0.01:
            return "highly_stable"
        elif volatility < 0.3 and trend < 0.05:
            return "stable"
        elif volatility < 0.5:
            return "moderate_volatility"
        else:
            return "high_volatility"
    
    def _classify_stability_trend(self, slope: float, breakdown_points: List[int]) -> str:
        """Classify stability trend based on slope and breakdowns."""
        if slope > 0.01 and not breakdown_points:
            return "improving"
        elif slope < -0.01 or breakdown_points:
            return "deteriorating"
        else:
            return "stable"
    
    def _detect_emergence_clusters(self, epochs: List[int]) -> List[Dict[str, Any]]:
        """Detect clusters of emergence points."""
        if len(epochs) < 2:
            return []
        
        clusters = []
        current_cluster = [epochs[0]]
        
        for epoch in epochs[1:]:
            if epoch - current_cluster[-1] <= 5:  # Within 5 epochs
                current_cluster.append(epoch)
            else:
                if len(current_cluster) > 1:
                    clusters.append({
                        "start_epoch": current_cluster[0],
                        "end_epoch": current_cluster[-1],
                        "size": len(current_cluster)
                    })
                current_cluster = [epoch]
        
        # Add final cluster
        if len(current_cluster) > 1:
            clusters.append({
                "start_epoch": current_cluster[0],
                "end_epoch": current_cluster[-1],
                "size": len(current_cluster)
            })
        
        return clusters
    
    def _empty_analysis_result(self) -> AnalysisResult:
        """Return empty analysis result when no data is available."""
        return AnalysisResult(
            emergence_points=[],
            stability_boundaries=[],
            intervention_leverage=[],
            attention_evolution=[],
            drift_patterns=[],
            created_at=datetime.utcnow()
        )


# Import these after the main class to avoid circular imports
from .pattern_detector import PatternDetector
from .stability_analyzer import StabilityAnalyzer 