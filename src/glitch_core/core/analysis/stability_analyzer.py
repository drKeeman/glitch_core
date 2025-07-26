"""
Stability analysis algorithms for detecting boundaries and breakdown conditions.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

from glitch_core.config.logging import get_logger


class StabilityAnalyzer:
    """
    Analyzes stability boundaries and breakdown conditions in personality evolution.
    """
    
    def __init__(self):
        self.logger = get_logger("stability_analyzer")
    
    def analyze_stability_boundaries(
        self, 
        stability_scores: List[float], 
        emotional_states: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze stability boundaries and breakdown conditions.
        
        Args:
            stability_scores: List of stability scores over time
            emotional_states: List of emotional state dictionaries
            
        Returns:
            List of stability boundary analysis results
        """
        if len(stability_scores) < 10:
            return []
        
        boundaries = []
        
        # Detect critical stability thresholds
        threshold_boundaries = self._detect_stability_thresholds(stability_scores)
        boundaries.extend(threshold_boundaries)
        
        # Detect stability trend changes
        trend_boundaries = self._detect_stability_trends(stability_scores)
        boundaries.extend(trend_boundaries)
        
        # Detect emotional-stability coupling breakdowns
        coupling_boundaries = self._detect_coupling_breakdowns(stability_scores, emotional_states)
        boundaries.extend(coupling_boundaries)
        
        # Detect stability oscillation patterns
        oscillation_boundaries = self._detect_stability_oscillations(stability_scores)
        boundaries.extend(oscillation_boundaries)
        
        # Sort by epoch
        boundaries.sort(key=lambda x: x.get("epoch", 0))
        
        return boundaries
    
    def _detect_stability_thresholds(self, stability_scores: List[float]) -> List[Dict[str, Any]]:
        """Detect critical stability threshold crossings."""
        boundaries = []
        
        # Define critical thresholds
        critical_thresholds = [0.2, 0.3, 0.5, 0.7, 0.8]
        
        for i in range(1, len(stability_scores)):
            prev_score = stability_scores[i-1]
            curr_score = stability_scores[i]
            
            for threshold in critical_thresholds:
                # Detect downward crossing
                if prev_score >= threshold and curr_score < threshold:
                    boundaries.append({
                        "epoch": i,
                        "boundary_type": "threshold_crossing",
                        "threshold": threshold,
                        "direction": "downward",
                        "severity": self._calculate_threshold_severity(threshold, curr_score),
                        "description": f"Stability dropped below critical threshold {threshold}"
                    })
                
                # Detect upward crossing
                elif prev_score < threshold and curr_score >= threshold:
                    boundaries.append({
                        "epoch": i,
                        "boundary_type": "threshold_crossing",
                        "threshold": threshold,
                        "direction": "upward",
                        "severity": self._calculate_threshold_severity(threshold, curr_score),
                        "description": f"Stability recovered above threshold {threshold}"
                    })
        
        return boundaries
    
    def _detect_stability_trends(self, stability_scores: List[float]) -> List[Dict[str, Any]]:
        """Detect significant changes in stability trends."""
        if len(stability_scores) < 15:
            return []
        
        boundaries = []
        
        # Calculate rolling trends
        window_size = 10
        trends = []
        
        for i in range(window_size, len(stability_scores)):
            window = stability_scores[i-window_size:i]
            trend = np.polyfit(range(len(window)), window, 1)[0]
            trends.append(trend)
        
        # Detect trend change points
        for i in range(1, len(trends)):
            trend_change = abs(trends[i] - trends[i-1])
            
            if trend_change > 0.02:  # Significant trend change
                epoch = i + window_size
                boundaries.append({
                    "epoch": epoch,
                    "boundary_type": "trend_change",
                    "previous_trend": trends[i-1],
                    "current_trend": trends[i],
                    "trend_change": trend_change,
                    "direction": "improving" if trends[i] > trends[i-1] else "deteriorating",
                    "description": f"Stability trend changed from {trends[i-1]:.3f} to {trends[i]:.3f}"
                })
        
        return boundaries
    
    def _detect_coupling_breakdowns(
        self, 
        stability_scores: List[float], 
        emotional_states: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Detect breakdowns in emotional-stability coupling."""
        if len(stability_scores) < 10 or len(emotional_states) < 10:
            return []
        
        boundaries = []
        
        # Calculate emotional volatility
        emotional_volatility = self._calculate_emotional_volatility(emotional_states)
        
        # Detect decoupling points where stability and emotional volatility diverge
        for i in range(10, len(stability_scores)):
            # Calculate correlation between stability and emotional volatility
            stability_window = stability_scores[i-10:i]
            volatility_window = emotional_volatility[i-10:i]
            
            if len(stability_window) == len(volatility_window) and len(stability_window) > 1:
                correlation = np.corrcoef(stability_window, volatility_window)[0, 1]
                
                # Detect breakdown in negative correlation (stability should be negatively correlated with volatility)
                if not np.isnan(correlation) and correlation > -0.3:
                    boundaries.append({
                        "epoch": i,
                        "boundary_type": "coupling_breakdown",
                        "correlation": correlation,
                        "stability_score": stability_scores[i],
                        "volatility_score": emotional_volatility[i],
                        "description": f"Emotional-stability coupling breakdown (correlation: {correlation:.3f})"
                    })
        
        return boundaries
    
    def _detect_stability_oscillations(self, stability_scores: List[float]) -> List[Dict[str, Any]]:
        """Detect oscillation patterns in stability."""
        if len(stability_scores) < 20:
            return []
        
        boundaries = []
        
        # Find peaks and troughs in stability
        peaks, _ = find_peaks(stability_scores, height=0.5, distance=5)
        troughs, _ = find_peaks([-s for s in stability_scores], height=-0.3, distance=5)
        
        # Detect oscillation patterns
        if len(peaks) >= 2 and len(troughs) >= 2:
            # Calculate oscillation characteristics
            peak_heights = [stability_scores[p] for p in peaks]
            trough_heights = [stability_scores[t] for t in troughs]
            
            avg_peak_height = np.mean(peak_heights)
            avg_trough_height = np.mean(trough_heights)
            oscillation_amplitude = avg_peak_height - avg_trough_height
            
            if oscillation_amplitude > 0.3:  # Significant oscillation
                boundaries.append({
                    "epoch": peaks[-1] if peaks else 0,
                    "boundary_type": "oscillation_pattern",
                    "amplitude": oscillation_amplitude,
                    "peak_count": len(peaks),
                    "trough_count": len(troughs),
                    "avg_peak_height": avg_peak_height,
                    "avg_trough_height": avg_trough_height,
                    "description": f"Stability oscillation detected (amplitude: {oscillation_amplitude:.3f})"
                })
        
        return boundaries
    
    def _calculate_emotional_volatility(self, emotional_states: List[Dict[str, float]]) -> List[float]:
        """Calculate emotional volatility over time."""
        volatility_scores = []
        
        for i in range(1, len(emotional_states)):
            prev_state = emotional_states[i-1]
            curr_state = emotional_states[i]
            
            # Calculate emotional distance
            distance = sum((curr_state.get(k, 0) - prev_state.get(k, 0))**2 
                         for k in set(prev_state) | set(curr_state))**0.5
            volatility_scores.append(distance)
        
        return volatility_scores
    
    def _calculate_threshold_severity(self, threshold: float, current_score: float) -> str:
        """Calculate severity of threshold crossing."""
        distance_below = threshold - current_score
        
        if distance_below > 0.3:
            return "critical"
        elif distance_below > 0.2:
            return "severe"
        elif distance_below > 0.1:
            return "moderate"
        else:
            return "mild"
    
    def find_breakdown_points(self, stability_scores: List[float]) -> List[Dict[str, Any]]:
        """Find specific breakdown points in stability."""
        breakdown_points = []
        
        for i in range(1, len(stability_scores)):
            # Detect sudden drops
            if (stability_scores[i] < stability_scores[i-1] - 0.2 and 
                stability_scores[i] < 0.4):
                breakdown_points.append({
                    "epoch": i,
                    "stability_drop": stability_scores[i-1] - stability_scores[i],
                    "final_stability": stability_scores[i],
                    "severity": self._classify_breakdown_severity(stability_scores[i])
                })
        
        return breakdown_points
    
    def _classify_breakdown_severity(self, stability_score: float) -> str:
        """Classify the severity of a stability breakdown."""
        if stability_score < 0.2:
            return "catastrophic"
        elif stability_score < 0.3:
            return "severe"
        elif stability_score < 0.4:
            return "moderate"
        else:
            return "mild"
    
    def calculate_stability_metrics(self, stability_scores: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive stability metrics."""
        if not stability_scores:
            return {}
        
        return {
            "mean_stability": np.mean(stability_scores),
            "std_stability": np.std(stability_scores),
            "min_stability": np.min(stability_scores),
            "max_stability": np.max(stability_scores),
            "stability_trend": np.polyfit(range(len(stability_scores)), stability_scores, 1)[0],
            "volatility": np.std(np.diff(stability_scores)),
            "breakdown_count": len([s for s in stability_scores if s < 0.3]),
            "recovery_count": len([s for s in stability_scores if s > 0.7])
        } 