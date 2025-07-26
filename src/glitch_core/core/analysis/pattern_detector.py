"""
Pattern detection algorithms for identifying emergence points and pattern changes.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from scipy import signal
from scipy.stats import zscore

from glitch_core.config.logging import get_logger


class PatternDetector:
    """
    Detects pattern emergence points and changes in personality evolution.
    """
    
    def __init__(self):
        self.logger = get_logger("pattern_detector")
    
    def detect_emergence_points(
        self, 
        emotional_states: List[Dict[str, float]], 
        stability_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Detect emergence points where new patterns emerge.
        
        Args:
            emotional_states: List of emotional state dictionaries
            stability_scores: List of stability scores
            
        Returns:
            List of emergence point dictionaries
        """
        if len(emotional_states) < 10:
            return []
        
        emergence_points = []
        
        # Detect emotional pattern emergence
        emotional_emergence = self._detect_emotional_emergence(emotional_states)
        emergence_points.extend(emotional_emergence)
        
        # Detect stability pattern emergence
        stability_emergence = self._detect_stability_emergence(stability_scores)
        emergence_points.extend(stability_emergence)
        
        # Detect cross-correlation emergence
        correlation_emergence = self._detect_correlation_emergence(
            emotional_states, stability_scores
        )
        emergence_points.extend(correlation_emergence)
        
        # Sort by epoch and remove duplicates
        emergence_points.sort(key=lambda x: x.get("epoch", 0))
        return self._deduplicate_emergence_points(emergence_points)
    
    def _detect_emotional_emergence(
        self, 
        emotional_states: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Detect emergence points in emotional patterns."""
        emergence_points = []
        
        # Convert to numerical arrays for analysis
        emotion_arrays = self._extract_emotion_arrays(emotional_states)
        
        for emotion_name, values in emotion_arrays.items():
            if len(values) < 5:
                continue
            
            # Detect sudden changes using change point detection
            change_points = self._detect_change_points(values)
            
            for point in change_points:
                confidence = self._calculate_emergence_confidence(values, point)
                if confidence > 0.6:  # High confidence threshold
                    emergence_points.append({
                        "epoch": point,
                        "pattern_type": "emotional_emergence",
                        "emotion": emotion_name,
                        "confidence": confidence,
                        "change_magnitude": abs(values[point] - values[max(0, point-1)]),
                        "description": f"Emergence of new {emotion_name} pattern"
                    })
        
        return emergence_points
    
    def _detect_stability_emergence(self, stability_scores: List[float]) -> List[Dict[str, Any]]:
        """Detect emergence points in stability patterns."""
        if len(stability_scores) < 5:
            return []
        
        emergence_points = []
        
        # Detect stability breakdown points
        breakdown_points = self._detect_stability_breakdowns(stability_scores)
        
        for point in breakdown_points:
            confidence = self._calculate_stability_emergence_confidence(stability_scores, point)
            if confidence > 0.7:  # High confidence for stability changes
                emergence_points.append({
                    "epoch": point,
                    "pattern_type": "stability_emergence",
                    "confidence": confidence,
                    "stability_drop": stability_scores[point] - stability_scores[max(0, point-1)],
                    "description": "Stability pattern breakdown detected"
                })
        
        # Detect stability recovery points
        recovery_points = self._detect_stability_recoveries(stability_scores)
        
        for point in recovery_points:
            confidence = self._calculate_stability_emergence_confidence(stability_scores, point)
            if confidence > 0.6:
                emergence_points.append({
                    "epoch": point,
                    "pattern_type": "stability_recovery",
                    "confidence": confidence,
                    "stability_gain": stability_scores[point] - stability_scores[max(0, point-1)],
                    "description": "Stability pattern recovery detected"
                })
        
        return emergence_points
    
    def _detect_correlation_emergence(
        self, 
        emotional_states: List[Dict[str, float]], 
        stability_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect emergence of new correlation patterns between emotions and stability."""
        if len(emotional_states) < 10 or len(stability_scores) < 10:
            return []
        
        emergence_points = []
        
        # Calculate rolling correlations
        window_size = min(10, len(emotional_states) // 2)
        emotion_arrays = self._extract_emotion_arrays(emotional_states)
        
        for emotion_name, values in emotion_arrays.items():
            if len(values) < window_size:
                continue
            
            # Calculate rolling correlation with stability
            correlations = self._calculate_rolling_correlation(values, stability_scores, window_size)
            
            # Detect correlation change points
            correlation_changes = self._detect_correlation_changes(correlations)
            
            for point in correlation_changes:
                if point >= window_size:  # Adjust for window offset
                    confidence = self._calculate_correlation_confidence(correlations, point)
                    if confidence > 0.5:
                        emergence_points.append({
                            "epoch": point,
                            "pattern_type": "correlation_emergence",
                            "emotion": emotion_name,
                            "confidence": confidence,
                            "correlation_change": correlations[point] - correlations[max(0, point-1)],
                            "description": f"New correlation pattern between {emotion_name} and stability"
                        })
        
        return emergence_points
    
    def _extract_emotion_arrays(self, emotional_states: List[Dict[str, float]]) -> Dict[str, List[float]]:
        """Extract emotion values into separate arrays."""
        emotion_arrays = {}
        
        for state in emotional_states:
            for emotion, value in state.items():
                if emotion not in emotion_arrays:
                    emotion_arrays[emotion] = []
                emotion_arrays[emotion].append(value)
        
        return emotion_arrays
    
    def _detect_change_points(self, values: List[float]) -> List[int]:
        """Detect change points in a time series using statistical methods."""
        if len(values) < 5:
            return []
        
        # Convert to numpy array
        arr = np.array(values)
        
        # Calculate z-scores for outlier detection
        z_scores = zscore(arr)
        
        # Find points where z-score exceeds threshold
        threshold = 2.0
        change_points = np.where(np.abs(z_scores) > threshold)[0]
        
        # Filter out consecutive points (keep only the first in a cluster)
        filtered_points = []
        for point in change_points:
            if not filtered_points or point - filtered_points[-1] > 3:
                filtered_points.append(point)
        
        return filtered_points.tolist()
    
    def _detect_stability_breakdowns(self, stability_scores: List[float]) -> List[int]:
        """Detect points where stability breaks down."""
        breakdown_points = []
        
        for i in range(1, len(stability_scores)):
            # Detect significant drops in stability
            if (stability_scores[i] < 0.3 and 
                stability_scores[i-1] >= 0.3 and 
                stability_scores[i] < stability_scores[i-1] - 0.2):
                breakdown_points.append(i)
        
        return breakdown_points
    
    def _detect_stability_recoveries(self, stability_scores: List[float]) -> List[int]:
        """Detect points where stability recovers."""
        recovery_points = []
        
        for i in range(1, len(stability_scores)):
            # Detect significant improvements in stability
            if (stability_scores[i] > 0.6 and 
                stability_scores[i-1] <= 0.4 and 
                stability_scores[i] > stability_scores[i-1] + 0.2):
                recovery_points.append(i)
        
        return recovery_points
    
    def _calculate_rolling_correlation(
        self, 
        values1: List[float], 
        values2: List[float], 
        window_size: int
    ) -> List[float]:
        """Calculate rolling correlation between two time series."""
        correlations = []
        
        for i in range(window_size, len(values1)):
            window1 = values1[i-window_size:i]
            window2 = values2[i-window_size:i]
            
            if len(window1) == len(window2) and len(window1) > 1:
                correlation = np.corrcoef(window1, window2)[0, 1]
                correlations.append(correlation if not np.isnan(correlation) else 0.0)
            else:
                correlations.append(0.0)
        
        return correlations
    
    def _detect_correlation_changes(self, correlations: List[float]) -> List[int]:
        """Detect significant changes in correlation patterns."""
        if len(correlations) < 3:
            return []
        
        # Detect points where correlation changes significantly
        change_points = []
        
        for i in range(1, len(correlations)):
            if abs(correlations[i] - correlations[i-1]) > 0.3:
                change_points.append(i)
        
        return change_points
    
    def _calculate_emergence_confidence(self, values: List[float], point: int) -> float:
        """Calculate confidence score for an emergence point."""
        if point >= len(values) or point < 1:
            return 0.0
        
        # Calculate the magnitude of change
        change_magnitude = abs(values[point] - values[point-1])
        
        # Calculate local volatility
        window_size = min(5, point)
        local_values = values[max(0, point-window_size):point]
        if local_values:
            local_std = np.std(local_values)
            volatility_ratio = change_magnitude / (local_std + 1e-6)
        else:
            volatility_ratio = 0.0
        
        # Combine factors for confidence
        confidence = min(1.0, (change_magnitude * 2 + volatility_ratio * 0.5) / 3)
        
        return confidence
    
    def _calculate_stability_emergence_confidence(self, values: List[float], point: int) -> float:
        """Calculate confidence for stability emergence points."""
        if point >= len(values) or point < 1:
            return 0.0
        
        # Calculate the magnitude of stability change
        change_magnitude = abs(values[point] - values[point-1])
        
        # Higher confidence for larger changes
        confidence = min(1.0, change_magnitude * 2)
        
        return confidence
    
    def _calculate_correlation_confidence(self, correlations: List[float], point: int) -> float:
        """Calculate confidence for correlation emergence points."""
        if point >= len(correlations) or point < 1:
            return 0.0
        
        # Calculate the magnitude of correlation change
        change_magnitude = abs(correlations[point] - correlations[point-1])
        
        # Higher confidence for larger correlation changes
        confidence = min(1.0, change_magnitude * 2)
        
        return confidence
    
    def _deduplicate_emergence_points(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate emergence points that are too close together."""
        if not points:
            return []
        
        deduplicated = [points[0]]
        
        for point in points[1:]:
            # Check if this point is too close to the last one
            last_point = deduplicated[-1]
            if abs(point.get("epoch", 0) - last_point.get("epoch", 0)) > 3:
                deduplicated.append(point)
            else:
                # Keep the one with higher confidence
                if point.get("confidence", 0) > last_point.get("confidence", 0):
                    deduplicated[-1] = point
        
        return deduplicated 