"""
Comprehensive drift analysis for personality evolution patterns.
"""

import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from glitch_core.config.logging import get_logger


@dataclass
class DriftPattern:
    """A detected drift pattern in personality evolution."""
    pattern_type: str
    start_epoch: int
    end_epoch: int
    confidence: float
    characteristics: Dict[str, Any]
    impact_score: float
    description: str


@dataclass
class StabilityAnalysis:
    """Analysis of personality stability over time."""
    overall_stability: float
    emotional_volatility: float
    breakdown_risk: float
    resilience_score: float
    adaptation_rate: float
    stability_trends: List[Dict[str, Any]]
    critical_points: List[Dict[str, Any]]


@dataclass
class PersonalityEvolution:
    """Analysis of personality trait evolution."""
    trait_evolution: Dict[str, List[float]]
    trait_correlations: Dict[str, float]
    evolution_clusters: List[Dict[str, Any]]
    significant_changes: List[Dict[str, Any]]
    evolution_summary: Dict[str, Any]


class DriftAnalyzer:
    """
    Comprehensive analyzer for personality drift patterns and evolution.
    """
    
    def __init__(self):
        self.logger = get_logger("drift_analyzer")
        
        # Analysis parameters
        self.stability_threshold = 0.3
        self.volatility_threshold = 0.5
        self.pattern_confidence_threshold = 0.7
        
    def analyze_stability_trends(
        self, 
        emotional_states: List[Dict[str, float]]
    ) -> StabilityAnalysis:
        """
        Analyze stability trends in personality evolution.
        
        Args:
            emotional_states: List of emotional state dictionaries over time
            
        Returns:
            Stability analysis results
        """
        if len(emotional_states) < 10:
            return self._create_empty_stability_analysis()
        
        try:
            # Calculate stability metrics
            stability_scores = self._calculate_stability_scores(emotional_states)
            volatility_scores = self._calculate_volatility_scores(emotional_states)
            
            # Detect trends
            stability_trends = self._detect_stability_trends(stability_scores)
            critical_points = self._detect_critical_points(stability_scores, volatility_scores)
            
            # Calculate overall metrics
            overall_stability = np.mean(stability_scores)
            emotional_volatility = np.mean(volatility_scores)
            breakdown_risk = self._calculate_breakdown_risk(stability_scores, volatility_scores)
            resilience_score = self._calculate_resilience_score(stability_scores)
            adaptation_rate = self._calculate_adaptation_rate(emotional_states)
            
            return StabilityAnalysis(
                overall_stability=overall_stability,
                emotional_volatility=emotional_volatility,
                breakdown_risk=breakdown_risk,
                resilience_score=resilience_score,
                adaptation_rate=adaptation_rate,
                stability_trends=stability_trends,
                critical_points=critical_points
            )
            
        except Exception as e:
            self.logger.error("stability_analysis_failed", error=str(e))
            return self._create_empty_stability_analysis()
    
    def detect_pattern_emergence(
        self, 
        states: List[Dict[str, float]]
    ) -> List[DriftPattern]:
        """
        Detect emerging patterns in personality evolution.
        
        Args:
            states: List of emotional/personality states over time
            
        Returns:
            List of detected patterns
        """
        if len(states) < 20:
            return []
        
        try:
            patterns = []
            
            # Detect oscillation patterns
            oscillation_patterns = self._detect_oscillation_patterns(states)
            patterns.extend(oscillation_patterns)
            
            # Detect trend patterns
            trend_patterns = self._detect_trend_patterns(states)
            patterns.extend(trend_patterns)
            
            # Detect clustering patterns
            clustering_patterns = self._detect_clustering_patterns(states)
            patterns.extend(clustering_patterns)
            
            # Detect sudden change patterns
            sudden_change_patterns = self._detect_sudden_change_patterns(states)
            patterns.extend(sudden_change_patterns)
            
            # Sort by confidence and filter
            patterns.sort(key=lambda p: p.confidence, reverse=True)
            patterns = [p for p in patterns if p.confidence >= self.pattern_confidence_threshold]
            
            self.logger.info(
                "pattern_detection_completed",
                total_patterns=len(patterns),
                high_confidence_patterns=len([p for p in patterns if p.confidence > 0.8])
            )
            
            return patterns
            
        except Exception as e:
            self.logger.error("pattern_detection_failed", error=str(e))
            return []
    
    def analyze_personality_evolution(
        self, 
        trait_evolution: Dict[str, List[float]]
    ) -> PersonalityEvolution:
        """
        Analyze personality trait evolution over time.
        
        Args:
            trait_evolution: Dictionary mapping trait names to evolution values
            
        Returns:
            Personality evolution analysis
        """
        if not trait_evolution:
            return self._create_empty_personality_evolution()
        
        try:
            # Calculate trait correlations
            trait_correlations = self._calculate_trait_correlations(trait_evolution)
            
            # Detect evolution clusters
            evolution_clusters = self._detect_evolution_clusters(trait_evolution)
            
            # Detect significant changes
            significant_changes = self._detect_significant_changes(trait_evolution)
            
            # Create evolution summary
            evolution_summary = self._create_evolution_summary(trait_evolution)
            
            return PersonalityEvolution(
                trait_evolution=trait_evolution,
                trait_correlations=trait_correlations,
                evolution_clusters=evolution_clusters,
                significant_changes=significant_changes,
                evolution_summary=evolution_summary
            )
            
        except Exception as e:
            self.logger.error("personality_evolution_analysis_failed", error=str(e))
            return self._create_empty_personality_evolution()
    
    def compare_personas(
        self, 
        persona_data: Dict[str, List[Dict[str, float]]]
    ) -> Dict[str, Any]:
        """
        Compare multiple personas for analysis.
        
        Args:
            persona_data: Dictionary mapping persona IDs to state lists
            
        Returns:
            Comparison analysis results
        """
        if len(persona_data) < 2:
            return {"error": "Need at least 2 personas for comparison"}
        
        try:
            comparison_results = {
                "persona_count": len(persona_data),
                "comparisons": {},
                "similarity_matrix": {},
                "divergence_points": [],
                "convergence_points": []
            }
            
            # Calculate pairwise comparisons
            persona_ids = list(persona_data.keys())
            for i, persona1 in enumerate(persona_ids):
                for j, persona2 in enumerate(persona_ids[i+1:], i+1):
                    comparison = self._compare_two_personas(
                        persona_data[persona1], 
                        persona_data[persona2]
                    )
                    comparison_results["comparisons"][f"{persona1}_vs_{persona2}"] = comparison
            
            # Calculate similarity matrix
            similarity_matrix = self._calculate_similarity_matrix(persona_data)
            comparison_results["similarity_matrix"] = similarity_matrix
            
            # Detect divergence and convergence points
            divergence_points = self._detect_divergence_points(persona_data)
            convergence_points = self._detect_convergence_points(persona_data)
            
            comparison_results["divergence_points"] = divergence_points
            comparison_results["convergence_points"] = convergence_points
            
            return comparison_results
            
        except Exception as e:
            self.logger.error("persona_comparison_failed", error=str(e))
            return {"error": f"Comparison failed: {str(e)}"}
    
    def export_analysis_results(
        self, 
        analysis_results: Dict[str, Any],
        format: str = "json"
    ) -> Union[str, bytes]:
        """
        Export analysis results in various formats.
        
        Args:
            analysis_results: Analysis results to export
            format: Export format ("json", "csv", "excel")
            
        Returns:
            Exported data in specified format
        """
        try:
            if format.lower() == "json":
                return json.dumps(analysis_results, indent=2, default=str)
            
            elif format.lower() == "csv":
                return self._export_to_csv(analysis_results)
            
            elif format.lower() == "excel":
                return self._export_to_excel(analysis_results)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error("export_failed", format=format, error=str(e))
            return f"Export failed: {str(e)}"
    
    def _calculate_stability_scores(self, emotional_states: List[Dict[str, float]]) -> List[float]:
        """Calculate stability scores for each time point."""
        stability_scores = []
        
        for i in range(len(emotional_states)):
            if i == 0:
                stability_scores.append(1.0)  # First point is stable
                continue
            
            # Calculate variance from previous state
            current_state = emotional_states[i]
            previous_state = emotional_states[i-1]
            
            # Get common emotions
            common_emotions = set(current_state.keys()) & set(previous_state.keys())
            
            if not common_emotions:
                stability_scores.append(0.5)  # Neutral if no common emotions
                continue
            
            # Calculate stability as inverse of change
            changes = []
            for emotion in common_emotions:
                change = abs(current_state[emotion] - previous_state[emotion])
                changes.append(change)
            
            avg_change = np.mean(changes)
            stability = max(0.0, 1.0 - avg_change)
            stability_scores.append(stability)
        
        return stability_scores
    
    def _calculate_volatility_scores(self, emotional_states: List[Dict[str, float]]) -> List[float]:
        """Calculate volatility scores for each time point."""
        volatility_scores = []
        
        for i in range(len(emotional_states)):
            if i < 2:
                volatility_scores.append(0.0)  # Need at least 3 points for volatility
                continue
            
            # Calculate volatility over a window
            window_size = min(5, i + 1)
            window_states = emotional_states[i-window_size+1:i+1]
            
            # Calculate variance across all emotions
            all_values = []
            for state in window_states:
                all_values.extend(state.values())
            
            volatility = np.std(all_values)
            volatility_scores.append(volatility)
        
        return volatility_scores
    
    def _detect_stability_trends(self, stability_scores: List[float]) -> List[Dict[str, Any]]:
        """Detect trends in stability scores."""
        trends = []
        
        if len(stability_scores) < 10:
            return trends
        
        # Use rolling window to detect trends
        window_size = min(10, len(stability_scores) // 2)
        
        for i in range(window_size, len(stability_scores)):
            window = stability_scores[i-window_size:i+1]
            
            # Calculate trend
            x = np.arange(len(window))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, window)
            
            if abs(slope) > 0.01:  # Significant trend
                trend_type = "improving" if slope > 0 else "declining"
                trends.append({
                    "epoch": i,
                    "trend_type": trend_type,
                    "slope": slope,
                    "confidence": abs(r_value),
                    "window_size": window_size
                })
        
        return trends
    
    def _detect_critical_points(
        self, 
        stability_scores: List[float], 
        volatility_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect critical points in stability/volatility."""
        critical_points = []
        
        for i in range(len(stability_scores)):
            stability = stability_scores[i]
            volatility = volatility_scores[i] if i < len(volatility_scores) else 0.0
            
            # Detect critical conditions
            if stability < self.stability_threshold:
                critical_points.append({
                    "epoch": i,
                    "type": "low_stability",
                    "severity": 1.0 - stability,
                    "stability_score": stability,
                    "volatility_score": volatility
                })
            
            if volatility > self.volatility_threshold:
                critical_points.append({
                    "epoch": i,
                    "type": "high_volatility",
                    "severity": volatility,
                    "stability_score": stability,
                    "volatility_score": volatility
                })
        
        return critical_points
    
    def _calculate_breakdown_risk(
        self, 
        stability_scores: List[float], 
        volatility_scores: List[float]
    ) -> float:
        """Calculate overall breakdown risk."""
        if not stability_scores or not volatility_scores:
            return 0.0
        
        # Risk factors
        low_stability_risk = len([s for s in stability_scores if s < self.stability_threshold]) / len(stability_scores)
        high_volatility_risk = len([v for v in volatility_scores if v > self.volatility_threshold]) / len(volatility_scores)
        
        # Combined risk
        breakdown_risk = (low_stability_risk + high_volatility_risk) / 2
        return min(1.0, breakdown_risk)
    
    def _calculate_resilience_score(self, stability_scores: List[float]) -> float:
        """Calculate resilience score based on recovery patterns."""
        if len(stability_scores) < 10:
            return 0.5
        
        # Count recovery events (stability increases after low points)
        recovery_count = 0
        low_stability_count = 0
        
        for i in range(1, len(stability_scores)):
            if stability_scores[i-1] < self.stability_threshold:
                low_stability_count += 1
                if stability_scores[i] > stability_scores[i-1]:
                    recovery_count += 1
        
        if low_stability_count == 0:
            return 1.0  # No low points = high resilience
        
        resilience = recovery_count / low_stability_count
        return min(1.0, resilience)
    
    def _calculate_adaptation_rate(self, emotional_states: List[Dict[str, float]]) -> float:
        """Calculate adaptation rate based on emotional state changes."""
        if len(emotional_states) < 10:
            return 0.0
        
        # Calculate rate of change over time
        changes = []
        for i in range(1, len(emotional_states)):
            current = emotional_states[i]
            previous = emotional_states[i-1]
            
            # Calculate total change
            total_change = sum(abs(current.get(k, 0) - previous.get(k, 0)) for k in set(current.keys()) | set(previous.keys()))
            changes.append(total_change)
        
        # Adaptation rate is the trend in change magnitude
        if len(changes) < 3:
            return 0.0
        
        x = np.arange(len(changes))
        slope, _, _, _, _ = stats.linregress(x, changes)
        
        # Normalize adaptation rate
        adaptation_rate = 1.0 / (1.0 + math.exp(-slope * 10))  # Sigmoid normalization
        return adaptation_rate
    
    def _detect_oscillation_patterns(self, states: List[Dict[str, float]]) -> List[DriftPattern]:
        """Detect oscillation patterns in personality evolution."""
        patterns = []
        
        # Convert states to numerical series for analysis
        emotion_series = self._extract_emotion_series(states)
        
        for emotion, values in emotion_series.items():
            if len(values) < 10:
                continue
            
            # Find peaks and troughs
            peaks, _ = find_peaks(values)
            troughs, _ = find_peaks([-v for v in values])
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                # Calculate oscillation characteristics
                oscillation_frequency = len(peaks) / len(values)
                oscillation_amplitude = np.std(values)
                
                if oscillation_frequency > 0.1 and oscillation_amplitude > 0.1:
                    pattern = DriftPattern(
                        pattern_type="oscillation",
                        start_epoch=0,
                        end_epoch=len(values) - 1,
                        confidence=min(1.0, oscillation_frequency * 5),
                        characteristics={
                            "emotion": emotion,
                            "frequency": oscillation_frequency,
                            "amplitude": oscillation_amplitude,
                            "peak_count": len(peaks),
                            "trough_count": len(troughs)
                        },
                        impact_score=oscillation_amplitude,
                        description=f"Oscillation pattern in {emotion} with frequency {oscillation_frequency:.3f}"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_trend_patterns(self, states: List[Dict[str, float]]) -> List[DriftPattern]:
        """Detect trend patterns in personality evolution."""
        patterns = []
        
        emotion_series = self._extract_emotion_series(states)
        
        for emotion, values in emotion_series.items():
            if len(values) < 10:
                continue
            
            # Calculate trend
            x = np.arange(len(values))
            slope, _, r_value, _, _ = stats.linregress(x, values)
            
            if abs(slope) > 0.01 and abs(r_value) > 0.5:
                trend_type = "increasing" if slope > 0 else "decreasing"
                pattern = DriftPattern(
                    pattern_type=f"{trend_type}_trend",
                    start_epoch=0,
                    end_epoch=len(values) - 1,
                    confidence=abs(r_value),
                    characteristics={
                        "emotion": emotion,
                        "slope": slope,
                        "r_squared": r_value ** 2,
                        "trend_type": trend_type
                    },
                    impact_score=abs(slope),
                    description=f"{trend_type.capitalize()} trend in {emotion} (slope: {slope:.3f})"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_clustering_patterns(self, states: List[Dict[str, float]]) -> List[DriftPattern]:
        """Detect clustering patterns in personality states."""
        patterns = []
        
        if len(states) < 20:
            return patterns
        
        # Convert states to feature vectors
        feature_vectors = []
        for state in states:
            vector = list(state.values())
            feature_vectors.append(vector)
        
        # Apply PCA for dimensionality reduction
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(feature_vectors)
        
        # Apply clustering
        n_clusters = min(3, len(scaled_vectors) // 10)
        if n_clusters < 2:
            return patterns
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_vectors)
        
        # Analyze cluster patterns
        unique_labels = set(cluster_labels)
        if len(unique_labels) > 1:
            # Calculate cluster characteristics
            cluster_sizes = [list(cluster_labels).count(label) for label in unique_labels]
            cluster_ratio = max(cluster_sizes) / sum(cluster_sizes)
            
            if cluster_ratio < 0.8:  # Not dominated by one cluster
                pattern = DriftPattern(
                    pattern_type="clustering",
                    start_epoch=0,
                    end_epoch=len(states) - 1,
                    confidence=1.0 - cluster_ratio,
                    characteristics={
                        "cluster_count": len(unique_labels),
                        "cluster_sizes": cluster_sizes,
                        "dominance_ratio": cluster_ratio
                    },
                    impact_score=1.0 - cluster_ratio,
                    description=f"Clustering pattern with {len(unique_labels)} distinct clusters"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_sudden_change_patterns(self, states: List[Dict[str, float]]) -> List[DriftPattern]:
        """Detect sudden change patterns in personality evolution."""
        patterns = []
        
        emotion_series = self._extract_emotion_series(states)
        
        for emotion, values in emotion_series.items():
            if len(values) < 10:
                continue
            
            # Calculate change magnitudes
            changes = []
            for i in range(1, len(values)):
                change = abs(values[i] - values[i-1])
                changes.append(change)
            
            # Find sudden changes (outliers)
            if changes:
                mean_change = np.mean(changes)
                std_change = np.std(changes)
                
                sudden_changes = []
                for i, change in enumerate(changes):
                    if change > mean_change + 2 * std_change:
                        sudden_changes.append(i + 1)  # +1 because changes[i] corresponds to epoch i+1
                
                if len(sudden_changes) >= 2:
                    pattern = DriftPattern(
                        pattern_type="sudden_changes",
                        start_epoch=min(sudden_changes),
                        end_epoch=max(sudden_changes),
                        confidence=min(1.0, len(sudden_changes) / 5),
                        characteristics={
                            "emotion": emotion,
                            "sudden_change_count": len(sudden_changes),
                            "change_epochs": sudden_changes,
                            "average_change_magnitude": mean_change
                        },
                        impact_score=len(sudden_changes) / len(values),
                        description=f"Sudden changes in {emotion} at epochs {sudden_changes}"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _extract_emotion_series(self, states: List[Dict[str, float]]) -> Dict[str, List[float]]:
        """Extract time series for each emotion."""
        emotion_series = {}
        
        for state in states:
            for emotion, value in state.items():
                if emotion not in emotion_series:
                    emotion_series[emotion] = []
                emotion_series[emotion].append(value)
        
        return emotion_series
    
    def _calculate_trait_correlations(self, trait_evolution: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate correlations between personality traits."""
        correlations = {}
        trait_names = list(trait_evolution.keys())
        
        for i, trait1 in enumerate(trait_names):
            for j, trait2 in enumerate(trait_names[i+1:], i+1):
                values1 = trait_evolution[trait1]
                values2 = trait_evolution[trait2]
                
                if len(values1) == len(values2) and len(values1) > 1:
                    correlation, _ = stats.pearsonr(values1, values2)
                    correlations[f"{trait1}_vs_{trait2}"] = correlation
        
        return correlations
    
    def _detect_evolution_clusters(self, trait_evolution: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Detect clusters in trait evolution patterns."""
        clusters = []
        
        if len(trait_evolution) < 2:
            return clusters
        
        # Convert to feature matrix
        trait_names = list(trait_evolution.keys())
        feature_matrix = []
        
        for trait in trait_names:
            values = trait_evolution[trait]
            # Use summary statistics as features
            features = [
                np.mean(values),
                np.std(values),
                np.max(values),
                np.min(values),
                np.ptp(values)  # peak-to-peak
            ]
            feature_matrix.append(features)
        
        # Apply clustering
        if len(feature_matrix) >= 3:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            n_clusters = min(3, len(scaled_features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Group traits by cluster
            for cluster_id in set(cluster_labels):
                cluster_traits = [trait_names[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                clusters.append({
                    "cluster_id": int(cluster_id),
                    "traits": cluster_traits,
                    "size": len(cluster_traits)
                })
        
        return clusters
    
    def _detect_significant_changes(self, trait_evolution: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Detect significant changes in trait evolution."""
        significant_changes = []
        
        for trait, values in trait_evolution.items():
            if len(values) < 10:
                continue
            
            # Calculate change points using rolling statistics
            window_size = min(5, len(values) // 2)
            
            for i in range(window_size, len(values)):
                window = values[i-window_size:i+1]
                current_value = values[i]
                
                # Check if current value is significantly different from window
                window_mean = np.mean(window)
                window_std = np.std(window)
                
                if abs(current_value - window_mean) > 2 * window_std:
                    significant_changes.append({
                        "trait": trait,
                        "epoch": i,
                        "change_magnitude": abs(current_value - window_mean),
                        "previous_mean": window_mean,
                        "current_value": current_value
                    })
        
        return significant_changes
    
    def _create_evolution_summary(self, trait_evolution: Dict[str, List[float]]) -> Dict[str, Any]:
        """Create summary of personality evolution."""
        summary = {
            "total_traits": len(trait_evolution),
            "evolution_length": len(next(iter(trait_evolution.values()))) if trait_evolution else 0,
            "trait_statistics": {}
        }
        
        for trait, values in trait_evolution.items():
            summary["trait_statistics"][trait] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "trend": "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
            }
        
        return summary
    
    def _compare_two_personas(
        self, 
        persona1_states: List[Dict[str, float]], 
        persona2_states: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Compare two personas."""
        # Align states by length
        min_length = min(len(persona1_states), len(persona2_states))
        persona1_aligned = persona1_states[:min_length]
        persona2_aligned = persona2_states[:min_length]
        
        # Calculate similarity metrics
        similarities = []
        for i in range(min_length):
            state1 = persona1_aligned[i]
            state2 = persona2_aligned[i]
            
            # Calculate cosine similarity
            emotions1 = list(state1.values())
            emotions2 = list(state2.values())
            
            if len(emotions1) == len(emotions2) and len(emotions1) > 0:
                similarity = np.dot(emotions1, emotions2) / (np.linalg.norm(emotions1) * np.linalg.norm(emotions2))
                similarities.append(similarity)
        
        return {
            "average_similarity": np.mean(similarities) if similarities else 0,
            "similarity_std": np.std(similarities) if similarities else 0,
            "min_similarity": np.min(similarities) if similarities else 0,
            "max_similarity": np.max(similarities) if similarities else 0
        }
    
    def _calculate_similarity_matrix(self, persona_data: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """Calculate similarity matrix between all personas."""
        persona_ids = list(persona_data.keys())
        similarity_matrix = {}
        
        for persona1 in persona_ids:
            similarity_matrix[persona1] = {}
            for persona2 in persona_ids:
                if persona1 == persona2:
                    similarity_matrix[persona1][persona2] = 1.0
                else:
                    comparison = self._compare_two_personas(
                        persona_data[persona1], 
                        persona_data[persona2]
                    )
                    similarity_matrix[persona1][persona2] = comparison["average_similarity"]
        
        return similarity_matrix
    
    def _detect_divergence_points(self, persona_data: Dict[str, List[Dict[str, float]]]) -> List[Dict[str, Any]]:
        """Detect points where personas diverge."""
        # Implementation for divergence detection
        return []
    
    def _detect_convergence_points(self, persona_data: Dict[str, List[Dict[str, float]]]) -> List[Dict[str, Any]]:
        """Detect points where personas converge."""
        # Implementation for convergence detection
        return []
    
    def _export_to_csv(self, analysis_results: Dict[str, Any]) -> str:
        """Export analysis results to CSV format."""
        # Implementation for CSV export
        return "CSV export not implemented"
    
    def _export_to_excel(self, analysis_results: Dict[str, Any]) -> bytes:
        """Export analysis results to Excel format."""
        # Implementation for Excel export
        return b"Excel export not implemented"
    
    def _create_empty_stability_analysis(self) -> StabilityAnalysis:
        """Create empty stability analysis."""
        return StabilityAnalysis(
            overall_stability=0.0,
            emotional_volatility=0.0,
            breakdown_risk=0.0,
            resilience_score=0.0,
            adaptation_rate=0.0,
            stability_trends=[],
            critical_points=[]
        )
    
    def _create_empty_personality_evolution(self) -> PersonalityEvolution:
        """Create empty personality evolution."""
        return PersonalityEvolution(
            trait_evolution={},
            trait_correlations={},
            evolution_clusters=[],
            significant_changes=[],
            evolution_summary={}
        ) 