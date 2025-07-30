"""
Personality drift detection using mechanistic analysis.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.models.mechanistic import (
    MechanisticAnalysis, 
    DriftDetection,
    AttentionCapture,
    ActivationCapture
)
from src.models.persona import Persona
from src.models.assessment import AssessmentResult


logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect personality drift using mechanistic analysis."""
    
    def __init__(self):
        """Initialize drift detector."""
        self.baseline_data: Dict[str, Dict[str, Any]] = {}
        self.drift_history: Dict[str, List[DriftDetection]] = {}
        self.detection_threshold = 0.1
        self.significance_threshold = 0.05
        
        # Drift detection configuration
        self.min_baseline_samples = 5
        self.drift_window_days = 7
        self.early_warning_threshold = 0.05
        
        # Performance tracking
        self.total_detections = 0
        self.total_processing_time = 0.0
    
    async def establish_baseline(
        self, 
        persona_id: str, 
        mechanistic_analyses: List[MechanisticAnalysis],
        assessment_results: List[AssessmentResult]
    ) -> bool:
        """Establish baseline for drift detection."""
        try:
            if len(mechanistic_analyses) < self.min_baseline_samples:
                logger.warning(f"Insufficient baseline samples for {persona_id}")
                return False
            
            # Extract baseline metrics
            baseline_metrics = {
                "attention_patterns": [],
                "activation_patterns": [],
                "clinical_scores": [],
                "trait_scores": []
            }
            
            # Process mechanistic analyses
            for analysis in mechanistic_analyses:
                if analysis.attention_capture:
                    baseline_metrics["attention_patterns"].append({
                        "self_reference": analysis.attention_capture.self_reference_attention,
                        "emotional_salience": analysis.attention_capture.emotional_salience,
                        "memory_integration": analysis.attention_capture.memory_integration,
                        "attention_entropy": analysis.attention_capture.get_attention_summary()["attention_entropy"]
                    })
                
                if analysis.activation_capture:
                    baseline_metrics["activation_patterns"].append({
                        "magnitude": analysis.activation_capture.activation_magnitude,
                        "sparsity": analysis.activation_capture.activation_sparsity,
                        "specialization": analysis.activation_capture.circuit_specialization
                    })
            
            # Process assessment results
            for assessment in assessment_results:
                if hasattr(assessment, 'phq9_score') and assessment.phq9_score is not None:
                    baseline_metrics["clinical_scores"].append({
                        "phq9": assessment.phq9_score,
                        "gad7": getattr(assessment, 'gad7_score', None),
                        "pss10": getattr(assessment, 'pss10_score', None)
                    })
            
            # Calculate baseline statistics
            baseline_stats = self._calculate_baseline_statistics(baseline_metrics)
            
            # Store baseline
            self.baseline_data[persona_id] = {
                "metrics": baseline_metrics,
                "statistics": baseline_stats,
                "established_at": datetime.utcnow(),
                "sample_count": len(mechanistic_analyses)
            }
            
            logger.info(f"Established baseline for {persona_id} with {len(mechanistic_analyses)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error establishing baseline for {persona_id}: {e}")
            return False
    
    def _calculate_baseline_statistics(self, baseline_metrics: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate baseline statistics for drift detection."""
        stats_dict = {}
        
        # Attention pattern statistics
        if baseline_metrics["attention_patterns"]:
            attention_data = np.array([[p["self_reference"], p["emotional_salience"], 
                                     p["memory_integration"], p["attention_entropy"]] 
                                    for p in baseline_metrics["attention_patterns"]])
            
            stats_dict["attention"] = {
                "mean": np.mean(attention_data, axis=0).tolist(),
                "std": np.std(attention_data, axis=0).tolist(),
                "median": np.median(attention_data, axis=0).tolist(),
                "range": (np.max(attention_data, axis=0) - np.min(attention_data, axis=0)).tolist()
            }
        
        # Activation pattern statistics
        if baseline_metrics["activation_patterns"]:
            activation_data = np.array([[p["magnitude"], p["sparsity"], p["specialization"]] 
                                      for p in baseline_metrics["activation_patterns"]])
            
            stats_dict["activation"] = {
                "mean": np.mean(activation_data, axis=0).tolist(),
                "std": np.std(activation_data, axis=0).tolist(),
                "median": np.median(activation_data, axis=0).tolist(),
                "range": (np.max(activation_data, axis=0) - np.min(activation_data, axis=0)).tolist()
            }
        
        # Clinical score statistics
        if baseline_metrics["clinical_scores"]:
            clinical_data = []
            for score in baseline_metrics["clinical_scores"]:
                if score["phq9"] is not None:
                    clinical_data.append([score["phq9"], 
                                       score["gad7"] or 0, 
                                       score["pss10"] or 0])
            
            if clinical_data:
                clinical_array = np.array(clinical_data)
                stats_dict["clinical"] = {
                    "mean": np.mean(clinical_array, axis=0).tolist(),
                    "std": np.std(clinical_array, axis=0).tolist(),
                    "median": np.median(clinical_array, axis=0).tolist(),
                    "range": (np.max(clinical_array, axis=0) - np.min(clinical_array, axis=0)).tolist()
                }
        
        return stats_dict
    
    async def detect_drift(
        self,
        persona_id: str,
        current_analysis: MechanisticAnalysis,
        current_assessment: Optional[AssessmentResult] = None,
        simulation_day: int = 0
    ) -> Optional[DriftDetection]:
        """Detect personality drift from current data."""
        if persona_id not in self.baseline_data:
            logger.warning(f"No baseline established for {persona_id}")
            return None
        
        start_time = time.time()
        
        try:
            baseline = self.baseline_data[persona_id]
            baseline_stats = baseline["statistics"]
            
            # Calculate drift measurements
            trait_drift = {}
            clinical_drift = {}
            mechanistic_drift = {}
            
            # Mechanistic drift calculation
            if current_analysis.attention_capture and "attention" in baseline_stats:
                attention_drift = self._calculate_attention_drift(
                    current_analysis.attention_capture, baseline_stats["attention"]
                )
                mechanistic_drift.update(attention_drift)
            
            if current_analysis.activation_capture and "activation" in baseline_stats:
                activation_drift = self._calculate_activation_drift(
                    current_analysis.activation_capture, baseline_stats["activation"]
                )
                mechanistic_drift.update(activation_drift)
            
            # Clinical drift calculation
            if current_assessment and "clinical" in baseline_stats:
                clinical_drift = self._calculate_clinical_drift(
                    current_assessment, baseline_stats["clinical"]
                )
            
            # Determine drift significance
            drift_detected = any(abs(value) >= self.detection_threshold 
                               for value in list(trait_drift.values()) + 
                               list(clinical_drift.values()) + 
                               list(mechanistic_drift.values()))
            
            significant_drift = any(abs(value) >= self.significance_threshold 
                                 for value in list(trait_drift.values()) + 
                                 list(clinical_drift.values()) + 
                                 list(mechanistic_drift.values()))
            
            # Calculate overall drift magnitude
            all_drift_values = list(trait_drift.values()) + list(clinical_drift.values()) + list(mechanistic_drift.values())
            drift_magnitude = np.mean([abs(v) for v in all_drift_values]) if all_drift_values else 0.0
            
            # Determine drift direction
            drift_direction = self._determine_drift_direction(trait_drift, clinical_drift, mechanistic_drift)
            
            # Identify affected components
            affected_traits = [k for k, v in trait_drift.items() if abs(v) >= self.detection_threshold]
            affected_circuits = [k for k, v in mechanistic_drift.items() if abs(v) >= self.detection_threshold]
            clinical_implications = self._identify_clinical_implications(clinical_drift, mechanistic_drift)
            
            # Create drift detection result
            detection = DriftDetection(
                detection_id=str(uuid.uuid4()),
                persona_id=persona_id,
                baseline_day=0,  # Baseline established at day 0
                current_day=simulation_day,
                trait_drift=trait_drift,
                clinical_drift=clinical_drift,
                mechanistic_drift=mechanistic_drift,
                drift_threshold=self.detection_threshold,
                significance_threshold=self.significance_threshold,
                drift_detected=drift_detected,
                significant_drift=significant_drift,
                drift_magnitude=drift_magnitude,
                drift_direction=drift_direction,
                affected_traits=affected_traits,
                affected_circuits=affected_circuits,
                clinical_implications=clinical_implications
            )
            
            # Store in history
            if persona_id not in self.drift_history:
                self.drift_history[persona_id] = []
            self.drift_history[persona_id].append(detection)
            
            # Update performance metrics
            self.total_detections += 1
            self.total_processing_time += time.time() - start_time
            
            logger.debug(f"Drift detection completed for {persona_id} in {time.time() - start_time:.3f}s")
            return detection
            
        except Exception as e:
            logger.error(f"Error detecting drift for {persona_id}: {e}")
            return None
    
    def _calculate_attention_drift(
        self, 
        attention_capture: AttentionCapture, 
        baseline_stats: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate attention pattern drift."""
        drift = {}
        
        current_values = [
            attention_capture.self_reference_attention,
            attention_capture.emotional_salience,
            attention_capture.memory_integration,
            attention_capture.get_attention_summary()["attention_entropy"]
        ]
        
        baseline_means = baseline_stats["mean"]
        baseline_stds = baseline_stats["std"]
        
        for i, (current, baseline_mean, baseline_std) in enumerate(
            zip(current_values, baseline_means, baseline_stds)
        ):
            if baseline_std > 0:
                # Z-score based drift
                drift_score = (current - baseline_mean) / baseline_std
            else:
                # Simple difference if no variance in baseline
                drift_score = current - baseline_mean
            
            drift[f"attention_{i}"] = drift_score
        
        return drift
    
    def _calculate_activation_drift(
        self, 
        activation_capture: ActivationCapture, 
        baseline_stats: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate activation pattern drift."""
        drift = {}
        
        current_values = [
            activation_capture.activation_magnitude,
            activation_capture.activation_sparsity,
            activation_capture.circuit_specialization
        ]
        
        baseline_means = baseline_stats["mean"]
        baseline_stds = baseline_stats["std"]
        
        for i, (current, baseline_mean, baseline_std) in enumerate(
            zip(current_values, baseline_means, baseline_stds)
        ):
            if baseline_std > 0:
                # Z-score based drift
                drift_score = (current - baseline_mean) / baseline_std
            else:
                # Simple difference if no variance in baseline
                drift_score = current - baseline_mean
            
            drift[f"activation_{i}"] = drift_score
        
        return drift
    
    def _calculate_clinical_drift(
        self, 
        assessment: AssessmentResult, 
        baseline_stats: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate clinical score drift."""
        drift = {}
        
        current_values = [
            assessment.phq9_score or 0,
            getattr(assessment, 'gad7_score', None) or 0,
            getattr(assessment, 'pss10_score', None) or 0
        ]
        
        baseline_means = baseline_stats["mean"]
        baseline_stds = baseline_stats["std"]
        
        clinical_names = ["phq9", "gad7", "pss10"]
        
        for i, (current, baseline_mean, baseline_std, name) in enumerate(
            zip(current_values, baseline_means, baseline_stds, clinical_names)
        ):
            if baseline_std > 0:
                # Z-score based drift
                drift_score = (current - baseline_mean) / baseline_std
            else:
                # Simple difference if no variance in baseline
                drift_score = current - baseline_mean
            
            drift[name] = drift_score
        
        return drift
    
    def _determine_drift_direction(
        self, 
        trait_drift: Dict[str, float], 
        clinical_drift: Dict[str, float], 
        mechanistic_drift: Dict[str, float]
    ) -> str:
        """Determine primary drift direction."""
        all_drift_values = list(trait_drift.values()) + list(clinical_drift.values()) + list(mechanistic_drift.values())
        
        if not all_drift_values:
            return "neutral"
        
        # Calculate average drift
        avg_drift = np.mean(all_drift_values)
        
        if avg_drift > 0.1:
            return "positive"
        elif avg_drift < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _identify_clinical_implications(
        self, 
        clinical_drift: Dict[str, float], 
        mechanistic_drift: Dict[str, float]
    ) -> List[str]:
        """Identify clinical implications of drift."""
        implications = []
        
        # Check for clinical score changes
        for measure, drift in clinical_drift.items():
            if abs(drift) >= 0.5:  # Significant clinical change
                if measure == "phq9" and drift > 0:
                    implications.append("Increased depression symptoms")
                elif measure == "gad7" and drift > 0:
                    implications.append("Increased anxiety symptoms")
                elif measure == "pss10" and drift > 0:
                    implications.append("Increased stress levels")
        
        # Check for mechanistic changes that might indicate clinical issues
        if "attention_0" in mechanistic_drift and mechanistic_drift["attention_0"] > 0.5:
            implications.append("Increased self-reference attention")
        
        if "activation_0" in mechanistic_drift and mechanistic_drift["activation_0"] > 0.5:
            implications.append("Increased neural activation magnitude")
        
        return implications
    
    async def get_drift_history(self, persona_id: str, days: int = 30) -> List[DriftDetection]:
        """Get drift detection history for a persona."""
        if persona_id not in self.drift_history:
            return []
        
        # Filter by time window
        cutoff_day = max(0, days)
        return [detection for detection in self.drift_history[persona_id] 
                if detection.current_day >= cutoff_day]
    
    async def get_early_warnings(self, persona_id: str) -> List[DriftDetection]:
        """Get early warning drift detections."""
        if persona_id not in self.drift_history:
            return []
        
        warnings = []
        for detection in self.drift_history[persona_id]:
            if (detection.drift_magnitude >= self.early_warning_threshold and 
                not detection.significant_drift):
                warnings.append(detection)
        
        return warnings
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_detections": self.total_detections,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / max(self.total_detections, 1),
            "baselines_established": len(self.baseline_data),
            "detection_threshold": self.detection_threshold,
            "significance_threshold": self.significance_threshold
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.baseline_data.clear()
        self.drift_history.clear()
        logger.info("Drift detector cleaned up") 