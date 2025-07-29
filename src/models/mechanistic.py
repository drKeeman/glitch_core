"""
Mechanistic analysis models for neural circuit monitoring.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class AnalysisType(str, Enum):
    """Types of mechanistic analysis."""
    ATTENTION = "attention"
    ACTIVATION = "activation"
    DRIFT = "drift"
    INTERVENTION = "intervention"


class AttentionCapture(BaseModel):
    """Attention pattern capture during inference."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Capture metadata
    capture_id: str = Field(..., description="Unique capture identifier")
    persona_id: str = Field(..., description="Persona being analyzed")
    simulation_day: int = Field(..., ge=0, description="Simulation day")
    simulation_hour: int = Field(..., ge=0, le=23, description="Simulation hour")
    
    # Input context
    input_tokens: List[str] = Field(..., description="Input token sequence")
    prompt_context: str = Field(..., description="Full prompt context")
    assessment_type: Optional[str] = Field(None, description="Assessment type if applicable")
    
    # Attention data
    attention_weights: List[List[float]] = Field(..., description="Attention weight matrix")
    layer_attention: Dict[int, List[List[float]]] = Field(default_factory=dict, description="Per-layer attention patterns")
    head_attention: Dict[str, List[List[float]]] = Field(default_factory=dict, description="Per-head attention patterns")
    
    # Analysis metadata
    token_count: int = Field(..., ge=1, description="Number of tokens analyzed")
    layer_count: int = Field(..., ge=1, description="Number of layers analyzed")
    head_count: int = Field(..., ge=1, description="Number of attention heads")
    
    # Salience metrics
    self_reference_attention: float = Field(default=0.0, ge=0.0, le=1.0, description="Self-reference attention score")
    emotional_salience: float = Field(default=0.0, ge=0.0, le=1.0, description="Emotional content salience")
    memory_integration: float = Field(default=0.0, ge=0.0, le=1.0, description="Memory integration score")
    
    # Timing
    capture_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Capture timestamp")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Processing time in milliseconds")
    
    def get_attention_summary(self) -> Dict[str, Any]:
        """Get attention pattern summary."""
        return {
            "total_attention": sum(sum(row) for row in self.attention_weights),
            "max_attention": max(max(row) for row in self.attention_weights),
            "attention_entropy": self._calculate_entropy(),
            "self_reference": self.self_reference_attention,
            "emotional_salience": self.emotional_salience,
            "memory_integration": self.memory_integration,
        }
    
    def _calculate_entropy(self) -> float:
        """Calculate attention pattern entropy."""
        import math
        total_attention = sum(sum(row) for row in self.attention_weights)
        if total_attention == 0:
            return 0.0
        
        entropy = 0.0
        for row in self.attention_weights:
            for weight in row:
                if weight > 0:
                    p = weight / total_attention
                    entropy -= p * math.log2(p)
        
        return entropy
    
    def get_layer_attention(self, layer_idx: int) -> Optional[List[List[float]]]:
        """Get attention pattern for specific layer."""
        return self.layer_attention.get(layer_idx)
    
    def get_head_attention(self, head_name: str) -> Optional[List[List[float]]]:
        """Get attention pattern for specific head."""
        return self.head_attention.get(head_name)
    
    def is_high_self_reference(self) -> bool:
        """Check if attention shows high self-reference."""
        return self.self_reference_attention >= 0.7
    
    def is_emotionally_salient(self) -> bool:
        """Check if attention shows emotional salience."""
        return self.emotional_salience >= 0.6


class ActivationCapture(BaseModel):
    """Neural activation capture during inference."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Capture metadata
    capture_id: str = Field(..., description="Unique capture identifier")
    persona_id: str = Field(..., description="Persona being analyzed")
    simulation_day: int = Field(..., ge=0, description="Simulation day")
    simulation_hour: int = Field(..., ge=0, le=23, description="Simulation hour")
    
    # Activation data
    layer_activations: Dict[int, List[float]] = Field(..., description="Per-layer activation values")
    neuron_activations: Dict[str, List[float]] = Field(default_factory=dict, description="Per-neuron activation values")
    circuit_activations: Dict[str, List[float]] = Field(default_factory=dict, description="Circuit-level activations")
    
    # Analysis metadata
    layer_count: int = Field(..., ge=1, description="Number of layers analyzed")
    neuron_count: int = Field(..., ge=1, description="Number of neurons analyzed")
    circuit_count: int = Field(..., ge=1, description="Number of circuits analyzed")
    
    # Activation metrics
    activation_magnitude: float = Field(default=0.0, ge=0.0, description="Overall activation magnitude")
    activation_sparsity: float = Field(default=0.0, ge=0.0, le=1.0, description="Activation sparsity")
    circuit_specialization: float = Field(default=0.0, ge=0.0, le=1.0, description="Circuit specialization score")
    
    # Timing
    capture_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Capture timestamp")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Processing time in milliseconds")
    
    def get_activation_summary(self) -> Dict[str, Any]:
        """Get activation pattern summary."""
        return {
            "magnitude": self.activation_magnitude,
            "sparsity": self.activation_sparsity,
            "specialization": self.circuit_specialization,
            "layer_count": self.layer_count,
            "neuron_count": self.neuron_count,
            "circuit_count": self.circuit_count,
        }
    
    def get_layer_activation(self, layer_idx: int) -> Optional[List[float]]:
        """Get activation values for specific layer."""
        return self.layer_activations.get(layer_idx)
    
    def get_circuit_activation(self, circuit_name: str) -> Optional[List[float]]:
        """Get activation values for specific circuit."""
        return self.circuit_activations.get(circuit_name)
    
    def is_highly_activated(self) -> bool:
        """Check if overall activation is high."""
        return self.activation_magnitude >= 0.7
    
    def is_sparse_activation(self) -> bool:
        """Check if activation pattern is sparse."""
        return self.activation_sparsity >= 0.8


class DriftDetection(BaseModel):
    """Personality drift detection results."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Detection metadata
    detection_id: str = Field(..., description="Unique detection identifier")
    persona_id: str = Field(..., description="Persona being analyzed")
    baseline_day: int = Field(..., ge=0, description="Baseline measurement day")
    current_day: int = Field(..., ge=0, description="Current measurement day")
    
    # Drift measurements
    trait_drift: Dict[str, float] = Field(..., description="Personality trait drift values")
    clinical_drift: Dict[str, float] = Field(..., description="Clinical measure drift values")
    mechanistic_drift: Dict[str, float] = Field(..., description="Mechanistic measure drift values")
    
    # Detection thresholds
    drift_threshold: float = Field(default=0.1, ge=0.0, description="Drift detection threshold")
    significance_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Statistical significance threshold")
    
    # Analysis results
    drift_detected: bool = Field(default=False, description="Whether drift was detected")
    significant_drift: bool = Field(default=False, description="Whether drift is statistically significant")
    drift_magnitude: float = Field(default=0.0, ge=0.0, description="Overall drift magnitude")
    drift_direction: str = Field(default="neutral", description="Primary drift direction")
    
    # Affected components
    affected_traits: List[str] = Field(default_factory=list, description="Traits showing significant drift")
    affected_circuits: List[str] = Field(default_factory=list, description="Neural circuits showing drift")
    clinical_implications: List[str] = Field(default_factory=list, description="Clinical implications of drift")
    
    # Timing
    detection_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
    
    def calculate_drift_magnitude(self) -> float:
        """Calculate overall drift magnitude."""
        all_drift_values = []
        all_drift_values.extend(self.trait_drift.values())
        all_drift_values.extend(self.clinical_drift.values())
        all_drift_values.extend(self.mechanistic_drift.values())
        
        if not all_drift_values:
            return 0.0
        
        return sum(abs(value) for value in all_drift_values) / len(all_drift_values)
    
    def detect_significant_drift(self) -> bool:
        """Detect if any drift exceeds significance threshold."""
        for drift_values in [self.trait_drift, self.clinical_drift, self.mechanistic_drift]:
            for value in drift_values.values():
                if abs(value) >= self.significance_threshold:
                    return True
        return False
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get comprehensive drift summary."""
        return {
            "drift_detected": self.drift_detected,
            "significant_drift": self.significant_drift,
            "drift_magnitude": self.drift_magnitude,
            "drift_direction": self.drift_direction,
            "affected_traits": self.affected_traits,
            "affected_circuits": self.affected_circuits,
            "clinical_implications": self.clinical_implications,
            "days_since_baseline": self.current_day - self.baseline_day,
        }
    
    def is_clinically_significant(self) -> bool:
        """Check if drift has clinical significance."""
        return self.significant_drift and len(self.clinical_implications) > 0
    
    def requires_intervention(self) -> bool:
        """Check if drift requires intervention."""
        return self.drift_magnitude >= 0.3 or len(self.clinical_implications) >= 2


class MechanisticAnalysis(BaseModel):
    """Complete mechanistic analysis result."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Analysis metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    persona_id: str = Field(..., description="Persona being analyzed")
    simulation_day: int = Field(..., ge=0, description="Simulation day")
    analysis_type: AnalysisType = Field(..., description="Type of analysis")
    
    # Analysis components
    attention_capture: Optional[AttentionCapture] = Field(None, description="Attention pattern capture")
    activation_capture: Optional[ActivationCapture] = Field(None, description="Activation pattern capture")
    drift_detection: Optional[DriftDetection] = Field(None, description="Drift detection results")
    
    # Analysis metadata
    input_context: str = Field(..., description="Input context for analysis")
    output_response: str = Field(..., description="Output response from persona")
    analysis_duration_ms: float = Field(default=0.0, ge=0.0, description="Analysis duration in milliseconds")
    
    # Quality metrics
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Data quality score")
    analysis_completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="Analysis completeness score")
    
    # Timing
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary."""
        summary = {
            "analysis_id": self.analysis_id,
            "persona_id": self.persona_id,
            "simulation_day": self.simulation_day,
            "analysis_type": self.analysis_type.value,
            "data_quality": self.data_quality_score,
            "completeness": self.analysis_completeness,
            "duration_ms": self.analysis_duration_ms,
        }
        
        if self.attention_capture:
            summary["attention"] = self.attention_capture.get_attention_summary()
        
        if self.activation_capture:
            summary["activation"] = self.activation_capture.get_activation_summary()
        
        if self.drift_detection:
            summary["drift"] = self.drift_detection.get_drift_summary()
        
        return summary
    
    def is_complete(self) -> bool:
        """Check if analysis is complete."""
        return self.analysis_completeness >= 0.9
    
    def is_high_quality(self) -> bool:
        """Check if analysis data is high quality."""
        return self.data_quality_score >= 0.8
    
    def has_significant_findings(self) -> bool:
        """Check if analysis has significant findings."""
        if self.drift_detection and self.drift_detection.significant_drift:
            return True
        if self.attention_capture and self.attention_capture.is_high_self_reference():
            return True
        if self.activation_capture and self.activation_capture.is_highly_activated():
            return True
        return False 