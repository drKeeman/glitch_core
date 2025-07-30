"""
Neural circuit tracking and monitoring.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import logging

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.models.mechanistic import (
    MechanisticAnalysis,
    AttentionCapture,
    ActivationCapture
)
from src.models.persona import Persona


logger = logging.getLogger(__name__)


class NeuralCircuit:
    """Represents a neural circuit with activation patterns."""
    
    def __init__(self, circuit_id: str, name: str, description: str):
        """Initialize neural circuit."""
        self.circuit_id = circuit_id
        self.name = name
        self.description = description
        self.activation_history: List[float] = []
        self.attention_history: List[float] = []
        self.specialization_score = 0.0
        self.stability_score = 0.0
        
        # Circuit characteristics
        self.layer_indices: Set[int] = set()
        self.head_indices: Set[int] = set()
        self.neuron_indices: Set[int] = set()
        
        # Performance tracking
        self.total_activations = 0
        self.activation_magnitude = 0.0
    
    def add_activation(self, activation_value: float, timestamp: datetime):
        """Add activation measurement to circuit."""
        self.activation_history.append(activation_value)
        self.total_activations += 1
        self.activation_magnitude = np.mean(self.activation_history)
        
        # Keep only recent history (last 100 measurements)
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]
    
    def add_attention(self, attention_value: float, timestamp: datetime):
        """Add attention measurement to circuit."""
        self.attention_history.append(attention_value)
        
        # Keep only recent history (last 100 measurements)
        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-100:]
    
    def calculate_specialization(self) -> float:
        """Calculate circuit specialization score."""
        if len(self.activation_history) < 5:
            return 0.0
        
        # Specialization based on variance and consistency
        variance = np.var(self.activation_history)
        consistency = 1.0 - np.std(self.activation_history) / (np.mean(self.activation_history) + 1e-8)
        
        self.specialization_score = (variance * consistency)
        return self.specialization_score
    
    def calculate_stability(self) -> float:
        """Calculate circuit stability score."""
        if len(self.activation_history) < 5:
            return 0.0
        
        # Stability based on low variance and consistent patterns
        variance = np.var(self.activation_history)
        stability = 1.0 / (1.0 + variance)
        
        self.stability_score = stability
        return self.stability_score
    
    def get_activation_trend(self, window: int = 10) -> float:
        """Get activation trend over recent window."""
        if len(self.activation_history) < window:
            return 0.0
        
        recent_activations = self.activation_history[-window:]
        if len(recent_activations) < 2:
            return 0.0
        
        # Linear trend
        x = np.arange(len(recent_activations))
        slope = np.polyfit(x, recent_activations, 1)[0]
        return slope
    
    def is_highly_active(self) -> bool:
        """Check if circuit is highly active."""
        return self.activation_magnitude > 0.7
    
    def is_stable(self) -> bool:
        """Check if circuit is stable."""
        return self.stability_score > 0.8
    
    def get_summary(self) -> Dict[str, Any]:
        """Get circuit summary."""
        return {
            "circuit_id": self.circuit_id,
            "name": self.name,
            "description": self.description,
            "activation_magnitude": self.activation_magnitude,
            "specialization_score": self.specialization_score,
            "stability_score": self.stability_score,
            "total_activations": self.total_activations,
            "recent_trend": self.get_activation_trend(),
            "is_highly_active": self.is_highly_active(),
            "is_stable": self.is_stable(),
            "layer_count": len(self.layer_indices),
            "head_count": len(self.head_indices),
            "neuron_count": len(self.neuron_indices)
        }


class CircuitTracker:
    """Track and monitor neural circuits during persona interactions."""
    
    def __init__(self):
        """Initialize circuit tracker."""
        self.circuits: Dict[str, NeuralCircuit] = {}
        self.persona_circuits: Dict[str, Dict[str, NeuralCircuit]] = {}
        
        # Circuit identification
        self.auto_discover_circuits = True
        self.min_circuit_activation = 0.1
        self.circuit_discovery_threshold = 0.3
        
        # Performance tracking
        self.total_circuits_tracked = 0
        self.total_measurements = 0
        self.total_processing_time = 0.0
    
    async def setup_circuit_tracking(self, persona_id: str) -> bool:
        """Setup circuit tracking for a persona."""
        try:
            if persona_id not in self.persona_circuits:
                self.persona_circuits[persona_id] = {}
            
            # Initialize default circuits
            default_circuits = [
                ("self_reference", "Self-Reference Circuit", "Handles self-referential processing"),
                ("emotional", "Emotional Processing Circuit", "Processes emotional content and responses"),
                ("memory", "Memory Integration Circuit", "Integrates and retrieves memories"),
                ("attention", "Attention Control Circuit", "Controls attention allocation"),
                ("language", "Language Processing Circuit", "Handles language comprehension and generation"),
                ("reasoning", "Reasoning Circuit", "Handles logical reasoning and problem-solving")
            ]
            
            for circuit_id, name, description in default_circuits:
                circuit = NeuralCircuit(circuit_id, name, description)
                self.circuits[circuit_id] = circuit
                self.persona_circuits[persona_id][circuit_id] = circuit
            
            self.total_circuits_tracked += len(default_circuits)
            logger.info(f"Setup circuit tracking for {persona_id} with {len(default_circuits)} circuits")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up circuit tracking for {persona_id}: {e}")
            return False
    
    async def track_circuit_activations(
        self,
        persona_id: str,
        mechanistic_analysis: MechanisticAnalysis
    ) -> Dict[str, float]:
        """Track circuit activations from mechanistic analysis."""
        start_time = time.time()
        
        try:
            circuit_activations = {}
            
            if persona_id not in self.persona_circuits:
                await self.setup_circuit_tracking(persona_id)
            
            # Process attention-based circuits
            if mechanistic_analysis.attention_capture:
                attention_activations = self._extract_attention_circuits(
                    mechanistic_analysis.attention_capture
                )
                circuit_activations.update(attention_activations)
            
            # Process activation-based circuits
            if mechanistic_analysis.activation_capture:
                activation_circuits = self._extract_activation_circuits(
                    mechanistic_analysis.activation_capture
                )
                circuit_activations.update(activation_circuits)
            
            # Update circuit measurements
            timestamp = mechanistic_analysis.analysis_timestamp
            for circuit_id, activation in circuit_activations.items():
                if circuit_id in self.persona_circuits[persona_id]:
                    circuit = self.persona_circuits[persona_id][circuit_id]
                    circuit.add_activation(activation, timestamp)
                    
                    # Update specialization and stability
                    circuit.calculate_specialization()
                    circuit.calculate_stability()
            
            # Auto-discover new circuits if enabled
            if self.auto_discover_circuits:
                await self._discover_new_circuits(persona_id, mechanistic_analysis)
            
            # Update performance metrics
            self.total_measurements += 1
            self.total_processing_time += time.time() - start_time
            
            logger.debug(f"Tracked {len(circuit_activations)} circuit activations for {persona_id}")
            return circuit_activations
            
        except Exception as e:
            logger.error(f"Error tracking circuit activations for {persona_id}: {e}")
            return {}
    
    def _extract_attention_circuits(self, attention_capture: AttentionCapture) -> Dict[str, float]:
        """Extract circuit activations from attention patterns."""
        circuits = {}
        
        # Self-reference circuit
        circuits["self_reference"] = attention_capture.self_reference_attention
        
        # Emotional processing circuit
        circuits["emotional"] = attention_capture.emotional_salience
        
        # Memory integration circuit
        circuits["memory"] = attention_capture.memory_integration
        
        # Attention control circuit (based on attention entropy)
        attention_summary = attention_capture.get_attention_summary()
        circuits["attention"] = 1.0 - attention_summary["attention_entropy"] / 10.0  # Normalize
        
        return circuits
    
    def _extract_activation_circuits(self, activation_capture: ActivationCapture) -> Dict[str, float]:
        """Extract circuit activations from neural activations."""
        circuits = {}
        
        # Language processing circuit (based on activation magnitude)
        circuits["language"] = activation_capture.activation_magnitude
        
        # Reasoning circuit (based on circuit specialization)
        circuits["reasoning"] = activation_capture.circuit_specialization
        
        # Memory circuit (based on activation sparsity - sparse activations suggest memory retrieval)
        circuits["memory"] = activation_capture.activation_sparsity
        
        return circuits
    
    async def _discover_new_circuits(self, persona_id: str, mechanistic_analysis: MechanisticAnalysis):
        """Auto-discover new circuits based on activation patterns."""
        try:
            if not mechanistic_analysis.activation_capture:
                return
            
            # Analyze layer activations for patterns
            layer_activations = mechanistic_analysis.activation_capture.layer_activations
            
            if len(layer_activations) < 3:
                return
            
            # Convert to numpy array for analysis
            activation_matrix = np.array(list(layer_activations.values()))
            
            # Use PCA to identify principal components
            if activation_matrix.shape[1] > 1:
                pca = PCA(n_components=min(3, activation_matrix.shape[1]))
                principal_components = pca.fit_transform(activation_matrix.T)
                
                # Check for significant components that might represent new circuits
                explained_variance = pca.explained_variance_ratio_
                
                for i, variance in enumerate(explained_variance):
                    if variance > self.circuit_discovery_threshold:
                        # This component explains significant variance - potential new circuit
                        circuit_id = f"discovered_circuit_{i}_{int(time.time())}"
                        circuit_name = f"Discovered Circuit {i+1}"
                        circuit_desc = f"Auto-discovered circuit explaining {variance:.2%} of variance"
                        
                        circuit = NeuralCircuit(circuit_id, circuit_name, circuit_desc)
                        self.circuits[circuit_id] = circuit
                        self.persona_circuits[persona_id][circuit_id] = circuit
                        
                        logger.info(f"Discovered new circuit: {circuit_name} for {persona_id}")
            
        except Exception as e:
            logger.error(f"Error discovering new circuits: {e}")
    
    async def get_circuit_summary(self, persona_id: str) -> Dict[str, Any]:
        """Get summary of all circuits for a persona."""
        if persona_id not in self.persona_circuits:
            return {}
        
        circuits = self.persona_circuits[persona_id]
        summaries = {}
        
        for circuit_id, circuit in circuits.items():
            summaries[circuit_id] = circuit.get_summary()
        
        return summaries
    
    async def get_highly_active_circuits(self, persona_id: str) -> List[str]:
        """Get list of highly active circuits for a persona."""
        if persona_id not in self.persona_circuits:
            return []
        
        active_circuits = []
        for circuit_id, circuit in self.persona_circuits[persona_id].items():
            if circuit.is_highly_active():
                active_circuits.append(circuit_id)
        
        return active_circuits
    
    async def get_unstable_circuits(self, persona_id: str) -> List[str]:
        """Get list of unstable circuits for a persona."""
        if persona_id not in self.persona_circuits:
            return []
        
        unstable_circuits = []
        for circuit_id, circuit in self.persona_circuits[persona_id].items():
            if not circuit.is_stable():
                unstable_circuits.append(circuit_id)
        
        return unstable_circuits
    
    async def get_circuit_trends(self, persona_id: str, window: int = 10) -> Dict[str, float]:
        """Get activation trends for all circuits."""
        if persona_id not in self.persona_circuits:
            return {}
        
        trends = {}
        for circuit_id, circuit in self.persona_circuits[persona_id].items():
            trends[circuit_id] = circuit.get_activation_trend(window)
        
        return trends
    
    async def identify_circuit_anomalies(self, persona_id: str) -> Dict[str, List[str]]:
        """Identify anomalous circuit behavior."""
        if persona_id not in self.persona_circuits:
            return {}
        
        anomalies = {
            "highly_active": [],
            "unstable": [],
            "trending_up": [],
            "trending_down": []
        }
        
        for circuit_id, circuit in self.persona_circuits[persona_id].items():
            if circuit.is_highly_active():
                anomalies["highly_active"].append(circuit_id)
            
            if not circuit.is_stable():
                anomalies["unstable"].append(circuit_id)
            
            trend = circuit.get_activation_trend()
            if trend > 0.1:
                anomalies["trending_up"].append(circuit_id)
            elif trend < -0.1:
                anomalies["trending_down"].append(circuit_id)
        
        return anomalies
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_circuits_tracked": self.total_circuits_tracked,
            "total_measurements": self.total_measurements,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / max(self.total_measurements, 1),
            "personas_tracked": len(self.persona_circuits),
            "auto_discovery_enabled": self.auto_discover_circuits
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.circuits.clear()
        self.persona_circuits.clear()
        logger.info("Circuit tracker cleaned up") 