"""
Core mechanistic analysis for attention and activation capture.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import logging

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from src.core.config import settings
from src.models.mechanistic import (
    AttentionCapture, 
    ActivationCapture, 
    MechanisticAnalysis,
    AnalysisType
)
from src.models.persona import Persona
from src.services.llm_service import LLMService


logger = logging.getLogger(__name__)


class AttentionHook:
    """Hook for capturing attention weights during inference."""
    
    def __init__(self, layer_idx: int, head_idx: Optional[int] = None):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.attention_weights: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []
        
    def __call__(self, module, input, output):
        """Capture attention weights and activations."""
        if hasattr(output, 'attentions') and output.attentions is not None:
            # Capture attention weights
            attention = output.attentions[self.layer_idx]
            if self.head_idx is not None:
                attention = attention[:, self.head_idx:self.head_idx+1]
            self.attention_weights.append(attention.detach().cpu())
        
        # Capture activations
        if hasattr(output, 'hidden_states'):
            hidden_state = output.hidden_states[self.layer_idx]
            self.activations.append(hidden_state.detach().cpu())


class MechanisticAnalyzer:
    """Core mechanistic analysis for neural circuit monitoring."""
    
    def __init__(self, llm_service: LLMService):
        """Initialize mechanistic analyzer."""
        self.llm_service = llm_service
        self.model = None
        self.tokenizer = None
        self.hooks: List[AttentionHook] = []
        self.is_analyzing = False
        
        # Analysis configuration
        self.capture_attention = True
        self.capture_activations = True
        self.capture_self_reference = True
        self.capture_emotional_salience = True
        
        # Performance tracking
        self.total_analyses = 0
        self.total_capture_time = 0.0
        
    async def setup_analysis(self) -> bool:
        """Setup mechanistic analysis hooks."""
        try:
            if not self.llm_service.is_loaded:
                logger.error("LLM service not loaded")
                return False
            
            self.model = self.llm_service.model
            self.tokenizer = self.llm_service.tokenizer
            
            # Register hooks for attention capture
            if self.capture_attention:
                await self._register_attention_hooks()
            
            logger.info("Mechanistic analysis setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup mechanistic analysis: {e}")
            return False
    
    async def _register_attention_hooks(self):
        """Register hooks for attention weight capture."""
        try:
            # Clear existing hooks
            self._remove_hooks()
            
            # Find transformer layers
            transformer_layers = []
            for name, module in self.model.named_modules():
                if "transformer" in name.lower() and "layer" in name.lower():
                    transformer_layers.append((name, module))
            
            # Register hooks for key layers (first, middle, last)
            layer_indices = [0, len(transformer_layers)//2, len(transformer_layers)-1]
            
            for layer_idx in layer_indices:
                if layer_idx < len(transformer_layers):
                    name, module = transformer_layers[layer_idx]
                    hook = AttentionHook(layer_idx)
                    module.register_forward_hook(hook)
                    self.hooks.append(hook)
                    logger.debug(f"Registered attention hook for layer {layer_idx}")
            
            logger.info(f"Registered {len(self.hooks)} attention hooks")
            
        except Exception as e:
            logger.error(f"Failed to register attention hooks: {e}")
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            if hasattr(hook, 'remove'):
                hook.remove()
        self.hooks.clear()
    
    async def capture_mechanistic_data(
        self, 
        persona: Persona, 
        input_context: str, 
        instruction: str,
        simulation_day: int,
        simulation_hour: int
    ) -> Optional[MechanisticAnalysis]:
        """Capture mechanistic data during persona response generation."""
        if not self.is_analyzing:
            logger.warning("Mechanistic analysis not enabled")
            return None
        
        start_time = time.time()
        analysis_id = str(uuid.uuid4())
        
        try:
            # Clear previous captures
            for hook in self.hooks:
                hook.attention_weights.clear()
                hook.activations.clear()
            
            # Generate response with mechanistic capture
            response, metrics = await self.llm_service.generate_response(
                persona, input_context, instruction
            )
            
            # Process captured data
            attention_capture = None
            activation_capture = None
            
            if self.capture_attention and self.hooks:
                attention_capture = await self._process_attention_capture(
                    analysis_id, persona, simulation_day, simulation_hour,
                    input_context, instruction, response
                )
            
            if self.capture_activations and self.hooks:
                activation_capture = await self._process_activation_capture(
                    analysis_id, persona, simulation_day, simulation_hour,
                    input_context, instruction, response
                )
            
            # Create analysis result
            analysis = MechanisticAnalysis(
                analysis_id=analysis_id,
                persona_id=persona.state.persona_id,
                simulation_day=simulation_day,
                analysis_type=AnalysisType.ATTENTION if attention_capture else AnalysisType.ACTIVATION,
                attention_capture=attention_capture,
                activation_capture=activation_capture,
                input_context=input_context,
                output_response=response,
                analysis_duration_ms=(time.time() - start_time) * 1000,
                data_quality_score=self._calculate_data_quality(attention_capture, activation_capture),
                analysis_completeness=self._calculate_completeness(attention_capture, activation_capture)
            )
            
            # Update performance metrics
            self.total_analyses += 1
            self.total_capture_time += time.time() - start_time
            
            logger.debug(f"Captured mechanistic data in {time.time() - start_time:.3f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Error capturing mechanistic data: {e}")
            return None
    
    async def _process_attention_capture(
        self,
        analysis_id: str,
        persona: Persona,
        simulation_day: int,
        simulation_hour: int,
        input_context: str,
        instruction: str,
        response: str
    ) -> Optional[AttentionCapture]:
        """Process captured attention data."""
        try:
            # Aggregate attention weights across layers
            all_attention_weights = []
            layer_attention = {}
            head_attention = {}
            
            for i, hook in enumerate(self.hooks):
                if hook.attention_weights:
                    # Average attention weights across batch
                    layer_weights = torch.cat(hook.attention_weights, dim=0).mean(dim=0)
                    layer_attention[i] = layer_weights.tolist()
                    all_attention_weights.append(layer_weights)
            
            if not all_attention_weights:
                return None
            
            # Calculate overall attention pattern
            overall_attention = torch.stack(all_attention_weights).mean(dim=0)
            attention_weights = overall_attention.tolist()
            
            # Calculate salience metrics
            self_reference_attention = self._calculate_self_reference_attention(attention_weights)
            emotional_salience = self._calculate_emotional_salience(attention_weights, input_context)
            memory_integration = self._calculate_memory_integration(attention_weights, persona)
            
            # Tokenize for analysis
            input_tokens = self.tokenizer.tokenize(input_context + " " + instruction)
            
            return AttentionCapture(
                capture_id=str(uuid.uuid4()),
                persona_id=persona.state.persona_id,
                simulation_day=simulation_day,
                simulation_hour=simulation_hour,
                input_tokens=input_tokens,
                prompt_context=input_context + " " + instruction,
                attention_weights=attention_weights,
                layer_attention=layer_attention,
                head_attention=head_attention,
                token_count=len(input_tokens),
                layer_count=len(self.hooks),
                head_count=1,  # Simplified for now
                self_reference_attention=self_reference_attention,
                emotional_salience=emotional_salience,
                memory_integration=memory_integration,
                processing_time_ms=time.time() * 1000
            )
            
        except Exception as e:
            logger.error(f"Error processing attention capture: {e}")
            return None
    
    async def _process_activation_capture(
        self,
        analysis_id: str,
        persona: Persona,
        simulation_day: int,
        simulation_hour: int,
        input_context: str,
        instruction: str,
        response: str
    ) -> Optional[ActivationCapture]:
        """Process captured activation data."""
        try:
            # Aggregate activations across layers
            layer_activations = {}
            circuit_activations = {}
            
            for i, hook in enumerate(self.hooks):
                if hook.activations:
                    # Average activations across batch and sequence
                    layer_activation = torch.cat(hook.activations, dim=0).mean(dim=(0, 1))
                    layer_activations[i] = layer_activation.tolist()
                    
                    # Identify circuit activations (simplified)
                    circuit_name = f"layer_{i}_circuit"
                    circuit_activations[circuit_name] = layer_activation.tolist()
            
            if not layer_activations:
                return None
            
            # Calculate activation metrics
            all_activations = torch.cat([torch.tensor(acts) for acts in layer_activations.values()])
            activation_magnitude = all_activations.abs().mean().item()
            activation_sparsity = (all_activations == 0).float().mean().item()
            circuit_specialization = self._calculate_circuit_specialization(circuit_activations)
            
            return ActivationCapture(
                capture_id=str(uuid.uuid4()),
                persona_id=persona.state.persona_id,
                simulation_day=simulation_day,
                simulation_hour=simulation_hour,
                layer_activations=layer_activations,
                circuit_activations=circuit_activations,
                layer_count=len(layer_activations),
                neuron_count=sum(len(acts) for acts in layer_activations.values()),
                circuit_count=len(circuit_activations),
                activation_magnitude=activation_magnitude,
                activation_sparsity=activation_sparsity,
                circuit_specialization=circuit_specialization,
                processing_time_ms=time.time() * 1000
            )
            
        except Exception as e:
            logger.error(f"Error processing activation capture: {e}")
            return None
    
    def _calculate_self_reference_attention(self, attention_weights: List[List[float]]) -> float:
        """Calculate self-reference attention score."""
        if not attention_weights:
            return 0.0
        
        # Simplified: look for attention to self-referential tokens
        attention_matrix = np.array(attention_weights)
        
        # Calculate attention to first token (often self-referential)
        if attention_matrix.shape[0] > 0:
            first_token_attention = attention_matrix[:, 0].mean()
            return min(first_token_attention, 1.0)
        
        return 0.0
    
    def _calculate_emotional_salience(self, attention_weights: List[List[float]], context: str) -> float:
        """Calculate emotional salience score."""
        if not attention_weights:
            return 0.0
        
        # Simplified: look for emotional keywords in context
        emotional_keywords = [
            "happy", "sad", "angry", "anxious", "depressed", "excited",
            "worried", "fear", "joy", "love", "hate", "stress"
        ]
        
        context_lower = context.lower()
        emotional_count = sum(1 for word in emotional_keywords if word in context_lower)
        
        # Normalize by context length
        word_count = len(context.split())
        if word_count == 0:
            return 0.0
        
        return min(emotional_count / word_count, 1.0)
    
    def _calculate_memory_integration(self, attention_weights: List[List[float]], persona: Persona) -> float:
        """Calculate memory integration score."""
        if not attention_weights:
            return 0.0
        
        # Simplified: check if persona's memories are referenced
        memory_keywords = []
        for memory in persona.baseline.core_memories:
            memory_keywords.extend(memory.lower().split()[:3])  # First 3 words
        
        # This would require more sophisticated analysis in practice
        # For now, return a baseline score
        return 0.3
    
    def _calculate_circuit_specialization(self, circuit_activations: Dict[str, List[float]]) -> float:
        """Calculate circuit specialization score."""
        if not circuit_activations:
            return 0.0
        
        # Calculate variance in circuit activations
        all_activations = []
        for activations in circuit_activations.values():
            all_activations.extend(activations)
        
        if not all_activations:
            return 0.0
        
        # Higher variance indicates more specialization
        variance = np.var(all_activations)
        return min(variance, 1.0)
    
    def _calculate_data_quality(self, attention_capture: Optional[AttentionCapture], 
                              activation_capture: Optional[ActivationCapture]) -> float:
        """Calculate data quality score."""
        quality_scores = []
        
        if attention_capture:
            # Check attention data quality
            if attention_capture.attention_weights:
                quality_scores.append(0.9)
            if attention_capture.layer_attention:
                quality_scores.append(0.8)
        
        if activation_capture:
            # Check activation data quality
            if activation_capture.layer_activations:
                quality_scores.append(0.9)
            if activation_capture.circuit_activations:
                quality_scores.append(0.8)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_completeness(self, attention_capture: Optional[AttentionCapture],
                              activation_capture: Optional[ActivationCapture]) -> float:
        """Calculate analysis completeness score."""
        completeness = 0.0
        
        if attention_capture:
            completeness += 0.5
        if activation_capture:
            completeness += 0.5
        
        return completeness
    
    def start_analysis(self):
        """Start mechanistic analysis."""
        self.is_analyzing = True
        logger.info("Mechanistic analysis started")
    
    def stop_analysis(self):
        """Stop mechanistic analysis."""
        self.is_analyzing = False
        self._remove_hooks()
        logger.info("Mechanistic analysis stopped")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_analyses": self.total_analyses,
            "total_capture_time": self.total_capture_time,
            "average_capture_time": self.total_capture_time / max(self.total_analyses, 1),
            "is_analyzing": self.is_analyzing,
            "hooks_registered": len(self.hooks)
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_analysis()
        logger.info("Mechanistic analyzer cleaned up") 