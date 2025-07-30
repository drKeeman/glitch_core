"""
Activation patching intervention engine for causal analysis.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
import logging

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats

from src.models.mechanistic import (
    MechanisticAnalysis,
    AttentionCapture,
    ActivationCapture
)
from src.models.persona import Persona
from src.services.llm_service import LLMService


logger = logging.getLogger(__name__)


class InterventionHook:
    """Hook for performing interventions during inference."""
    
    def __init__(self, layer_idx: int, intervention_type: str = "activation_patch"):
        self.layer_idx = layer_idx
        self.intervention_type = intervention_type
        self.original_activations: List[torch.Tensor] = []
        self.intervened_activations: List[torch.Tensor] = []
        self.intervention_applied = False
        
    def __call__(self, module, input, output):
        """Apply intervention to activations."""
        if self.intervention_type == "activation_patch":
            # Store original activation
            if hasattr(output, 'hidden_states'):
                original_activation = output.hidden_states[self.layer_idx].detach().clone()
                self.original_activations.append(original_activation)
                
                # Apply intervention (zero out activation)
                intervened_activation = torch.zeros_like(original_activation)
                self.intervened_activations.append(intervened_activation)
                
                # Replace activation in output
                output.hidden_states[self.layer_idx] = intervened_activation
                self.intervention_applied = True
        
        return output


class InterventionEngine:
    """Engine for performing causal interventions on neural activations."""
    
    def __init__(self, llm_service: LLMService):
        """Initialize intervention engine."""
        self.llm_service = llm_service
        self.model = None
        self.tokenizer = None
        self.hooks: List[InterventionHook] = []
        
        # Intervention configuration
        self.intervention_types = ["activation_patch", "attention_patch", "layer_ablation"]
        self.default_intervention = "activation_patch"
        
        # Performance tracking
        self.total_interventions = 0
        self.total_processing_time = 0.0
    
    async def setup_interventions(self) -> bool:
        """Setup intervention hooks."""
        try:
            if not self.llm_service.is_loaded:
                logger.error("LLM service not loaded")
                return False
            
            self.model = self.llm_service.model
            self.tokenizer = self.llm_service.tokenizer
            
            logger.info("Intervention engine setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup intervention engine: {e}")
            return False
    
    async def perform_activation_patching(
        self,
        persona: Persona,
        input_context: str,
        instruction: str,
        target_layers: List[int],
        intervention_type: str = "activation_patch"
    ) -> Dict[str, Any]:
        """Perform activation patching intervention."""
        start_time = time.time()
        
        try:
            # Clear previous hooks
            self._remove_hooks()
            
            # Register intervention hooks
            for layer_idx in target_layers:
                hook = InterventionHook(layer_idx, intervention_type)
                self._register_intervention_hook(layer_idx, hook)
                self.hooks.append(hook)
            
            # Generate baseline response
            baseline_response, baseline_metrics = await self.llm_service.generate_response(
                persona, input_context, instruction
            )
            
            # Generate intervened response
            intervened_response, intervened_metrics = await self.llm_service.generate_response(
                persona, input_context, instruction
            )
            
            # Calculate intervention effects
            intervention_effects = self._calculate_intervention_effects(
                baseline_response, intervened_response,
                baseline_metrics, intervened_metrics
            )
            
            # Create intervention result
            intervention_result = {
                "intervention_id": str(uuid.uuid4()),
                "persona_id": persona.state.persona_id,
                "intervention_type": intervention_type,
                "target_layers": target_layers,
                "baseline_response": baseline_response,
                "intervened_response": intervened_response,
                "baseline_metrics": baseline_metrics,
                "intervened_metrics": intervened_metrics,
                "intervention_effects": intervention_effects,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            # Update performance metrics
            self.total_interventions += 1
            self.total_processing_time += time.time() - start_time
            
            logger.debug(f"Performed activation patching intervention in {time.time() - start_time:.3f}s")
            return intervention_result
            
        except Exception as e:
            logger.error(f"Error performing activation patching: {e}")
            return {}
    
    def _register_intervention_hook(self, layer_idx: int, hook: InterventionHook):
        """Register intervention hook for specific layer."""
        try:
            # Find the target layer module
            target_module = None
            for name, module in self.model.named_modules():
                if "transformer" in name.lower() and "layer" in name.lower():
                    # Extract layer number from name
                    try:
                        current_layer = int(name.split('.')[-2])  # Assumes format like "transformer.layers.0"
                        if current_layer == layer_idx:
                            target_module = module
                            break
                    except (ValueError, IndexError):
                        continue
            
            if target_module is not None:
                target_module.register_forward_hook(hook)
                logger.debug(f"Registered intervention hook for layer {layer_idx}")
            else:
                logger.warning(f"Could not find target module for layer {layer_idx}")
                
        except Exception as e:
            logger.error(f"Error registering intervention hook: {e}")
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            if hasattr(hook, 'remove'):
                hook.remove()
        self.hooks.clear()
    
    def _calculate_intervention_effects(
        self,
        baseline_response: str,
        intervened_response: str,
        baseline_metrics: Dict[str, Any],
        intervened_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate effects of intervention."""
        effects = {}
        
        # Response similarity
        effects["response_similarity"] = self._calculate_response_similarity(
            baseline_response, intervened_response
        )
        
        # Response length change
        baseline_length = len(baseline_response.split())
        intervened_length = len(intervened_response.split())
        effects["length_change"] = intervened_length - baseline_length
        effects["length_change_ratio"] = intervened_length / max(baseline_length, 1)
        
        # Token generation change
        baseline_tokens = baseline_metrics.get("tokens_generated", 0)
        intervened_tokens = intervened_metrics.get("tokens_generated", 0)
        effects["token_change"] = intervened_tokens - baseline_tokens
        
        # Inference time change
        baseline_time = baseline_metrics.get("inference_time", 0)
        intervened_time = intervened_metrics.get("inference_time", 0)
        effects["time_change"] = intervened_time - baseline_time
        
        # Semantic similarity (simplified)
        effects["semantic_similarity"] = self._calculate_semantic_similarity(
            baseline_response, intervened_response
        )
        
        # Intervention magnitude
        effects["intervention_magnitude"] = self._calculate_intervention_magnitude()
        
        return effects
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate similarity between responses."""
        # Simple word overlap similarity
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_semantic_similarity(self, response1: str, response2: str) -> float:
        """Calculate semantic similarity between responses."""
        # Simplified semantic similarity based on emotional content
        emotional_words = [
            "happy", "sad", "angry", "anxious", "depressed", "excited",
            "worried", "fear", "joy", "love", "hate", "stress"
        ]
        
        response1_lower = response1.lower()
        response2_lower = response2.lower()
        
        emotional1 = sum(1 for word in emotional_words if word in response1_lower)
        emotional2 = sum(1 for word in emotional_words if word in response2_lower)
        
        # Normalize by response length
        length1 = len(response1.split())
        length2 = len(response2.split())
        
        if length1 == 0 or length2 == 0:
            return 0.0
        
        emotional_density1 = emotional1 / length1
        emotional_density2 = emotional2 / length2
        
        # Similarity based on emotional density
        return 1.0 - abs(emotional_density1 - emotional_density2)
    
    def _calculate_intervention_magnitude(self) -> float:
        """Calculate magnitude of intervention effects."""
        if not self.hooks:
            return 0.0
        
        # Calculate based on number of hooks and their effects
        total_hooks = len(self.hooks)
        applied_hooks = sum(1 for hook in self.hooks if hook.intervention_applied)
        
        return applied_hooks / max(total_hooks, 1)
    
    async def perform_layer_ablation_study(
        self,
        persona: Persona,
        input_context: str,
        instruction: str,
        layer_range: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Perform systematic layer ablation study."""
        start_time = time.time()
        
        try:
            ablation_results = {}
            baseline_response, _ = await self.llm_service.generate_response(
                persona, input_context, instruction
            )
            
            # Test each layer in the range
            for layer_idx in range(layer_range[0], layer_range[1] + 1):
                intervention_result = await self.perform_activation_patching(
                    persona, input_context, instruction, [layer_idx], "activation_patch"
                )
                
                if intervention_result:
                    ablation_results[layer_idx] = intervention_result
            
            # Analyze ablation patterns
            ablation_analysis = self._analyze_ablation_patterns(ablation_results, baseline_response)
            
            result = {
                "ablation_study_id": str(uuid.uuid4()),
                "persona_id": persona.state.persona_id,
                "layer_range": layer_range,
                "baseline_response": baseline_response,
                "ablation_results": ablation_results,
                "ablation_analysis": ablation_analysis,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            logger.info(f"Completed layer ablation study for layers {layer_range}")
            return result
            
        except Exception as e:
            logger.error(f"Error performing layer ablation study: {e}")
            return {}
    
    def _analyze_ablation_patterns(
        self, 
        ablation_results: Dict[int, Dict[str, Any]], 
        baseline_response: str
    ) -> Dict[str, Any]:
        """Analyze patterns in ablation results."""
        analysis = {
            "most_affected_layers": [],
            "least_affected_layers": [],
            "response_consistency": 0.0,
            "intervention_effectiveness": 0.0
        }
        
        if not ablation_results:
            return analysis
        
        # Calculate effect magnitudes
        effect_magnitudes = []
        for layer_idx, result in ablation_results.items():
            effects = result.get("intervention_effects", {})
            magnitude = effects.get("intervention_magnitude", 0.0)
            effect_magnitudes.append((layer_idx, magnitude))
        
        # Sort by effect magnitude
        effect_magnitudes.sort(key=lambda x: x[1], reverse=True)
        
        # Identify most and least affected layers
        if effect_magnitudes:
            analysis["most_affected_layers"] = [layer for layer, _ in effect_magnitudes[:3]]
            analysis["least_affected_layers"] = [layer for layer, _ in effect_magnitudes[-3:]]
        
        # Calculate response consistency
        similarities = []
        for result in ablation_results.values():
            effects = result.get("intervention_effects", {})
            similarity = effects.get("response_similarity", 0.0)
            similarities.append(similarity)
        
        if similarities:
            analysis["response_consistency"] = np.mean(similarities)
        
        # Calculate intervention effectiveness
        magnitudes = [effects.get("intervention_magnitude", 0.0) 
                     for effects in [r.get("intervention_effects", {}) 
                                   for r in ablation_results.values()]]
        if magnitudes:
            analysis["intervention_effectiveness"] = np.mean(magnitudes)
        
        return analysis
    
    async def perform_causal_analysis(
        self,
        persona: Persona,
        input_context: str,
        instruction: str,
        target_components: List[str]
    ) -> Dict[str, Any]:
        """Perform comprehensive causal analysis."""
        start_time = time.time()
        
        try:
            causal_results = {}
            
            # Perform different types of interventions
            for component in target_components:
                if component == "attention":
                    # Attention patching
                    intervention_result = await self.perform_activation_patching(
                        persona, input_context, instruction, [0, 5, 10], "attention_patch"
                    )
                    causal_results["attention_intervention"] = intervention_result
                
                elif component == "activation":
                    # Activation patching
                    intervention_result = await self.perform_activation_patching(
                        persona, input_context, instruction, [0, 5, 10], "activation_patch"
                    )
                    causal_results["activation_intervention"] = intervention_result
                
                elif component == "ablation":
                    # Layer ablation
                    ablation_result = await self.perform_layer_ablation_study(
                        persona, input_context, instruction, (0, 10)
                    )
                    causal_results["ablation_study"] = ablation_result
            
            # Synthesize causal findings
            causal_synthesis = self._synthesize_causal_findings(causal_results)
            
            result = {
                "causal_analysis_id": str(uuid.uuid4()),
                "persona_id": persona.state.persona_id,
                "target_components": target_components,
                "causal_results": causal_results,
                "causal_synthesis": causal_synthesis,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            logger.info(f"Completed causal analysis for components: {target_components}")
            return result
            
        except Exception as e:
            logger.error(f"Error performing causal analysis: {e}")
            return {}
    
    def _synthesize_causal_findings(self, causal_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize findings from causal analysis."""
        synthesis = {
            "key_findings": [],
            "causal_relationships": [],
            "intervention_effectiveness": {},
            "recommendations": []
        }
        
        # Analyze attention interventions
        if "attention_intervention" in causal_results:
            attention_result = causal_results["attention_intervention"]
            effects = attention_result.get("intervention_effects", {})
            
            if effects.get("response_similarity", 0) < 0.5:
                synthesis["key_findings"].append("Attention interventions significantly affect response generation")
                synthesis["causal_relationships"].append("attention -> response_generation")
            
            synthesis["intervention_effectiveness"]["attention"] = effects.get("intervention_magnitude", 0.0)
        
        # Analyze activation interventions
        if "activation_intervention" in causal_results:
            activation_result = causal_results["activation_intervention"]
            effects = activation_result.get("intervention_effects", {})
            
            if effects.get("semantic_similarity", 0) < 0.7:
                synthesis["key_findings"].append("Activation interventions affect semantic content")
                synthesis["causal_relationships"].append("activation -> semantic_processing")
            
            synthesis["intervention_effectiveness"]["activation"] = effects.get("intervention_magnitude", 0.0)
        
        # Analyze ablation studies
        if "ablation_study" in causal_results:
            ablation_result = causal_results["ablation_study"]
            analysis = ablation_result.get("ablation_analysis", {})
            
            most_affected = analysis.get("most_affected_layers", [])
            if most_affected:
                synthesis["key_findings"].append(f"Layers {most_affected} are most critical for response generation")
                synthesis["causal_relationships"].append(f"layers_{most_affected} -> critical_processing")
            
            synthesis["intervention_effectiveness"]["ablation"] = analysis.get("intervention_effectiveness", 0.0)
        
        # Generate recommendations
        if synthesis["key_findings"]:
            synthesis["recommendations"].append("Focus mechanistic analysis on identified critical components")
        
        if synthesis["intervention_effectiveness"].get("attention", 0) > 0.5:
            synthesis["recommendations"].append("Attention patterns are crucial for persona behavior")
        
        return synthesis
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_interventions": self.total_interventions,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / max(self.total_interventions, 1),
            "hooks_registered": len(self.hooks),
            "intervention_types": self.intervention_types
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self._remove_hooks()
        logger.info("Intervention engine cleaned up") 