"""
Drift Engine: Orchestrates personality evolution over compressed time.
"""

import asyncio
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from glitch_core.config.settings import get_settings
from glitch_core.config.logging import get_logger, DriftLogger, PersonalityLogger
from glitch_core.core.memory import MemoryManager
from glitch_core.core.llm import ReflectionEngine


@dataclass
class SimulationResult:
    """Result of a drift simulation."""
    experiment_id: str
    persona_config: Dict[str, Any]
    drift_profile: Dict[str, Any]
    epochs: int
    events_per_epoch: int
    start_time: datetime
    end_time: datetime
    emotional_states: List[Dict[str, float]]
    pattern_emergence: List[Dict[str, Any]]
    stability_warnings: List[Dict[str, Any]]
    interventions: List[Dict[str, Any]]


class DriftEngine:
    """
    Runs personality through N epochs of events
    Applies drift profiles (evolution rules)
    Captures interpretability metrics at each step
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.active_simulations: Dict[str, asyncio.Task] = {}
        self.logger = get_logger("drift_engine")
        self.personality_logger = PersonalityLogger()
        
        # Initialize memory and reflection components
        self.memory_manager = MemoryManager()
        self.reflection_engine = ReflectionEngine()
    
    async def run_simulation(
        self,
        persona_config: Dict[str, Any],
        drift_profile: Dict[str, Any],
        epochs: int = None,
        events_per_epoch: int = None,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Run a complete drift simulation.
        
        Args:
            persona_config: Personality configuration
            drift_profile: Evolution rules and patterns
            epochs: Number of epochs to simulate
            events_per_epoch: Events per epoch
            seed: Random seed for reproducibility
            
        Returns:
            SimulationResult with complete metrics
        """
        if seed is not None:
            random.seed(seed)
            self.logger.info("random_seed_set", seed=seed)
        
        epochs = epochs or self.settings.DEFAULT_EPOCHS
        events_per_epoch = events_per_epoch or self.settings.DEFAULT_EVENTS_PER_EPOCH
        
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        start_time = datetime.now()
        
        # Create experiment-specific logger
        drift_logger = DriftLogger(experiment_id)
        
        self.logger.info(
            "simulation_starting",
            experiment_id=experiment_id,
            epochs=epochs,
            events_per_epoch=events_per_epoch,
            persona_type=persona_config.get("type", "unknown"),
            drift_profile_type=drift_profile.get("type", "unknown")
        )
        
        # Log personality profile loading
        self.personality_logger.profile_loaded(
            profile_name=persona_config.get("type", "unknown"),
            traits=persona_config.get("traits", {})
        )
        
        # Initialize tracking lists
        emotional_states = []
        pattern_emergence = []
        stability_warnings = []
        interventions = []
        
        # Current emotional state (starts from persona baseline)
        current_emotional_state = persona_config.get("emotional_baselines", {}).copy()
        
        # Initialize memory and reflection components
        try:
            await self.memory_manager.initialize()
            await self.reflection_engine.initialize()
            self.logger.info("memory_and_reflection_initialized", experiment_id=experiment_id)
        except Exception as e:
            self.logger.error("memory_reflection_init_failed", experiment_id=experiment_id, error=str(e))
            # Continue without memory/reflection if initialization fails
        
        for epoch in range(epochs):
            self.logger.debug(
                "epoch_starting",
                experiment_id=experiment_id,
                epoch=epoch + 1,
                total_epochs=epochs
            )
            
            # Generate events for this epoch
            events = self._generate_epoch_events(events_per_epoch, persona_config)
            
            # Process each event
            for event in events:
                # Apply event impact to emotional state
                old_emotional_state = current_emotional_state.copy()
                current_emotional_state = self._apply_event_impact(
                    current_emotional_state, event, persona_config
                )
                
                # Log emotional shifts
                for emotion, new_value in current_emotional_state.items():
                    old_value = old_emotional_state.get(emotion, 0.0)
                    if abs(new_value - old_value) > 0.01:  # Only log significant changes
                        self.personality_logger.emotional_shift(
                            emotion=emotion,
                            old_value=old_value,
                            new_value=new_value,
                            trigger=f"event_{event['type']}"
                        )
                
                # Store event in memory
                try:
                    await self._store_event_memory(
                        event=event,
                        emotional_state=current_emotional_state,
                        persona_config=persona_config,
                        experiment_id=experiment_id
                    )
                except Exception as e:
                    self.logger.warning("memory_storage_failed", error=str(e))
                
                # Generate reflection if emotional change is significant
                try:
                    await self._generate_event_reflection(
                        event=event,
                        emotional_state=current_emotional_state,
                        persona_config=persona_config,
                        experiment_id=experiment_id
                    )
                except Exception as e:
                    self.logger.warning("reflection_generation_failed", error=str(e))
                
                # Apply drift profile evolution rules
                current_emotional_state = self._apply_drift_evolution(
                    current_emotional_state, drift_profile, epoch
                )
                
                # Check for pattern emergence
                pattern = self._detect_pattern_emergence(
                    current_emotional_state, epoch, events_per_epoch
                )
                if pattern:
                    pattern_emergence.append(pattern)
                    drift_logger.pattern_emerged(
                        pattern_type=pattern["type"],
                        emotions=pattern["emotions"],
                        epoch=epoch,
                        confidence=pattern["confidence"]
                    )
                
                # Check stability warnings
                warning = self._check_stability_warning(
                    current_emotional_state, drift_profile
                )
                if warning:
                    stability_warnings.append(warning)
                    drift_logger.stability_warning(
                        warning_type=warning["type"],
                        emotions=warning["emotions"],
                        risk_level=warning["risk_level"]
                    )
            
            # Record emotional state for this epoch
            emotional_states.append(current_emotional_state.copy())
            
            # Log epoch completion
            drift_logger.epoch_completed(
                epoch=epoch + 1,
                total_epochs=epochs,
                emotional_state=current_emotional_state
            )
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = SimulationResult(
            experiment_id=experiment_id,
            persona_config=persona_config,
            drift_profile=drift_profile,
            epochs=epochs,
            events_per_epoch=events_per_epoch,
            start_time=start_time,
            end_time=end_time,
            emotional_states=emotional_states,
            pattern_emergence=pattern_emergence,
            stability_warnings=stability_warnings,
            interventions=interventions
        )
        
        # Log simulation completion
        drift_logger.simulation_completed(
            duration_seconds=duration,
            patterns_detected=len(pattern_emergence),
            warnings=len(stability_warnings)
        )
        
        self.logger.info(
            "simulation_completed",
            experiment_id=experiment_id,
            duration_seconds=duration,
            patterns_detected=len(pattern_emergence),
            warnings=len(stability_warnings)
        )
        
        return result
    
    def _generate_epoch_events(self, count: int, persona_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate events for an epoch based on persona configuration."""
        event_types = [
            "social_interaction", "work_challenge", "personal_achievement",
            "stressful_situation", "creative_activity", "routine_task",
            "unexpected_event", "reflection_moment", "learning_opportunity"
        ]
        
        events = []
        for _ in range(count):
            event_type = random.choice(event_types)
            intensity = random.uniform(0.1, 1.0)
            
            events.append({
                "type": event_type,
                "intensity": intensity,
                "description": f"{event_type.replace('_', ' ')} with intensity {intensity:.2f}",
                "timestamp": datetime.now()
            })
        
        return events
    
    def _apply_event_impact(
        self,
        emotional_state: Dict[str, float],
        event: Dict[str, Any],
        persona_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply event impact to emotional state based on persona traits."""
        traits = persona_config.get("traits", {})
        
        # Define impact multipliers based on event type and persona traits
        impact_multipliers = {
            "social_interaction": traits.get("extraversion", 0.5),
            "work_challenge": traits.get("conscientiousness", 0.5),
            "personal_achievement": traits.get("openness", 0.5),
            "stressful_situation": traits.get("neuroticism", 0.5),
            "creative_activity": traits.get("openness", 0.5),
            "routine_task": traits.get("conscientiousness", 0.5),
            "unexpected_event": traits.get("neuroticism", 0.5),
            "reflection_moment": traits.get("openness", 0.5),
            "learning_opportunity": traits.get("openness", 0.5)
        }
        
        multiplier = impact_multipliers.get(event["type"], 0.5)
        impact = event["intensity"] * multiplier
        
        # Log trait activation
        self.personality_logger.trait_activation(
            trait=event["type"],
            activation_level=multiplier,
            context=f"event_{event['type']}"
        )
        
        # Apply impact to relevant emotional dimensions
        if event["type"] in ["personal_achievement", "creative_activity"]:
            emotional_state["joy"] = min(1.0, emotional_state.get("joy", 0.5) + impact * 0.3)
        elif event["type"] in ["stressful_situation", "unexpected_event"]:
            emotional_state["anxiety"] = min(1.0, emotional_state.get("anxiety", 0.5) + impact * 0.3)
        elif event["type"] in ["social_interaction"]:
            emotional_state["social_energy"] = min(1.0, emotional_state.get("social_energy", 0.5) + impact * 0.3)
        
        return emotional_state
    
    def _apply_drift_evolution(
        self,
        emotional_state: Dict[str, float],
        drift_profile: Dict[str, Any],
        epoch: int
    ) -> Dict[str, float]:
        """Apply drift profile evolution rules to emotional state."""
        evolution_rules = drift_profile.get("evolution_rules", [])
        
        for rule in evolution_rules:
            rule_type = rule.get("type")
            
            if rule_type == "decay":
                # Gradual decay of emotional states
                for emotion in emotional_state:
                    decay_rate = rule.get("rate", 0.01)
                    emotional_state[emotion] = max(0.0, emotional_state[emotion] - decay_rate)
            
            elif rule_type == "amplification":
                # Amplify certain emotions based on profile
                target_emotion = rule.get("target_emotion")
                if target_emotion in emotional_state:
                    amplification_factor = rule.get("factor", 1.1)
                    emotional_state[target_emotion] = min(1.0, emotional_state[target_emotion] * amplification_factor)
            
            elif rule_type == "oscillation":
                # Add oscillating patterns
                frequency = rule.get("frequency", 10)
                amplitude = rule.get("amplitude", 0.1)
                if epoch % frequency == 0:
                    target_emotion = rule.get("target_emotion")
                    if target_emotion in emotional_state:
                        emotional_state[target_emotion] = max(0.0, min(1.0, 
                            emotional_state[target_emotion] + amplitude * (1 if epoch // frequency % 2 else -1)))
        
        return emotional_state
    
    def _detect_pattern_emergence(
        self,
        emotional_state: Dict[str, float],
        epoch: int,
        events_per_epoch: int
    ) -> Optional[Dict[str, Any]]:
        """Detect emerging patterns in emotional state."""
        # Simple pattern detection - can be enhanced with more sophisticated algorithms
        high_emotions = [k for k, v in emotional_state.items() if v > 0.8]
        low_emotions = [k for k, v in emotional_state.items() if v < 0.2]
        
        if len(high_emotions) >= 2:
            return {
                "type": "emotional_amplification",
                "emotions": high_emotions,
                "epoch": epoch,
                "confidence": 0.7
            }
        
        if len(low_emotions) >= 2:
            return {
                "type": "emotional_dampening",
                "emotions": low_emotions,
                "epoch": epoch,
                "confidence": 0.7
            }
        
        return None
    
    def _check_stability_warning(
        self,
        emotional_state: Dict[str, float],
        drift_profile: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check for stability warnings based on drift profile."""
        stability_config = drift_profile.get("stability_metrics", {})
        warning_threshold = stability_config.get("warning_threshold", 0.9)
        
        # Check for extreme emotional states
        extreme_emotions = [k for k, v in emotional_state.items() if v > warning_threshold]
        
        if extreme_emotions:
            return {
                "type": "extreme_emotion_warning",
                "emotions": extreme_emotions,
                "risk_level": "high" if len(extreme_emotions) > 1 else "medium",
                "timestamp": datetime.now()
            }
        
        return None
    
    async def inject_intervention(
        self,
        experiment_id: str,
        intervention: Dict[str, Any]
    ) -> bool:
        """Inject an intervention into an active simulation."""
        if experiment_id not in self.active_simulations:
            self.logger.warning(
                "intervention_failed",
                experiment_id=experiment_id,
                reason="experiment_not_found"
            )
            return False
        
        # This would be implemented for real-time intervention
        # For now, just log the intervention
        self.logger.info(
            "intervention_injected",
            experiment_id=experiment_id,
            intervention_type=intervention.get("type"),
            impact_score=intervention.get("impact_score", 0.0)
        )
        return True 
    
    async def _store_event_memory(
        self,
        event: Dict[str, Any],
        emotional_state: Dict[str, float],
        persona_config: Dict[str, Any],
        experiment_id: str
    ):
        """Store event in memory with personality-biased encoding."""
        # Calculate emotional weight based on event impact
        emotional_weight = self._calculate_emotional_weight(event, emotional_state)
        
        # Create memory content
        memory_content = f"Event: {event.get('description', event.get('type', 'unknown'))}"
        
        # Get persona biases for memory encoding
        persona_bias = {
            **persona_config.get("traits", {}),
            **persona_config.get("cognitive_biases", {}),
            **persona_config.get("memory_patterns", {})
        }
        
        # Store in memory
        await self.memory_manager.save_memory(
            content=memory_content,
            emotional_weight=emotional_weight,
            persona_bias=persona_bias,
            memory_type="event",
            context={
                "event_type": event.get("type"),
                "emotional_state": emotional_state,
                "epoch": event.get("epoch", 0)
            },
            experiment_id=experiment_id
        )
    
    async def _generate_event_reflection(
        self,
        event: Dict[str, Any],
        emotional_state: Dict[str, float],
        persona_config: Dict[str, Any],
        experiment_id: str
    ):
        """Generate reflection for significant events."""
        # Only generate reflections for significant events
        if not self._is_significant_event(event, emotional_state):
            return
        
        # Retrieve relevant memories
        memories = await self.memory_manager.retrieve_contextual(
            query=event.get("description", event.get("type", "")),
            emotional_state=emotional_state,
            limit=3,
            experiment_id=experiment_id
        )
        
        # Extract memory content
        memory_content = [memory.content for memory in memories]
        
        # Create persona-specific prompt
        persona_prompt = self._create_persona_prompt(persona_config)
        
        # Generate reflection
        reflection_response = await self.reflection_engine.generate_reflection(
            trigger_event=event.get("description", event.get("type", "")),
            emotional_state=emotional_state,
            memories=memory_content,
            persona_prompt=persona_prompt,
            experiment_id=experiment_id
        )
        
        # Store reflection in memory
        await self.memory_manager.save_memory(
            content=reflection_response.reflection,
            emotional_weight=reflection_response.confidence,
            persona_bias=persona_config.get("traits", {}),
            memory_type="reflection",
            context={
                "trigger_event": event.get("type"),
                "generation_time": reflection_response.generation_time,
                "emotional_impact": reflection_response.emotional_impact
            },
            experiment_id=experiment_id
        )
    
    def _calculate_emotional_weight(self, event: Dict[str, Any], emotional_state: Dict[str, float]) -> float:
        """Calculate emotional weight of an event."""
        # Base weight from event type
        event_type = event.get("type", "neutral")
        base_weights = {
            "positive": 0.7,
            "negative": 0.8,
            "neutral": 0.3,
            "trauma": 0.9,
            "success": 0.6,
            "failure": 0.7
        }
        
        base_weight = base_weights.get(event_type, 0.5)
        
        # Adjust based on current emotional state
        emotional_intensity = sum(emotional_state.values()) / len(emotional_state) if emotional_state else 0.5
        
        return min(1.0, base_weight * (1 + emotional_intensity))
    
    def _is_significant_event(self, event: Dict[str, Any], emotional_state: Dict[str, float]) -> bool:
        """Determine if an event is significant enough for reflection."""
        # Check event type significance
        significant_types = ["trauma", "success", "failure", "positive", "negative"]
        if event.get("type") in significant_types:
            return True
        
        # Check emotional impact
        emotional_intensity = sum(emotional_state.values()) / len(emotional_state) if emotional_state else 0.5
        if emotional_intensity > 0.6:
            return True
        
        return False
    
    def _create_persona_prompt(self, persona_config: Dict[str, Any]) -> str:
        """Create a persona-specific prompt for reflection generation."""
        persona_type = persona_config.get("type", "balanced")
        
        prompts = {
            "resilient_optimist": "You are an optimistic and resilient personality. You tend to see the positive side of situations and bounce back quickly from challenges. You're generally cheerful and hopeful.",
            "anxious_overthinker": "You are an anxious personality who tends to overthink situations. You often worry about potential problems and can get caught in negative thought patterns. You're detail-oriented but sometimes struggle with uncertainty.",
            "stoic_philosopher": "You are a stoic personality who values rationality and emotional control. You tend to analyze situations calmly and don't let emotions cloud your judgment. You're thoughtful and philosophical.",
            "creative_volatile": "You are a creative and emotionally expressive personality. You experience emotions intensely and use them as fuel for creative expression. You're passionate but can be unpredictable."
        }
        
        return prompts.get(persona_type, "You are a balanced personality reflecting on your experiences.") 