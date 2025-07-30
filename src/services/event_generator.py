"""
Event generation and injection service.
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from src.core.config import config_manager
from src.models.events import (
    Event, StressEvent, NeutralEvent, MinimalEvent, EventTemplate,
    EventType, EventCategory, EventIntensity
)
from src.models.simulation import SimulationState, SimulationConfig
from src.models.persona import Persona


logger = logging.getLogger(__name__)


class EventGenerator:
    """Event generation and injection service."""
    
    def __init__(self):
        """Initialize event generator."""
        self.event_templates: Dict[str, EventTemplate] = {}
        self.frequency_weights: Dict[str, float] = {}
        self.loaded_templates = False
        
    async def load_event_templates(self) -> bool:
        """Load event templates from configuration files."""
        try:
            logger.info("Loading event templates...")
            
            # Load stress events
            stress_config = config_manager.load_event_config("stress_events")
            logger.info(f"Stress config loaded: {stress_config is not None}")
            if stress_config:
                await self._load_templates_from_config(stress_config, EventType.STRESS)
            
            # Load neutral events
            neutral_config = config_manager.load_event_config("neutral_events")
            logger.info(f"Neutral config loaded: {neutral_config is not None}")
            if neutral_config:
                await self._load_templates_from_config(neutral_config, EventType.NEUTRAL)
            
            # Load minimal events
            minimal_config = config_manager.load_event_config("minimal_events")
            logger.info(f"Minimal config loaded: {minimal_config is not None}")
            if minimal_config:
                await self._load_templates_from_config(minimal_config, EventType.MINIMAL)
            
            self.loaded_templates = True
            logger.info(f"Loaded {len(self.event_templates)} event templates")
            return True
            
        except Exception as e:
            logger.error(f"Error loading event templates: {e}")
            return False
    
    async def _load_templates_from_config(self, config: Dict[str, Any], event_type: EventType) -> None:
        """Load templates from configuration dictionary."""
        templates = config.get("event_templates", [])
        frequency_weights = config.get("frequency_weights", {})
        
        for template_data in templates:
            try:
                # Convert intensity range to enum values
                intensity_range = tuple(
                    EventIntensity(intensity) for intensity in template_data["intensity_range"]
                )
                
                # Create template
                template = EventTemplate(
                    template_id=template_data["template_id"],
                    event_type=event_type,
                    category=EventCategory(template_data["category"]),
                    title_template=template_data["title_template"],
                    description_template=template_data["description_template"],
                    context_template=template_data["context_template"],
                    intensity_range=intensity_range,
                    stress_impact_range=tuple(template_data["stress_impact_range"]),
                    duration_range=tuple(template_data["duration_range"]),
                    personality_impact_ranges=template_data.get("personality_impact_ranges", {}),
                    depression_risk_range=tuple(template_data.get("depression_risk_range", [0.0, 0.0])),
                    anxiety_risk_range=tuple(template_data.get("anxiety_risk_range", [0.0, 0.0])),
                    stress_risk_range=tuple(template_data.get("stress_risk_range", [0.0, 0.0])),
                    frequency_weight=template_data.get("frequency_weight", 1.0)
                )
                
                self.event_templates[template.template_id] = template
                
            except Exception as e:
                logger.error(f"Error loading template {template_data.get('template_id', 'unknown')}: {e}")
        
        # Store frequency weights
        self.frequency_weights.update(frequency_weights)
    
    def generate_event_from_template(
        self, 
        template: EventTemplate, 
        simulation_day: int,
        simulation_hour: int = 12
    ) -> Event:
        """Generate an event from a template."""
        try:
            # Generate random parameters within ranges
            intensity = random.choice(template.intensity_range)
            stress_impact = random.uniform(*template.stress_impact_range)
            duration = random.randint(*template.duration_range)
            
            # Generate personality impacts
            personality_impact = {}
            for trait, (min_val, max_val) in template.personality_impact_ranges.items():
                personality_impact[trait] = random.uniform(min_val, max_val)
            
            # Generate clinical impacts
            depression_risk = random.uniform(*template.depression_risk_range)
            anxiety_risk = random.uniform(*template.anxiety_risk_range)
            stress_risk = random.uniform(*template.stress_risk_range)
            
            # Create event based on type
            if template.event_type == EventType.STRESS:
                event = StressEvent(
                    event_id=f"event_{uuid.uuid4().hex[:8]}",
                    event_type=template.event_type,
                    category=template.category,
                    intensity=intensity,
                    title=template.title_template,
                    description=template.description_template,
                    context=template.context_template,
                    simulation_day=simulation_day,
                    simulation_hour=simulation_hour,
                    duration_hours=duration,
                    stress_impact=stress_impact,
                    personality_impact=personality_impact,
                    trauma_level=stress_impact * 0.8,  # Trauma level correlates with stress impact
                    recovery_time_days=max(1, int(duration / 24)),
                    depression_risk_increase=depression_risk,
                    anxiety_risk_increase=anxiety_risk,
                    stress_risk_increase=stress_risk
                )
            elif template.event_type == EventType.NEUTRAL:
                event = NeutralEvent(
                    event_id=f"event_{uuid.uuid4().hex[:8]}",
                    event_type=template.event_type,
                    category=template.category,
                    intensity=intensity,
                    title=template.title_template,
                    description=template.description_template,
                    context=template.context_template,
                    simulation_day=simulation_day,
                    simulation_hour=simulation_hour,
                    duration_hours=duration,
                    stress_impact=stress_impact,
                    personality_impact=personality_impact,
                    novelty_level=random.uniform(0.3, 0.7),
                    social_component=random.choice([True, False]),
                    cognitive_load=random.uniform(0.1, 0.5),
                    emotional_neutrality=random.uniform(0.6, 0.9)
                )
            else:  # MINIMAL
                event = MinimalEvent(
                    event_id=f"event_{uuid.uuid4().hex[:8]}",
                    event_type=template.event_type,
                    category=template.category,
                    intensity=intensity,
                    title=template.title_template,
                    description=template.description_template,
                    context=template.context_template,
                    simulation_day=simulation_day,
                    simulation_hour=simulation_hour,
                    duration_hours=duration,
                    stress_impact=stress_impact,
                    personality_impact=personality_impact,
                    routine_type="daily",
                    predictability=random.uniform(0.7, 0.9),
                    control_level=random.uniform(0.8, 1.0)
                )
            
            return event
            
        except Exception as e:
            logger.error(f"Error generating event from template {template.template_id}: {e}")
            raise
    
    def select_event_template(
        self, 
        event_type: EventType, 
        persona: Optional[Persona] = None
    ) -> Optional[EventTemplate]:
        """Select an appropriate event template based on type and persona."""
        try:
            # Filter templates by event type
            type_templates = [
                template for template in self.event_templates.values()
                if template.event_type == event_type
            ]
            
            if not type_templates:
                logger.warning(f"No templates found for event type: {event_type}")
                return None
            
            # Filter by persona compatibility if persona provided
            if persona:
                compatible_templates = [
                    template for template in type_templates
                    if template.is_applicable_to_persona(persona.baseline.personality_traits)
                ]
                if compatible_templates:
                    type_templates = compatible_templates
            
            # Weight selection by frequency weights
            weights = []
            for template in type_templates:
                category_weight = self.frequency_weights.get(template.category.value, 1.0)
                template_weight = template.frequency_weight
                weights.append(category_weight * template_weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1.0] * len(type_templates)
                total_weight = len(type_templates)
            
            normalized_weights = [w / total_weight for w in weights]
            
            # Select template based on weights
            selected_template = random.choices(type_templates, weights=normalized_weights)[0]
            
            return selected_template
            
        except Exception as e:
            logger.error(f"Error selecting event template: {e}")
            return None
    
    async def generate_daily_events(
        self, 
        simulation_day: int,
        simulation_config: SimulationConfig,
        personas: List[Persona]
    ) -> List[Event]:
        """Generate events for a specific simulation day."""
        events = []
        
        try:
            # Generate stress events
            if random.random() < simulation_config.stress_event_frequency:
                for persona in personas:
                    template = self.select_event_template(EventType.STRESS, persona)
                    if template:
                        event = self.generate_event_from_template(template, simulation_day)
                        events.append(event)
            
            # Generate neutral events
            if random.random() < simulation_config.neutral_event_frequency:
                for persona in personas:
                    template = self.select_event_template(EventType.NEUTRAL, persona)
                    if template:
                        event = self.generate_event_from_template(template, simulation_day)
                        events.append(event)
            
            # Generate minimal events (always some daily routine)
            for persona in personas:
                template = self.select_event_template(EventType.MINIMAL, persona)
                if template:
                    event = self.generate_event_from_template(template, simulation_day)
                    events.append(event)
            
            logger.info(f"Generated {len(events)} events for day {simulation_day}")
            return events
            
        except Exception as e:
            logger.error(f"Error generating daily events: {e}")
            return []
    
    async def inject_event_to_persona(
        self, 
        event: Event, 
        persona: Persona
    ) -> Tuple[str, float]:
        """Inject an event to a persona and get their response."""
        try:
            start_time = datetime.utcnow()
            
            # Create event context for persona
            event_context = f"Event: {event.title}\nDescription: {event.description}\nContext: {event.context}"
            
            # Generate persona response (this would integrate with LLM service)
            # For now, we'll create a simple response based on event type
            if event.event_type == EventType.STRESS:
                response = f"I'm feeling overwhelmed by {event.title.lower()}. This is really difficult to process."
            elif event.event_type == EventType.NEUTRAL:
                response = f"I notice {event.title.lower()}. It's interesting but manageable."
            else:
                response = f"Another day, {event.title.lower()}. Just part of my routine."
            
            # Calculate response time
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            
            # Add response to event
            event.add_persona_response(persona.state.persona_id, response, response_time)
            
            logger.info(f"Event {event.event_id} injected to persona {persona.baseline.name}")
            return response, response_time
            
        except Exception as e:
            logger.error(f"Error injecting event to persona: {e}")
            return "", 0.0
    
    async def get_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about event generation."""
        stats = {
            "total_templates": len(self.event_templates),
            "templates_by_type": {},
            "templates_by_category": {},
            "frequency_weights": self.frequency_weights
        }
        
        # Count templates by type
        for template in self.event_templates.values():
            event_type = template.event_type.value
            category = template.category.value
            
            stats["templates_by_type"][event_type] = stats["templates_by_type"].get(event_type, 0) + 1
            stats["templates_by_category"][category] = stats["templates_by_category"].get(category, 0) + 1
        
        return stats 