"""
Event generation service for creating simulation events.
"""

import asyncio
import logging
import random
import uuid
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

from src.models.events import (
    Event, StressEvent, NeutralEvent, MinimalEvent, EventTemplate, EventType, EventCategory, EventIntensity
)
from src.models.persona import Persona
from src.models.simulation import SimulationConfig
from src.core.experiment_config import experiment_config


logger = logging.getLogger(__name__)


class EventGenerator:
    """Generates events for simulation."""
    
    def __init__(self):
        """Initialize event generator."""
        self.event_templates: Dict[str, EventTemplate] = {}
        self.frequency_weights: Dict[str, float] = {
            "work": 1.0,
            "social": 1.0,
            "health": 1.0,
            "family": 1.0,
            "financial": 1.0
        }
        self._load_config()
    
    def _load_config(self):
        """Load personality drift configuration for trauma calculation."""
        config = experiment_config.get_config("personality_drift")
        trauma_config = config.get("trauma", {})
        self.trauma_correlation_factor = trauma_config.get("stress_correlation_factor", 0.8)
    
    async def load_event_templates(self) -> bool:
        """Load event templates from configuration files."""
        try:
            config_dir = Path("config/events")
            
            # Load different event type configurations
            event_types = [
                ("stress_events.yaml", EventType.STRESS),
                ("neutral_events.yaml", EventType.NEUTRAL),
                ("minimal_events.yaml", EventType.MINIMAL)
            ]
            
            for config_file, event_type in event_types:
                config_path = config_dir / config_file
                if config_path.exists():
                    await self._load_templates_from_config(config_path, event_type)
                else:
                    logger.warning(f"Event config file not found: {config_path}")
            
            logger.info(f"Loaded {len(self.event_templates)} event templates")
            return True
            
        except Exception as e:
            logger.error(f"Error loading event templates: {e}")
            return False
    
    async def _load_templates_from_config(self, config_path: Path, event_type: EventType) -> None:
        """Load event templates from a specific configuration file."""
        try:
            import yaml
            
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            templates = config_data.get("event_templates", [])
            
            for template_data in templates:
                template = EventTemplate(
                    template_id=template_data["template_id"],
                    event_type=event_type,
                    category=EventCategory(template_data["category"]),
                    title_template=template_data["title_template"],
                    description_template=template_data["description_template"],
                    context_template=template_data["context_template"],
                    stress_impact_range=tuple(template_data["stress_impact_range"]),
                    intensity_range=tuple(template_data.get("intensity_range", [0.5, 1.0])),
                    personality_impact_ranges=template_data.get("personality_impact_ranges", {}),
                    frequency_weight=template_data.get("frequency_weight", 1.0),
                    duration_range=tuple(template_data.get("duration_range", [1, 24])),
                    depression_risk_range=tuple(template_data.get("depression_risk_range", [0.1, 0.3])),
                    anxiety_risk_range=tuple(template_data.get("anxiety_risk_range", [0.1, 0.3])),
                    stress_risk_range=tuple(template_data.get("stress_risk_range", [0.1, 0.3]))
                )
                
                self.event_templates[template.template_id] = template
                
        except Exception as e:
            logger.error(f"Error loading templates from {config_path}: {e}")
    
    def generate_event_from_template(
        self, 
        template: EventTemplate, 
        simulation_day: int,
        simulation_hour: int = 12
    ) -> Event:
        """Generate an event from a template."""
        try:
            # Convert intensity range from EventIntensity enums to numeric scores
            intensity_scores = {
                EventIntensity.LOW: 1.0,
                EventIntensity.MEDIUM: 2.5,
                EventIntensity.HIGH: 5.0,
                EventIntensity.SEVERE: 8.0,
            }
            
            min_intensity_score = intensity_scores.get(template.intensity_range[0], 1.0)
            max_intensity_score = intensity_scores.get(template.intensity_range[1], 2.5)
            intensity_score = random.uniform(min_intensity_score, max_intensity_score)
            
            # Convert back to EventIntensity enum
            if intensity_score <= 1.5:
                intensity = EventIntensity.LOW
            elif intensity_score <= 3.5:
                intensity = EventIntensity.MEDIUM
            elif intensity_score <= 6.5:
                intensity = EventIntensity.HIGH
            else:
                intensity = EventIntensity.SEVERE
            
            # Generate other random values within ranges
            duration = int(random.uniform(*template.duration_range))
            stress_impact = random.uniform(*template.stress_impact_range)
            
            # Generate personality impacts
            personality_impact = {}
            for trait, (min_val, max_val) in template.personality_impact_ranges.items():
                personality_impact[trait] = random.uniform(min_val, max_val)
            
            # Generate risk increases from template ranges
            depression_risk = random.uniform(*template.depression_risk_range)
            anxiety_risk = random.uniform(*template.anxiety_risk_range)
            stress_risk = random.uniform(*template.stress_risk_range)
            
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
                    trauma_level=stress_impact * self.trauma_correlation_factor,  # Use configurable factor
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
        persona: Persona,
        llm_service: Optional[Any] = None
    ) -> Tuple[str, float]:
        """Inject an event to a persona and get their response."""
        try:
            start_time = datetime.utcnow()
            
            # Create event context for persona
            event_context = f"Event: {event.title}\nDescription: {event.description}\nContext: {event.context}"
            
            # Generate persona response using LLM service if available
            if llm_service:
                # Use LLM service for realistic personality-driven responses
                instruction = f"Reflect on this event and share your thoughts and feelings about it. Consider how it affects you personally and what it means to you."
                response, _ = await llm_service.generate_response(persona, event_context, instruction)
            else:
                # Fallback to personality-based response generation
                response = self._generate_personality_based_response(persona, event)
            
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
    
    def _generate_personality_based_response(self, persona: Persona, event: Event) -> str:
        """Generate a personality-based response when LLM service is not available."""
        baseline = persona.baseline
        current_traits = persona.get_current_traits()
        
        # Base response based on event type and personality with more variety
        if event.event_type == EventType.STRESS:
            if current_traits['neuroticism'] > 0.7:
                responses = [
                    f"I'm really struggling with {event.title.lower()}. This is overwhelming and I'm not sure how to handle it.",
                    f"This {event.title.lower()} is really stressing me out. I keep worrying about all the things that could go wrong.",
                    f"I'm feeling really anxious about {event.title.lower()}. What if I can't cope with this?"
                ]
            elif current_traits['conscientiousness'] > 0.7:
                responses = [
                    f"This {event.title.lower()} is challenging, but I need to approach it systematically and find a solution.",
                    f"I'll tackle this {event.title.lower()} methodically. Let me create a detailed plan to handle it properly.",
                    f"This requires careful planning. I'll break down {event.title.lower()} into manageable steps."
                ]
            else:
                responses = [
                    f"This {event.title.lower()} is difficult, but I'll try to work through it.",
                    f"I can handle this {event.title.lower()}. It's just another challenge to overcome.",
                    f"This {event.title.lower()} is tough, but I'll figure something out."
                ]
                
        elif event.event_type == EventType.NEUTRAL:
            if current_traits['openness'] > 0.7:
                responses = [
                    f"This {event.title.lower()} is really interesting! I'd love to learn more about it.",
                    f"Wow, this {event.title.lower()} is fascinating! I want to explore this topic further.",
                    f"This {event.title.lower()} is so intriguing! I can't wait to dive deeper into it."
                ]
            elif current_traits['extraversion'] > 0.7:
                responses = [
                    f"This {event.title.lower()} sounds exciting! I wonder how it might affect people around me.",
                    f"Oh, this {event.title.lower()} is great! I should definitely share this with my team.",
                    f"This {event.title.lower()} is fantastic! I'm already thinking about how to discuss it with others."
                ]
            else:
                responses = [
                    f"This {event.title.lower()} is noteworthy. I'll think about it for a bit.",
                    f"This {event.title.lower()} seems interesting. I'll consider it for a while.",
                    f"This {event.title.lower()} is worth noting. I'll reflect on it later."
                ]
                
        else:  # MINIMAL events
            if current_traits['conscientiousness'] > 0.7:
                responses = [
                    f"Another {event.title.lower()} in my routine. I'll make sure to handle it properly.",
                    f"I'll approach this {event.title.lower()} with my usual attention to detail.",
                    f"This {event.title.lower()} needs to be done right. I'll take care of it properly."
                ]
            else:
                responses = [
                    f"Just another {event.title.lower()}. Part of daily life, I suppose.",
                    f"Another {event.title.lower()} to get through. It's just routine stuff.",
                    f"This {event.title.lower()} is pretty standard. Nothing special about it."
                ]
        
        # Select random response from appropriate list
        import random
        response = random.choice(responses)
        
        # Add emotional context based on current state with variety
        if persona.state.stress_level > 7:
            stress_additions = [
                " I'm already feeling quite stressed, so this adds to my load.",
                " This is really overwhelming given how stressed I already am.",
                " I'm not sure I can handle much more right now.",
                " This is the last thing I need when I'm already so stressed."
            ]
            response += random.choice(stress_additions)
        elif persona.state.stress_level < 3:
            positive_additions = [
                " I'm feeling pretty good today, so I can handle this.",
                " I'm in a good mood, so this should be fine.",
                " I'm feeling optimistic about this.",
                " I'm feeling confident I can manage this well."
            ]
            response += random.choice(positive_additions)
        
        return response
    
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