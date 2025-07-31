"""
Main simulation engine for orchestrating AI personality drift experiments.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

from src.core.config import config_manager
from src.models.simulation import SimulationState, SimulationConfig, SimulationStatus, ExperimentalCondition
from src.models.persona import Persona, PersonaBaseline, PersonaState
from src.models.events import Event
from src.models.assessment import AssessmentSession
from src.services.persona_manager import PersonaManager
from src.services.event_generator import EventGenerator
from src.services.assessment_service import AssessmentService
from src.services.ollama_service import ollama_service
from src.services.memory_service import MemoryService
from src.storage.redis_client import redis_client
from src.storage.qdrant_client import qdrant_client
from src.storage.file_storage import FileStorage


logger = logging.getLogger(__name__)


class SimulationEngine:
    """Main simulation orchestration engine."""
    
    def __init__(self, websocket_manager=None):
        """Initialize simulation engine."""
        self.persona_manager = PersonaManager()
        self.event_generator = EventGenerator()
        self.assessment_service = AssessmentService()
        self.llm_service = ollama_service
        self.memory_service = MemoryService()
        self.file_storage = FileStorage()
        
        # WebSocket manager for real-time updates
        self.websocket_manager = websocket_manager
        
        # Simulation state
        self.simulation_state: Optional[SimulationState] = None
        self.active_personas: Dict[str, Persona] = {}
        self.simulation_config: Optional[SimulationConfig] = None
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.last_checkpoint: Optional[datetime] = None
        self.total_events_processed = 0
        self.total_assessments_completed = 0
        
        # Control flags
        self.should_stop = False
        self.should_pause = False
    
    def set_websocket_manager(self, websocket_manager):
        """Set the WebSocket manager for broadcasting updates."""
        self.websocket_manager = websocket_manager
    
    async def _broadcast_status_update(self, status_data: Dict[str, Any]):
        """Broadcast status update to WebSocket clients."""
        if self.websocket_manager and status_data:
            try:
                await self.websocket_manager.broadcast_simulation_status(status_data)
                logger.info(f"Broadcasted simulation status update: {status_data}")
            except Exception as e:
                logger.error(f"Error broadcasting status update: {e}")
        else:
            logger.info(f"Simulation status update (no WebSocket): {status_data}")
    
    async def _broadcast_progress_update(self, progress_data: Dict[str, Any]):
        """Broadcast progress update to WebSocket clients."""
        if self.websocket_manager and progress_data:
            try:
                await self.websocket_manager.broadcast_progress_update(progress_data)
                logger.info(f"Broadcasted progress update: {progress_data}")
            except Exception as e:
                logger.error(f"Error broadcasting progress update: {e}")
        else:
            logger.info(f"Progress update (no WebSocket): {progress_data}")
    
    async def initialize_simulation(
        self, 
        config_name: str = "experimental_design",
        experimental_condition: ExperimentalCondition = ExperimentalCondition.CONTROL
    ) -> bool:
        """Initialize simulation with configuration."""
        try:
            # Load simulation configuration
            config_data = config_manager.load_simulation_config(config_name)
            if not config_data:
                logger.error(f"Simulation configuration not found: {config_name}")
                return False
            
            # Create simulation config
            self.simulation_config = SimulationConfig(**config_data)
            self.simulation_config.experimental_condition = experimental_condition
            
            # Validate configuration
            if not self.simulation_config.is_valid_configuration():
                logger.error("Invalid simulation configuration")
                return False
            
            # Initialize services
            await self.event_generator.load_event_templates()
            await self.llm_service.initialize()
            await self.memory_service.initialize_memory_system()
            
            # Create simulation state
            simulation_id = f"sim_{uuid.uuid4().hex[:8]}"
            self.simulation_state = SimulationState(
                simulation_id=simulation_id,
                config=self.simulation_config
            )
            
            # Create personas based on experimental condition
            await self._create_personas_for_condition(experimental_condition)
            
            logger.info(f"Simulation initialized: {simulation_id}")
            logger.info(f"Condition: {experimental_condition.value}")
            logger.info(f"Duration: {self.simulation_config.duration_days} days")
            logger.info(f"Personas: {len(self.active_personas)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing simulation: {e}")
            return False
    
    async def _create_personas_for_condition(self, condition: ExperimentalCondition) -> None:
        """Create personas for the experimental condition."""
        # Dynamically load persona names from config directory
        persona_names = config_manager.list_persona_configs()
        
        if not persona_names:
            logger.error("No persona configs found in config directory")
            return
        
        logger.info(f"Found {len(persona_names)} persona configs: {persona_names}")
        
        for persona_name in persona_names:
            persona = await self.persona_manager.create_persona_from_config(persona_name)
            if persona:
                # Add debugging to see what was created
                logger.debug(f"DEBUG: Created persona for {persona_name}, type: {type(persona)}")
                logger.debug(f"DEBUG: persona attributes: {dir(persona)}")
                if hasattr(persona, 'baseline'):
                    logger.debug(f"DEBUG: baseline type: {type(persona.baseline)}")
                    if hasattr(persona.baseline, 'name'):
                        logger.debug(f"DEBUG: baseline.name: {persona.baseline.name}")
                if hasattr(persona, 'state'):
                    logger.debug(f"DEBUG: state type: {type(persona.state)}")
                    if hasattr(persona.state, 'persona_id'):
                        logger.debug(f"DEBUG: state.persona_id: {persona.state.persona_id}")
                
                # Store in active_personas
                self.active_personas[persona.state.persona_id] = persona
                logger.info(f"Created persona: {persona.baseline.name} (ID: {persona.state.persona_id})")
            else:
                logger.error(f"Failed to create persona: {persona_name}")
        
        logger.info(f"Created {len(self.active_personas)} personas for condition: {condition.value}")
    
    def validate_persona_configs(self, persona_names: List[str]) -> bool:
        """Validate that all required persona configs exist."""
        available_configs = config_manager.list_persona_configs()
        missing_configs = [name for name in persona_names if name not in available_configs]
        if missing_configs:
            logger.error(f"Missing persona configs: {missing_configs}")
            return False
        return True
    
    async def run_simulation(self) -> bool:
        """Run the complete simulation."""
        try:
            if not self.simulation_state or not self.simulation_config:
                logger.error("Simulation not initialized")
                return False
            
            # Mark simulation as started
            self.simulation_state.mark_started()
            self.start_time = datetime.utcnow()
            
            # Broadcast initial status
            try:
                status_data = await self.get_simulation_status()
                if status_data:
                    await self._broadcast_status_update(status_data)
            except Exception as e:
                logger.error(f"Error broadcasting initial status: {e}")
            
            logger.info("Starting simulation...")
            logger.info(f"Target duration: {self.simulation_config.duration_days} days")
            logger.info(f"Time compression: {self.simulation_config.time_compression_factor}x")
            logger.info(f"Simulation state before loop: {self.simulation_state.status}")
            
            # Main simulation loop
            for day in range(self.simulation_config.duration_days):
                logger.info(f"Processing simulation day {day + 1}/{self.simulation_config.duration_days}")
                
                if self.should_stop:
                    logger.info("Simulation stopped by user")
                    break
                
                if self.should_pause:
                    logger.info("Simulation paused")
                    self.simulation_state.mark_paused()
                    await asyncio.sleep(1)
                    continue
                
                # Process day
                await self._process_simulation_day(day)
                
                # Update simulation state
                self.simulation_state.current_day = day
                self.simulation_state.advance_time(24)
                
                # Broadcast status update to WebSocket clients
                try:
                    status_data = await self.get_simulation_status()
                    if status_data:
                        await self._broadcast_status_update(status_data)
                except Exception as e:
                    logger.error(f"Error broadcasting status update: {e}")
                
                # Checkpoint if needed
                if self._should_create_checkpoint():
                    await self._create_checkpoint()
                
                # Progress logging and broadcasting
                if day % 7 == 0:  # Weekly progress
                    progress = self.simulation_state.get_progress_percentage()
                    logger.info(f"Simulation progress: {progress:.1f}% (Day {day}/{self.simulation_config.duration_days})")
                    
                    # Broadcast progress update
                    progress_data = {
                        "current_day": day,
                        "total_days": self.simulation_config.duration_days,
                        "progress_percentage": progress,
                        "active_personas": len(self.active_personas),
                        "events_processed": self.total_events_processed,
                        "assessments_completed": self.total_assessments_completed,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await self._broadcast_progress_update(progress_data)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
            
            # Mark simulation as completed
            self.simulation_state.mark_completed()
            
            # Export final simulation results to file
            if self.simulation_state:
                final_results = {
                    "simulation_id": self.simulation_state.simulation_id,
                    "status": self.simulation_state.status.value,
                    "duration_days": self.simulation_config.duration_days if self.simulation_config else 0,
                    "experimental_condition": self.simulation_config.experimental_condition.value if self.simulation_config else "unknown",
                    "total_events_processed": self.total_events_processed,
                    "total_assessments_completed": self.total_assessments_completed,
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "completion_time": datetime.utcnow().isoformat(),
                    "personas": {},
                    "performance_metrics": {
                        "elapsed_time": self.simulation_state.get_elapsed_time(),
                        "average_response_time": self.simulation_state.average_response_time,
                        "memory_usage_mb": self.simulation_state.memory_usage_mb,
                        "cpu_usage_percent": self.simulation_state.cpu_usage_percent,
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Add persona results
                for persona_id, persona in self.active_personas.items():
                    persona_results = {
                        "name": persona.baseline.name,
                        "personality_traits": persona.baseline.personality_traits,
                        "current_traits": persona.get_current_traits(),
                        "stress_level": persona.state.stress_level,
                        "simulation_day": persona.state.simulation_day,
                        "last_assessment_day": persona.state.last_assessment_day,
                        "emotional_state": persona.state.emotional_state,
                        "drift_magnitude": persona.state.drift_magnitude,
                        "trait_changes": persona.state.trait_changes,
                    }
                    final_results["personas"][persona_id] = persona_results
                
                self.file_storage.save_simulation_data(
                    self.simulation_state.simulation_id,
                    final_results,
                    "final_results"
                )
                
                # Export results manifest
                self.file_storage.export_results(self.simulation_state.simulation_id)
            
            # Broadcast final status
            try:
                status_data = await self.get_simulation_status()
                if status_data:
                    await self._broadcast_status_update(status_data)
            except Exception as e:
                logger.error(f"Error broadcasting final status: {e}")
            
            logger.info("Simulation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            if self.simulation_state:
                self.simulation_state.mark_error(str(e))
            return False
    
    async def _process_simulation_day(self, day: int) -> None:
        """Process a single simulation day."""
        try:
            logger.info(f"Processing day {day} - generating events...")
            
            # Add debugging to check what's in active_personas
            logger.debug(f"DEBUG: active_personas keys: {list(self.active_personas.keys())}")
            for persona_id, persona in self.active_personas.items():
                logger.debug(f"DEBUG: persona {persona_id} type: {type(persona)}")
                logger.debug(f"DEBUG: persona {persona_id} attributes: {dir(persona)}")
                if hasattr(persona, 'baseline'):
                    logger.debug(f"DEBUG: persona {persona_id} baseline type: {type(persona.baseline)}")
                    if hasattr(persona.baseline, 'name'):
                        logger.debug(f"DEBUG: persona {persona_id} baseline.name: {persona.baseline.name}")
                if hasattr(persona, 'state'):
                    logger.debug(f"DEBUG: persona {persona_id} state type: {type(persona.state)}")
                    if hasattr(persona.state, 'persona_id'):
                        logger.debug(f"DEBUG: persona {persona_id} state.persona_id: {persona.state.persona_id}")
                # Check if persona has a 'name' attribute directly (this would be wrong)
                if hasattr(persona, 'name'):
                    logger.error(f"ERROR: persona {persona_id} has direct 'name' attribute: {persona.name}")
                    logger.error(f"ERROR: This suggests persona is a PersonaState object, not a Persona object")
                    logger.error(f"ERROR: persona type: {type(persona)}")
                    logger.error(f"ERROR: persona attributes: {dir(persona)}")
                    if hasattr(persona, 'persona_id'):
                        logger.error(f"ERROR: persona.persona_id: {persona.persona_id}")
                    if hasattr(persona, 'baseline') and hasattr(persona.baseline, 'name'):
                        logger.error(f"ERROR: persona.baseline.name: {persona.baseline.name}")
            
            # Validate all personas before processing
            valid_personas = []
            for persona in self.active_personas.values():
                if not isinstance(persona, Persona):
                    logger.error(f"Invalid persona type in active_personas: {type(persona)}")
                    continue
                
                if not hasattr(persona, 'baseline') or not hasattr(persona, 'state'):
                    logger.error(f"Persona missing required attributes: baseline={hasattr(persona, 'baseline')}, state={hasattr(persona, 'state')}")
                    continue
                
                if not hasattr(persona.baseline, 'name'):
                    logger.error(f"Persona baseline missing 'name' attribute")
                    continue
                
                valid_personas.append(persona)
            
            if not valid_personas:
                logger.error("No valid personas found for processing")
                return
            
            # Generate events for the day
            events = await self.event_generator.generate_daily_events(
                day, self.simulation_config, valid_personas
            )
            
            logger.info(f"Generated {len(events)} events for day {day}")
            
            # Save events data to file
            if events and self.simulation_state:
                events_data = {
                    "simulation_id": self.simulation_state.simulation_id,
                    "day": day,
                    "events": [event.model_dump() for event in events],
                    "timestamp": datetime.utcnow().isoformat()
                }
                self.file_storage.save_simulation_data(
                    self.simulation_state.simulation_id, 
                    events_data, 
                    "events"
                )
            
            # Process events for each persona
            for persona in valid_personas:
                logger.info(f"Processing events for {persona.baseline.name}")
                persona_events_data = []
                
                for event in events:
                    # Inject event to persona
                    response, response_time = await self.event_generator.inject_event_to_persona(event, persona)
                    
                    # Update persona state based on event
                    await self._update_persona_from_event(persona, event)
                    
                    # Store event in memory
                    await self._store_event_memory(persona, event, response)
                    
                    # Collect event data for saving
                    event_data = {
                        "event": event.model_dump(),
                        "persona_id": persona.state.persona_id,
                        "persona_name": persona.baseline.name,
                        "response": response,
                        "response_time": response_time,
                        "stress_impact": event.stress_impact,
                        "personality_impact": event.personality_impact,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    persona_events_data.append(event_data)
                    
                    self.total_events_processed += 1
                
                # Save persona-specific events data
                if persona_events_data and self.simulation_state:
                    persona_events_save_data = {
                        "simulation_id": self.simulation_state.simulation_id,
                        "persona_id": persona.state.persona_id,
                        "persona_name": persona.baseline.name,
                        "day": day,
                        "events": persona_events_data,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    self.file_storage.save_simulation_data(
                        self.simulation_state.simulation_id,
                        persona_events_save_data,
                        "persona_events"
                    )
                
                # Check if assessment is due
                if self._should_run_assessment(persona, day):
                    await self._run_persona_assessment(persona, day)
                
                # Save persona state to file
                if self.simulation_state:
                    persona_data = {
                        "simulation_id": self.simulation_state.simulation_id,
                        "persona_id": persona.state.persona_id,
                        "persona_name": persona.baseline.name,
                        "day": day,
                        "baseline": persona.baseline.model_dump(),
                        "state": persona.state.model_dump(),
                        "current_traits": persona.get_current_traits(),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    self.file_storage.save_simulation_data(
                        self.simulation_state.simulation_id,
                        persona_data,
                        "personas"
                    )
                
                # Save persona state to Redis (existing functionality)
                await self.persona_manager.save_persona_to_storage(persona)
            
            # Update simulation state
            self.simulation_state.total_events_processed += len(events)
            logger.info(f"Completed processing day {day}")
            
        except Exception as e:
            logger.error(f"Error processing day {day}: {e}")
            raise
    
    async def _update_persona_from_event(self, persona: Persona, event: Event) -> None:
        """Update persona state based on event impact."""
        try:
            # Update stress level
            current_stress = persona.state.stress_level or 0.0
            new_stress = min(10.0, current_stress + event.stress_impact)
            
            # Update personality traits in state (not baseline)
            for trait, impact in event.personality_impact.items():
                current_drift = persona.state.trait_changes.get(trait, 0.0)
                persona.state.trait_changes[trait] = current_drift + impact
            
            # Update persona state
            await self.persona_manager.update_persona_state(
                persona=persona,
                stress_level=new_stress,
                event_description=event.title
            )
            
        except Exception as e:
            logger.error(f"Error updating persona from event: {e}")
    
    async def _store_event_memory(self, persona: Persona, event: Event, response: str) -> None:
        """Store event in persona's memory."""
        try:
            # Store event memory using memory service
            await self.memory_service.store_event_memory(persona, event, response)
            
        except Exception as e:
            logger.error(f"Error storing event memory: {e}")
    
    def _should_run_assessment(self, persona: Persona, day: int) -> bool:
        """Check if assessment should be run for persona."""
        if not self.simulation_config:
            return False
        
        # Check if it's assessment day
        if day % self.simulation_config.assessment_interval_days != 0:
            return False
        
        # Check if persona hasn't been assessed today
        if persona.state.last_assessment_day == day:
            return False
        
        return True
    
    async def _run_persona_assessment(self, persona: Persona, day: int) -> None:
        """Run assessment for a persona."""
        try:
            # Add detailed debugging to catch the issue
            logger.debug(f"DEBUG: _run_persona_assessment called with persona type: {type(persona)}")
            logger.debug(f"DEBUG: persona object: {persona}")
            logger.debug(f"DEBUG: persona attributes: {dir(persona)}")
            
            # Add runtime type check to catch PersonaState objects early
            if not isinstance(persona, Persona):
                logger.error(f"Invalid persona type passed to assessment: {type(persona)}. Expected Persona. Persona ID: {getattr(persona, 'persona_id', 'unknown')}")
                logger.error(f"DEBUG: persona has attributes: {dir(persona)}")
                if hasattr(persona, 'persona_id'):
                    logger.error(f"DEBUG: persona.persona_id: {persona.persona_id}")
                if hasattr(persona, 'baseline') and hasattr(persona.baseline, 'name'):
                    logger.error(f"DEBUG: persona.baseline.name: {persona.baseline.name}")
                return
            
            # Additional validation to ensure persona has correct structure
            if not isinstance(persona.baseline, PersonaBaseline):
                logger.error(f"Persona baseline is not PersonaBaseline, got {type(persona.baseline)}")
                logger.error(f"DEBUG: baseline attributes: {dir(persona.baseline)}")
                return
            
            if not isinstance(persona.state, PersonaState):
                logger.error(f"Persona state is not PersonaState, got {type(persona.state)}")
                logger.error(f"DEBUG: state attributes: {dir(persona.state)}")
                return
            
            logger.info(f"Running assessment for {persona.baseline.name} on day {day}")
            
            # Run full assessment using the assessment service
            session = await self.assessment_service.conduct_full_assessment(persona)
            
            if session:
                # Update persona state
                await self.persona_manager.update_persona_state(
                    persona=persona,
                    simulation_day=day,
                    last_assessment_day=day
                )
                
                # Save session to Redis (existing functionality)
                await self.persona_manager.save_assessment_session(session)
                
                # Save assessment data to file
                if self.simulation_state:
                    assessment_data = {
                        "simulation_id": self.simulation_state.simulation_id,
                        "persona_id": persona.state.persona_id,
                        "persona_name": persona.baseline.name,
                        "day": day,
                        "session_id": session.session_id,
                        "phq9_result": session.phq9_result.model_dump() if session.phq9_result else None,
                        "gad7_result": session.gad7_result.model_dump() if session.gad7_result else None,
                        "pss10_result": session.pss10_result.model_dump() if session.pss10_result else None,
                        "personality_traits": persona.get_current_traits(),
                        "stress_level": persona.state.stress_level,
                        "emotional_state": persona.state.emotional_state,
                        "drift_magnitude": persona.state.drift_magnitude,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    self.file_storage.save_simulation_data(
                        self.simulation_state.simulation_id,
                        assessment_data,
                        "assessments"
                    )
                
                self.total_assessments_completed += 1
                self.simulation_state.total_assessments_completed += 1
                
                # Broadcast assessment completion event
                if self.websocket_manager:
                    try:
                        logger.debug(f"WebSocket manager found, attempting to broadcast assessment completion for {persona.baseline.name}")
                        assessment_data = {
                            "persona_id": persona.state.persona_id,
                            "persona_name": persona.baseline.name,
                            "simulation_day": day,
                            "assessment_type": "full",
                            "phq9_score": session.phq9_result.total_score if session.phq9_result else None,
                            "gad7_score": session.gad7_result.total_score if session.gad7_result else None,
                            "pss10_score": session.pss10_result.total_score if session.pss10_result else None,
                            "personality_traits": persona.get_current_traits(),
                            "stress_level": persona.state.stress_level,
                            "emotional_state": persona.state.emotional_state,
                            "drift_magnitude": persona.state.drift_magnitude,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        await self.websocket_manager.broadcast_assessment_completed(assessment_data)
                        logger.info(f"Broadcasted assessment completion for {persona.baseline.name}")
                        
                    except Exception as e:
                        logger.error(f"Error broadcasting assessment completion: {e}")
                else:
                    logger.warning(f"No WebSocket manager available for broadcasting assessment completion for {persona.baseline.name}")
                
                logger.info(f"Assessment completed for {persona.baseline.name}")
            else:
                logger.error(f"Failed to complete assessment for {persona.baseline.name}")
            
        except Exception as e:
            logger.error(f"Error running assessment for {persona.baseline.name}: {e}")
    
    def _should_create_checkpoint(self) -> bool:
        """Check if checkpoint should be created."""
        if not self.last_checkpoint:
            return True
        
        hours_since_checkpoint = (datetime.utcnow() - self.last_checkpoint).total_seconds() / 3600
        return hours_since_checkpoint >= self.simulation_config.checkpoint_interval_hours
    
    async def _create_checkpoint(self) -> None:
        """Create simulation checkpoint."""
        try:
            checkpoint_time = datetime.utcnow()
            
            # Save all persona states
            for persona in self.active_personas.values():
                await self.persona_manager.save_persona_to_storage(persona)
            
            # Update simulation state
            self.simulation_state.last_checkpoint = checkpoint_time
            
            # Convert simulation state to JSON for Redis storage
            import json
            simulation_state_dict = self.simulation_state.to_dict()
            
            # Convert datetime objects to ISO strings for JSON serialization
            def convert_datetimes(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetimes(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetimes(item) for item in obj]
                else:
                    return obj
            
            simulation_state_json = json.dumps(convert_datetimes(simulation_state_dict))
            
            # Save simulation state to Redis
            await redis_client.set(
                f"simulation:{self.simulation_state.simulation_id}:checkpoint",
                simulation_state_json,
                expire=86400  # 24 hours
            )
            
            self.last_checkpoint = checkpoint_time
            logger.info(f"Checkpoint created at {checkpoint_time}")
            
        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")
    
    async def pause_simulation(self) -> bool:
        """Pause the simulation."""
        try:
            self.should_pause = True
            if self.simulation_state:
                self.simulation_state.mark_paused()
            
            # Broadcast status update
            try:
                status_data = await self.get_simulation_status()
                if status_data:
                    await self._broadcast_status_update(status_data)
            except Exception as e:
                logger.error(f"Error broadcasting pause status: {e}")
            
            logger.info("Simulation paused")
            return True
        except Exception as e:
            logger.error(f"Error pausing simulation: {e}")
            return False
    
    async def resume_simulation(self) -> bool:
        """Resume the simulation."""
        try:
            self.should_pause = False
            if self.simulation_state:
                self.simulation_state.status = SimulationStatus.RUNNING
            
            # Broadcast status update
            try:
                status_data = await self.get_simulation_status()
                if status_data:
                    await self._broadcast_status_update(status_data)
            except Exception as e:
                logger.error(f"Error broadcasting resume status: {e}")
            
            logger.info("Simulation resumed")
            return True
        except Exception as e:
            logger.error(f"Error resuming simulation: {e}")
            return False
    
    async def stop_simulation(self) -> bool:
        """Stop the simulation."""
        try:
            self.should_stop = True
            if self.simulation_state:
                self.simulation_state.mark_completed()
            
            # Broadcast status update
            try:
                status_data = await self.get_simulation_status()
                if status_data:
                    await self._broadcast_status_update(status_data)
            except Exception as e:
                logger.error(f"Error broadcasting stop status: {e}")
            
            logger.info("Simulation stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping simulation: {e}")
            return False
    
    async def get_simulation_status(self) -> Optional[Dict[str, Any]]:
        """Get current simulation status."""
        if not self.simulation_state:
            return None
        
        # Get base status from simulation state
        status = self.simulation_state.to_dict()
        
        # Map to expected SimulationStatusResponse fields
        status_data = {
            "simulation_id": status.get("simulation_id"),
            "status": status.get("status", "unknown"),
            "current_day": status.get("current_day", 0),
            "total_days": self.simulation_config.duration_days if self.simulation_config else 30,
            "progress_percentage": self.simulation_state.get_progress_percentage() if self.simulation_state else 0.0,
            "active_personas": len(self.active_personas),
            "events_processed": self.total_events_processed,
            "assessments_completed": self.total_assessments_completed,
            "start_time": status.get("start_time"),
            "estimated_completion": self.simulation_state.get_estimated_completion_time() if self.simulation_state else None,
            "is_running": status.get("status") == "running",
            "is_paused": status.get("status") == "paused"
        }
        
        return status_data
    
    async def get_simulation_results(self) -> Dict[str, Any]:
        """Get simulation results and statistics."""
        if not self.simulation_state:
            return {}
        
        results = {
            "simulation_id": self.simulation_state.simulation_id,
            "status": self.simulation_state.status.value,
            "duration_days": self.simulation_config.duration_days if self.simulation_config else 0,
            "experimental_condition": self.simulation_config.experimental_condition.value if self.simulation_config else "unknown",
            "total_events_processed": self.total_events_processed,
            "total_assessments_completed": self.total_assessments_completed,
            "personas": {},
            "performance_metrics": {
                "elapsed_time": self.simulation_state.get_elapsed_time(),
                "average_response_time": self.simulation_state.average_response_time,
                "memory_usage_mb": self.simulation_state.memory_usage_mb,
                "cpu_usage_percent": self.simulation_state.cpu_usage_percent,
            }
        }
        
        # Add persona results
        for persona_id, persona in self.active_personas.items():
            persona_results = {
                "name": persona.baseline.name,
                "personality_traits": persona.baseline.personality_traits,
                "stress_level": persona.state.stress_level,
                "simulation_day": persona.state.simulation_day,
                "last_assessment_day": persona.state.last_assessment_day,
            }
            results["personas"][persona_id] = persona_results
        
        return results
    
    async def cleanup(self) -> None:
        """Clean up simulation resources."""
        try:
            # Save final states
            for persona in self.active_personas.values():
                await self.persona_manager.save_persona_to_storage(persona)
            
            # Cleanup services
            await self.llm_service.cleanup()
            
            logger.info("Simulation cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}") 