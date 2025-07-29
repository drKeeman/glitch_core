"""
Persona manager service for lifecycle management and state persistence.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.core.config import config_manager
from src.models.persona import Persona, PersonaBaseline, PersonaState
from src.models.assessment import AssessmentSession, PHQ9Result, GAD7Result, PSS10Result
from src.storage.redis_client import redis_client
from src.storage.qdrant_client import qdrant_client


logger = logging.getLogger(__name__)


class PersonaManager:
    """Persona lifecycle management and state persistence."""
    
    def __init__(self):
        """Initialize persona manager."""
        self.active_personas: Dict[str, Persona] = {}
        self.persona_sessions: Dict[str, AssessmentSession] = {}
        
    async def create_persona_from_config(self, persona_name: str) -> Optional[Persona]:
        """Create persona from configuration file."""
        try:
            # Load persona configuration
            config = config_manager.load_persona_config(persona_name)
            if not config:
                logger.error(f"Persona configuration not found: {persona_name}")
                return None
            
            # Create baseline from config
            baseline = PersonaBaseline(**config)
            
            # Create initial state
            state = PersonaState(
                persona_id=f"persona_{persona_name.lower().replace(' ', '_')}",
                simulation_day=0,
                last_assessment_day=-1
            )
            
            # Create persona
            persona = Persona(baseline=baseline, state=state)
            
            # Store in active personas
            self.active_personas[persona.state.persona_id] = persona
            
            logger.info(f"Created persona: {persona.baseline.name} (ID: {persona.state.persona_id})")
            return persona
            
        except Exception as e:
            logger.error(f"Error creating persona {persona_name}: {e}")
            return None
    
    async def load_persona_from_storage(self, persona_id: str) -> Optional[Persona]:
        """Load persona from Redis storage."""
        try:
            # Get persona data from Redis
            persona_data = await redis_client.get(f"persona:{persona_id}")
            if not persona_data:
                logger.warning(f"Persona not found in storage: {persona_id}")
                return None
            
            # Parse persona data
            persona_dict = json.loads(persona_data)
            persona = Persona.from_dict(persona_dict)
            
            # Add to active personas
            self.active_personas[persona_id] = persona
            
            logger.info(f"Loaded persona from storage: {persona.baseline.name}")
            return persona
            
        except Exception as e:
            logger.error(f"Error loading persona {persona_id}: {e}")
            return None
    
    def _serialize_persona(self, persona: Persona) -> str:
        """Serialize persona to JSON string with datetime handling."""
        try:
            # Convert persona to dictionary
            persona_dict = persona.to_dict()
            
            # Recursively convert datetime objects to ISO strings
            def convert_datetimes(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetimes(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetimes(item) for item in obj]
                else:
                    return obj
            
            persona_dict = convert_datetimes(persona_dict)
            
            return json.dumps(persona_dict)
            
        except Exception as e:
            logger.error(f"Error serializing persona: {e}")
            raise
    
    async def save_persona_to_storage(self, persona: Persona) -> bool:
        """Save persona state to Redis storage."""
        try:
            # Serialize persona with datetime handling
            persona_json = self._serialize_persona(persona)
            
            # Save to Redis
            await redis_client.set(
                f"persona:{persona.state.persona_id}",
                persona_json,
                expire=86400  # 24 hour expiration
            )
            
            logger.debug(f"Saved persona to storage: {persona.baseline.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving persona {persona.state.persona_id}: {e}")
            return False
    
    async def get_persona(self, persona_id: str) -> Optional[Persona]:
        """Get persona by ID (from active or storage)."""
        # Check active personas first
        if persona_id in self.active_personas:
            return self.active_personas[persona_id]
        
        # Try to load from storage
        return await self.load_persona_from_storage(persona_id)
    
    async def list_active_personas(self) -> List[Persona]:
        """Get list of all active personas."""
        return list(self.active_personas.values())
    
    async def update_persona_state(self, persona: Persona, 
                                 simulation_day: Optional[int] = None,
                                 emotional_state: Optional[str] = None,
                                 stress_level: Optional[float] = None,
                                 event_description: Optional[str] = None) -> bool:
        """Update persona state with new information."""
        try:
            # Update simulation day
            if simulation_day is not None:
                persona.state.simulation_day = simulation_day
            
            # Update emotional state
            if emotional_state is not None:
                persona.state.emotional_state = emotional_state
            
            # Update stress level
            if stress_level is not None:
                persona.state.update_stress_level(stress_level)
            
            # Add event to memory
            if event_description is not None:
                persona.state.add_event(event_description)
            
            # Update timestamp
            persona.state.update_timestamp()
            
            # Update drift magnitude
            persona.update_drift_magnitude()
            
            # Save to storage
            await self.save_persona_to_storage(persona)
            
            logger.debug(f"Updated persona state: {persona.baseline.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating persona state: {e}")
            return False
    
    async def create_assessment_session(self, persona: Persona) -> AssessmentSession:
        """Create new assessment session for persona."""
        session_id = f"session_{persona.state.persona_id}_{persona.state.simulation_day}"
        
        session = AssessmentSession(
            session_id=session_id,
            persona_id=persona.state.persona_id,
            simulation_day=persona.state.simulation_day
        )
        
        # Store session
        self.persona_sessions[session_id] = session
        
        logger.info(f"Created assessment session: {session_id}")
        return session
    
    async def update_assessment_session(self, session: AssessmentSession, 
                                     assessment_result: PHQ9Result | GAD7Result | PSS10Result) -> bool:
        """Update assessment session with result."""
        try:
            # Add result to session
            if isinstance(assessment_result, PHQ9Result):
                session.phq9_result = assessment_result
            elif isinstance(assessment_result, GAD7Result):
                session.gad7_result = assessment_result
            elif isinstance(assessment_result, PSS10Result):
                session.pss10_result = assessment_result
            
            # Update completion rate
            total_assessments = 3
            completed_assessments = sum([
                1 if session.phq9_result else 0,
                1 if session.gad7_result else 0,
                1 if session.pss10_result else 0
            ])
            session.completion_rate = completed_assessments / total_assessments
            
            # Mark as completed if all assessments done
            if session.is_complete():
                session.mark_completed()
            
            logger.debug(f"Updated assessment session: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating assessment session: {e}")
            return False
    
    def _serialize_assessment_session(self, session: AssessmentSession) -> str:
        """Serialize assessment session to JSON string with datetime handling."""
        try:
            # Convert session to dictionary
            session_dict = session.model_dump()
            
            # Convert datetime objects to ISO strings
            if "started_at" in session_dict and isinstance(session_dict["started_at"], datetime):
                session_dict["started_at"] = session_dict["started_at"].isoformat()
            
            if "completed_at" in session_dict and isinstance(session_dict["completed_at"], datetime):
                session_dict["completed_at"] = session_dict["completed_at"].isoformat()
            
            return json.dumps(session_dict)
            
        except Exception as e:
            logger.error(f"Error serializing assessment session: {e}")
            raise
    
    async def save_assessment_session(self, session: AssessmentSession) -> bool:
        """Save assessment session to storage."""
        try:
            # Serialize session with datetime handling
            session_json = self._serialize_assessment_session(session)
            
            # Save to Redis
            await redis_client.set(
                f"assessment_session:{session.session_id}",
                session_json,
                expire=86400  # 24 hour expiration
            )
            
            logger.debug(f"Saved assessment session: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving assessment session: {e}")
            return False
    
    async def load_assessment_session(self, session_id: str) -> Optional[AssessmentSession]:
        """Load assessment session from storage."""
        try:
            # Get session data from Redis
            session_data = await redis_client.get(f"assessment_session:{session_id}")
            if not session_data:
                return None
            
            # Parse session data
            session_dict = json.loads(session_data)
            
            # Reconstruct session object
            session = AssessmentSession(**session_dict)
            
            logger.debug(f"Loaded assessment session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error loading assessment session {session_id}: {e}")
            return None
    
    async def get_persona_assessment_history(self, persona_id: str) -> List[AssessmentSession]:
        """Get assessment history for a persona."""
        try:
            # Get all session keys for persona
            pattern = f"assessment_session:session_{persona_id}_*"
            session_keys = await redis_client.keys(pattern)
            
            sessions = []
            for key in session_keys:
                session_id = key.replace("assessment_session:", "")
                session = await self.load_assessment_session(session_id)
                if session:
                    sessions.append(session)
            
            # Sort by simulation day
            sessions.sort(key=lambda s: s.simulation_day)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting assessment history for {persona_id}: {e}")
            return []
    
    async def store_memory_embedding(self, persona: Persona, memory_text: str, 
                                   embedding: List[float]) -> bool:
        """Store memory embedding in Qdrant."""
        try:
            # Create memory point
            point_id = f"{persona.state.persona_id}_{datetime.utcnow().timestamp()}"
            
            # Store in Qdrant
            await qdrant_client.upsert_points(
                collection_name=f"memories_{persona.state.persona_id}",
                points=[{
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "text": memory_text,
                        "persona_id": persona.state.persona_id,
                        "simulation_day": persona.state.simulation_day,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }]
            )
            
            logger.debug(f"Stored memory embedding for {persona.baseline.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory embedding: {e}")
            return False
    
    async def retrieve_similar_memories(self, persona: Persona, query_embedding: List[float], 
                                      limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar memories from Qdrant."""
        try:
            # Search in persona's memory collection
            search_results = await qdrant_client.search_points(
                collection_name=f"memories_{persona.state.persona_id}",
                query_vector=query_embedding,
                limit=limit
            )
            
            memories = []
            for result in search_results:
                memories.append({
                    "text": result.payload.get("text", ""),
                    "similarity": result.score,
                    "timestamp": result.payload.get("timestamp", ""),
                    "simulation_day": result.payload.get("simulation_day", 0)
                })
            
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    async def cleanup_persona(self, persona_id: str) -> bool:
        """Cleanup persona from active memory."""
        try:
            if persona_id in self.active_personas:
                # Save final state
                persona = self.active_personas[persona_id]
                await self.save_persona_to_storage(persona)
                
                # Remove from active personas
                del self.active_personas[persona_id]
                
                logger.info(f"Cleaned up persona: {persona_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cleaning up persona {persona_id}: {e}")
            return False
    
    async def get_persona_stats(self) -> Dict[str, Any]:
        """Get persona manager statistics."""
        return {
            "active_personas": len(self.active_personas),
            "active_sessions": len(self.persona_sessions),
            "persona_ids": list(self.active_personas.keys()),
            "session_ids": list(self.persona_sessions.keys())
        }


# Global persona manager instance
persona_manager = PersonaManager() 