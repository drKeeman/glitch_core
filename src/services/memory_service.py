"""
Memory management service for persona memory integration.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import uuid

from src.models.persona import Persona
from src.models.events import Event, EventType
from src.storage.redis_client import redis_client
from src.storage.qdrant_client import qdrant_client
from src.core.config import config_manager


logger = logging.getLogger(__name__)


class MemoryService:
    """Memory management and integration service."""
    
    def __init__(self):
        """Initialize memory service."""
        self.memory_collections = {}
        self.embedding_dimension = 384  # Default embedding dimension
        
    async def initialize_memory_system(self) -> bool:
        """Initialize memory system and collections."""
        try:
            # Dynamically load persona names from config directory
            persona_names = config_manager.list_persona_configs()
            
            if not persona_names:
                logger.warning("No persona configs found in config directory")
                return False
            
            logger.info(f"Found {len(persona_names)} persona configs: {persona_names}")
            
            for persona_name in persona_names:
                collection_name = f"memories_{persona_name}"
                
                # Create collection if it doesn't exist
                await self._create_memory_collection(collection_name)
                self.memory_collections[persona_name] = collection_name
            
            logger.info(f"Memory system initialized successfully for {len(self.memory_collections)} personas")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing memory system: {e}")
            return False
    
    def validate_persona_configs(self, persona_names: List[str]) -> bool:
        """Validate that all required persona configs exist."""
        available_configs = config_manager.list_persona_configs()
        missing_configs = [name for name in persona_names if name not in available_configs]
        if missing_configs:
            logger.error(f"Missing persona configs: {missing_configs}")
            return False
        return True
    
    async def _create_memory_collection(self, collection_name: str) -> bool:
        """Create a memory collection in Qdrant."""
        try:
            # Ensure Qdrant client is connected
            if not qdrant_client._client:
                await qdrant_client.connect()
            
            # Check if collection exists
            collections = await qdrant_client._client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name not in existing_collections:
                # Create collection
                await qdrant_client.create_collection(
                    collection_name=collection_name,
                    vector_size=self.embedding_dimension,
                    distance="Cosine"
                )
                logger.info(f"Created memory collection: {collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating memory collection {collection_name}: {e}")
            return False
    
    async def store_event_memory(
        self, 
        persona: Persona, 
        event: Event, 
        response: str,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """Store an event memory for a persona."""
        try:
            persona_name = persona.baseline.name.lower()
            collection_name = self.memory_collections.get(persona_name)
            
            if not collection_name:
                logger.error(f"No memory collection found for persona: {persona_name}")
                return False
            
            # Create memory record
            memory_id = str(uuid.uuid4())
            memory_text = self._create_memory_text(event, response)
            
            # Generate embedding if not provided
            if embedding is None:
                embedding = await self._generate_memory_embedding(memory_text)
            
            # Store in Qdrant
            await qdrant_client.upsert_points(
                collection_name=collection_name,
                points=[
                    {
                        "id": memory_id,
                        "vector": embedding,
                        "payload": {
                            "memory_text": memory_text,
                            "event_id": event.event_id,
                            "persona_id": persona.state.persona_id,
                            "event_type": event.event_type.value,
                            "event_category": event.category.value,
                            "stress_impact": event.stress_impact,
                            "simulation_day": event.simulation_day,
                            "created_at": datetime.utcnow().isoformat(),
                            "memory_salience": event.memory_salience
                        }
                    }
                ]
            )
            
            # Store memory metadata in Redis
            await self._store_memory_metadata(persona, event, memory_id, memory_text)
            
            logger.info(f"Stored memory for {persona.baseline.name}: {event.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing event memory: {e}")
            return False
    
    def _create_memory_text(self, event: Event, response: str) -> str:
        """Create memory text from event and response."""
        memory_parts = [
            f"Event: {event.title}",
            f"Description: {event.description}",
            f"Context: {event.context}",
            f"Response: {response}",
            f"Impact: {event.stress_impact}/10",
            f"Day: {event.simulation_day}"
        ]
        
        return "\n".join(memory_parts)
    
    async def _generate_memory_embedding(self, memory_text: str) -> List[float]:
        """Generate embedding for memory text."""
        # This would integrate with the LLM service for actual embedding generation
        # For now, return a placeholder embedding
        import random
        random.seed(hash(memory_text) % 2**32)
        return [random.uniform(-1.0, 1.0) for _ in range(self.embedding_dimension)]
    
    async def _store_memory_metadata(
        self, 
        persona: Persona, 
        event: Event, 
        memory_id: str, 
        memory_text: str
    ) -> None:
        """Store memory metadata in Redis."""
        try:
            metadata = {
                "memory_id": memory_id,
                "event_id": event.event_id,
                "persona_id": persona.state.persona_id,
                "event_type": event.event_type.value,
                "event_category": event.category.value,
                "stress_impact": event.stress_impact,
                "simulation_day": event.simulation_day,
                "memory_salience": event.memory_salience,
                "created_at": datetime.utcnow().isoformat(),
                "memory_text": memory_text[:500]  # Truncate for storage
            }
            
            # Store in Redis with expiration
            await redis_client.set(
                f"memory:{memory_id}",
                json.dumps(metadata),
                expire=86400 * 30  # 30 days
            )
            
        except Exception as e:
            logger.error(f"Error storing memory metadata: {e}")
    
    async def retrieve_similar_memories(
        self, 
        persona: Persona, 
        query_text: str,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve similar memories for a persona."""
        try:
            persona_name = persona.baseline.name.lower()
            collection_name = self.memory_collections.get(persona_name)
            
            if not collection_name:
                logger.error(f"No memory collection found for persona: {persona_name}")
                return []
            
            # Generate query embedding
            query_embedding = await self._generate_memory_embedding(query_text)
            
            # Search for similar memories
            search_result = await qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=similarity_threshold
            )
            
            # Format results
            memories = []
            for point in search_result:
                memory_data = {
                    "memory_id": point.id,
                    "similarity_score": point.score,
                    "memory_text": point.payload.get("memory_text", ""),
                    "event_type": point.payload.get("event_type", ""),
                    "event_category": point.payload.get("event_category", ""),
                    "stress_impact": point.payload.get("stress_impact", 0.0),
                    "simulation_day": point.payload.get("simulation_day", 0),
                    "memory_salience": point.payload.get("memory_salience", 0.5),
                    "created_at": point.payload.get("created_at", "")
                }
                memories.append(memory_data)
            
            logger.info(f"Retrieved {len(memories)} similar memories for {persona.baseline.name}")
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving similar memories: {e}")
            return []
    
    async def retrieve_recent_memories(
        self, 
        persona: Persona, 
        days_back: int = 7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve recent memories for a persona."""
        try:
            persona_name = persona.baseline.name.lower()
            collection_name = self.memory_collections.get(persona_name)
            
            if not collection_name:
                logger.error(f"No memory collection found for persona: {persona_name}")
                return []
            
            # Calculate cutoff day
            current_day = persona.state.simulation_day or 0
            cutoff_day = max(0, current_day - days_back)
            
            # Search for recent memories
            search_result = await qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "simulation_day",
                            "range": {
                                "gte": cutoff_day
                            }
                        }
                    ]
                },
                limit=limit,
                with_payload=True
            )
            
            # Format results
            memories = []
            for point in search_result[0]:
                memory_data = {
                    "memory_id": point.id,
                    "memory_text": point.payload.get("memory_text", ""),
                    "event_type": point.payload.get("event_type", ""),
                    "event_category": point.payload.get("event_category", ""),
                    "stress_impact": point.payload.get("stress_impact", 0.0),
                    "simulation_day": point.payload.get("simulation_day", 0),
                    "memory_salience": point.payload.get("memory_salience", 0.5),
                    "created_at": point.payload.get("created_at", "")
                }
                memories.append(memory_data)
            
            # Sort by simulation day (most recent first)
            memories.sort(key=lambda x: x["simulation_day"], reverse=True)
            
            logger.info(f"Retrieved {len(memories)} recent memories for {persona.baseline.name}")
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving recent memories: {e}")
            return []
    
    async def retrieve_high_impact_memories(
        self, 
        persona: Persona, 
        impact_threshold: float = 5.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve high-impact memories for a persona."""
        try:
            persona_name = persona.baseline.name.lower()
            collection_name = self.memory_collections.get(persona_name)
            
            if not collection_name:
                logger.error(f"No memory collection found for persona: {persona_name}")
                return []
            
            # Search for high-impact memories
            search_result = await qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "stress_impact",
                            "range": {
                                "gte": impact_threshold
                            }
                        }
                    ]
                },
                limit=limit,
                with_payload=True
            )
            
            # Format results
            memories = []
            for point in search_result[0]:
                memory_data = {
                    "memory_id": point.id,
                    "memory_text": point.payload.get("memory_text", ""),
                    "event_type": point.payload.get("event_type", ""),
                    "event_category": point.payload.get("event_category", ""),
                    "stress_impact": point.payload.get("stress_impact", 0.0),
                    "simulation_day": point.payload.get("simulation_day", 0),
                    "memory_salience": point.payload.get("memory_salience", 0.5),
                    "created_at": point.payload.get("created_at", "")
                }
                memories.append(memory_data)
            
            # Sort by stress impact (highest first)
            memories.sort(key=lambda x: x["stress_impact"], reverse=True)
            
            logger.info(f"Retrieved {len(memories)} high-impact memories for {persona.baseline.name}")
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving high-impact memories: {e}")
            return []
    
    async def get_memory_statistics(self, persona: Persona) -> Dict[str, Any]:
        """Get memory statistics for a persona."""
        try:
            persona_name = persona.baseline.name.lower()
            collection_name = self.memory_collections.get(persona_name)
            
            if not collection_name:
                return {"error": f"No memory collection found for persona: {persona_name}"}
            
            # Get collection info
            collection_info = await qdrant_client.get_collection(collection_name)
            
            # Get recent memories for analysis
            recent_memories = await self.retrieve_recent_memories(persona, days_back=30)
            
            # Calculate statistics
            total_memories = collection_info.points_count
            stress_events = len([m for m in recent_memories if m["event_type"] == EventType.STRESS.value])
            neutral_events = len([m for m in recent_memories if m["event_type"] == EventType.NEUTRAL.value])
            minimal_events = len([m for m in recent_memories if m["event_type"] == EventType.MINIMAL.value])
            
            avg_stress_impact = 0.0
            if recent_memories:
                avg_stress_impact = sum(m["stress_impact"] for m in recent_memories) / len(recent_memories)
            
            stats = {
                "persona_name": persona.baseline.name,
                "total_memories": total_memories,
                "recent_memories": len(recent_memories),
                "stress_events": stress_events,
                "neutral_events": neutral_events,
                "minimal_events": minimal_events,
                "average_stress_impact": round(avg_stress_impact, 2),
                "memory_collection": collection_name
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_memories(self, days_to_keep: int = 30) -> bool:
        """Clean up old memories to save space."""
        try:
            for persona_name, collection_name in self.memory_collections.items():
                # Get current simulation day from persona state
                # For now, we'll use a simple approach
                current_day = 30  # This should come from simulation state
                cutoff_day = max(0, current_day - days_to_keep)
                
                # Delete old memories
                await qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector={
                        "filter": {
                            "must": [
                                {
                                    "key": "simulation_day",
                                    "range": {
                                        "lt": cutoff_day
                                    }
                                }
                            ]
                        }
                    }
                )
                
                logger.info(f"Cleaned up old memories for {persona_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
            return False 