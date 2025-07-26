"""
Persona management API endpoints.
"""

from typing import List
from uuid import UUID, uuid4
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from .models import PersonaConfigRequest, PersonaConfigResponse, ErrorResponse
from glitch_core.config.logging import get_logger

# Create router
router = APIRouter()

# In-memory storage for personas (in production, this would be a database)
personas_db = {}

logger = get_logger("personas_api")


@router.post("/", response_model=PersonaConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_persona(persona_request: PersonaConfigRequest) -> PersonaConfigResponse:
    """Create a new persona configuration."""
    try:
        persona_id = uuid4()
        now = datetime.utcnow()
        
        # Create persona config
        persona_config = PersonaConfigResponse(
            id=persona_id,
            name=persona_request.name,
            persona_type=persona_request.persona_type,
            traits=persona_request.traits,
            cognitive_biases=persona_request.cognitive_biases,
            emotional_baselines=persona_request.emotional_baselines,
            memory_patterns=persona_request.memory_patterns,
            created_at=now,
            updated_at=now
        )
        
        # Store in memory
        personas_db[persona_id] = persona_config
        
        logger.info(
            "persona_created",
            persona_id=str(persona_id),
            persona_type=persona_request.persona_type,
            name=persona_request.name
        )
        
        return persona_config
        
    except Exception as e:
        logger.error("persona_creation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create persona configuration"
        )


@router.get("/{persona_id}", response_model=PersonaConfigResponse)
async def get_persona(persona_id: UUID) -> PersonaConfigResponse:
    """Get a persona configuration by ID."""
    if persona_id not in personas_db:
        logger.warning("persona_not_found", persona_id=str(persona_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Persona configuration not found"
        )
    
    logger.info("persona_retrieved", persona_id=str(persona_id))
    return personas_db[persona_id]


@router.get("/", response_model=List[PersonaConfigResponse])
async def list_personas() -> List[PersonaConfigResponse]:
    """List all persona configurations."""
    personas = list(personas_db.values())
    logger.info("personas_listed", count=len(personas))
    return personas


@router.put("/{persona_id}", response_model=PersonaConfigResponse)
async def update_persona(
    persona_id: UUID, 
    persona_request: PersonaConfigRequest
) -> PersonaConfigResponse:
    """Update a persona configuration."""
    if persona_id not in personas_db:
        logger.warning("persona_not_found_for_update", persona_id=str(persona_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Persona configuration not found"
        )
    
    try:
        # Update persona config
        existing_persona = personas_db[persona_id]
        updated_persona = PersonaConfigResponse(
            id=persona_id,
            name=persona_request.name,
            persona_type=persona_request.persona_type,
            traits=persona_request.traits,
            cognitive_biases=persona_request.cognitive_biases,
            emotional_baselines=persona_request.emotional_baselines,
            memory_patterns=persona_request.memory_patterns,
            created_at=existing_persona.created_at,
            updated_at=datetime.utcnow()
        )
        
        personas_db[persona_id] = updated_persona
        
        logger.info(
            "persona_updated",
            persona_id=str(persona_id),
            persona_type=persona_request.persona_type
        )
        
        return updated_persona
        
    except Exception as e:
        logger.error("persona_update_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update persona configuration"
        )


@router.delete("/{persona_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_persona(persona_id: UUID):
    """Delete a persona configuration."""
    if persona_id not in personas_db:
        logger.warning("persona_not_found_for_delete", persona_id=str(persona_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Persona configuration not found"
        )
    
    del personas_db[persona_id]
    logger.info("persona_deleted", persona_id=str(persona_id))


@router.get("/types/available", response_model=List[str])
async def get_available_persona_types() -> List[str]:
    """Get list of available persona types."""
    # This would typically come from the personality system
    available_types = [
        "resilient_optimist",
        "anxious_overthinker", 
        "stoic_philosopher",
        "creative_volatile",
        "balanced_baseline"
    ]
    
    logger.info("available_persona_types_retrieved", count=len(available_types))
    return available_types 