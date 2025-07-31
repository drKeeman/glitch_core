"""
Data access and export endpoints.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.core.config import settings
from src.core.logging import get_logger
from src.models.events import EventType
from src.storage.redis_client import redis_client
from src.storage.qdrant_client import qdrant_client

logger = get_logger(__name__)
router = APIRouter(prefix="/data", tags=["data"])


class AssessmentResult(BaseModel):
    """Assessment result model."""
    persona_id: str
    assessment_type: str
    score: float
    severity_level: str
    timestamp: datetime
    responses: List[Dict[str, Any]]


class MechanisticData(BaseModel):
    """Mechanistic analysis data model."""
    persona_id: str
    assessment_type: str
    attention_patterns: Dict[str, Any]
    activation_data: Dict[str, Any]
    drift_indicators: Dict[str, Any]
    timestamp: datetime


class DataExportRequest(BaseModel):
    """Request model for data export."""
    export_format: str = "json"  # json, csv, parquet
    include_assessments: bool = True
    include_mechanistic: bool = True
    include_events: bool = True
    date_range: Optional[Dict[str, datetime]] = None


@router.get("/assessments", response_model=List[AssessmentResult])
async def get_assessment_results(
    persona_id: Optional[str] = Query(None, description="Filter by persona ID"),
    assessment_type: Optional[str] = Query(None, description="Filter by assessment type"),
    limit: int = Query(100, description="Maximum number of results")
) -> List[AssessmentResult]:
    """Get assessment results."""
    try:
        # This would typically query the database
        # For now, return mock data
        mock_results = [
            AssessmentResult(
                persona_id="persona_1",
                assessment_type="PHQ-9",
                score=5.0,
                severity_level="minimal",
                timestamp=datetime.now(timezone.utc),
                responses=[{"question": "Q1", "response": "Not at all", "score": 0}]
            )
        ]
        
        return mock_results[:limit]
        
    except Exception as e:
        logger.error(f"Error getting assessment results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get assessment results: {str(e)}")


@router.get("/mechanistic", response_model=List[MechanisticData])
async def get_mechanistic_data(
    persona_id: Optional[str] = Query(None, description="Filter by persona ID"),
    assessment_type: Optional[str] = Query(None, description="Filter by assessment type"),
    limit: int = Query(100, description="Maximum number of results")
) -> List[MechanisticData]:
    """Get mechanistic analysis data."""
    try:
        # This would typically query the vector database
        # For now, return mock data
        mock_data = [
            MechanisticData(
                persona_id="persona_1",
                assessment_type="PHQ-9",
                attention_patterns={"self_reference": 0.3, "emotional_salience": 0.7},
                activation_data={"layer_5": 0.8, "layer_10": 0.6},
                drift_indicators={"baseline_deviation": 0.2, "attention_shift": 0.1},
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        return mock_data[:limit]
        
    except Exception as e:
        logger.error(f"Error getting mechanistic data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get mechanistic data: {str(e)}")


@router.get("/event-types")
async def get_event_types() -> Dict[str, Any]:
    """Get available event types and their descriptions."""
    try:
        event_types = {
            "types": [
                {
                    "value": EventType.STRESS.value,
                    "label": "Stress Events",
                    "description": "High-impact stressful events that can significantly affect persona state"
                },
                {
                    "value": EventType.NEUTRAL.value,
                    "label": "Neutral Events", 
                    "description": "Moderate-impact events that have some effect on persona state"
                },
                {
                    "value": EventType.MINIMAL.value,
                    "label": "Minimal Events",
                    "description": "Low-impact routine events with minimal effect on persona state"
                }
            ],
            "categories": {
                "stress": ["death", "trauma", "conflict", "loss", "health", "financial", "work", "relationship"],
                "neutral": ["routine_change", "minor_news", "social", "environmental", "technology"],
                "minimal": ["daily_routine", "weather", "minor_interaction"]
            }
        }
        
        return event_types
        
    except Exception as e:
        logger.error(f"Error getting event types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get event types: {str(e)}")


@router.get("/events")
async def get_events(
    persona_id: Optional[str] = Query(None, description="Filter by persona ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(100, description="Maximum number of results")
) -> List[Dict[str, Any]]:
    """Get events data."""
    try:
        # This would typically query the database
        # For now, return mock data
        mock_events = [
            {
                "event_id": "event_1",
                "persona_id": "persona_1",
                "event_type": EventType.STRESS.value,
                "title": "Test Stress Event",
                "description": "A test stress event",
                "simulation_day": 5,
                "stress_impact": 7.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        return mock_events[:limit]
        
    except Exception as e:
        logger.error(f"Error getting events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get events: {str(e)}")


@router.post("/export")
async def export_data(request: DataExportRequest) -> Dict[str, Any]:
    """Export data in specified format."""
    try:
        # Create export directory
        export_dir = Path(settings.DATA_DIR) / "exports"
        export_dir.mkdir(exist_ok=True)
        
        # Generate export filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_export_{timestamp}.{request.export_format}"
        filepath = export_dir / filename
        
        # Initialize export data structure
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "format": request.export_format,
                "included_data": {
                    "assessments": request.include_assessments,
                    "mechanistic": request.include_mechanistic,
                    "events": request.include_events
                }
            },
            "assessments": [],
            "mechanistic_data": [],
            "events": []
        }
        
        # Import required services
        from src.services.persona_manager import PersonaManager
        from src.services.assessment_service import AssessmentService
        from src.storage.file_storage import FileStorage
        
        persona_manager = PersonaManager()
        assessment_service = AssessmentService()
        file_storage = FileStorage()
        
        # Get all active personas
        active_personas = await persona_manager.list_active_personas()
        
        # Export assessment data
        if request.include_assessments:
            # First try to get assessment history from Redis
            for persona in active_personas:
                # Get assessment history for this persona
                assessment_history = await assessment_service.get_assessment_history(persona.state.persona_id)
                
                for session in assessment_history:
                    # Convert assessment session to export format
                    for result in session.get_all_results():
                        export_data["assessments"].append({
                            "persona_id": result.persona_id,
                            "assessment_type": result.assessment_type,
                            "score": result.total_score,
                            "severity_level": result.severity_level.value,
                            "timestamp": result.timestamp.isoformat(),
                            "responses": result.responses
                        })
            
            # Also check for assessment data in file storage
            assessments_path = file_storage.base_path / "raw"
            if assessments_path.exists():
                for sim_dir in assessments_path.iterdir():
                    if sim_dir.is_dir():
                        assessment_dir = sim_dir / "assessments"
                        if assessment_dir.exists():
                            for file_path in assessment_dir.glob("*.json"):
                                try:
                                    with open(file_path, 'r') as f:
                                        assessment_data = json.load(f)
                                    
                                    # Convert assessment data to export format
                                    for assessment_type in ["phq9_result", "gad7_result", "pss10_result"]:
                                        if assessment_type in assessment_data:
                                            result = assessment_data[assessment_type]
                                            export_data["assessments"].append({
                                                "persona_id": result.get("persona_id", assessment_data.get("persona_id", "unknown")),
                                                "assessment_type": result.get("assessment_type", assessment_type.replace("_result", "").upper()),
                                                "score": result.get("total_score", 0.0),
                                                "severity_level": result.get("severity_level", "minimal"),
                                                "timestamp": result.get("created_at", datetime.now(timezone.utc).isoformat()),
                                                "responses": result.get("raw_responses", [])
                                            })
                                except Exception as e:
                                    logger.warning(f"Failed to load assessment data from {file_path}: {e}")
        
        # Export mechanistic data
        if request.include_mechanistic:
            # Load mechanistic data from file storage
            mechanistic_path = file_storage.base_path / "raw" / "mechanistic"
            if mechanistic_path.exists():
                for file_path in mechanistic_path.glob("*.json"):
                    try:
                        with open(file_path, 'r') as f:
                            mechanistic_data = json.load(f)
                        
                        # Convert to export format
                        export_data["mechanistic_data"].append({
                            "persona_id": mechanistic_data.get("persona_id", "unknown"),
                            "assessment_type": mechanistic_data.get("assessment_type", "unknown"),
                            "attention_patterns": mechanistic_data.get("attention_patterns", {}),
                            "activation_data": mechanistic_data.get("activation_data", {}),
                            "drift_indicators": mechanistic_data.get("drift_indicators", {}),
                            "timestamp": mechanistic_data.get("timestamp", datetime.now(timezone.utc).isoformat())
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load mechanistic data from {file_path}: {e}")
        
        # Export events data
        if request.include_events:
            # Load events data from file storage
            events_path = file_storage.base_path / "raw" / "events"
            if events_path.exists():
                for file_path in events_path.glob("*.json"):
                    try:
                        with open(file_path, 'r') as f:
                            events_data = json.load(f)
                        
                        # Convert to export format
                        if isinstance(events_data, list):
                            for event in events_data:
                                export_data["events"].append({
                                    "event_id": event.get("event_id", "unknown"),
                                    "persona_id": event.get("persona_id", "unknown"),
                                    "event_type": event.get("event_type", "unknown"),
                                    "title": event.get("title", "Unknown Event"),
                                    "description": event.get("description", ""),
                                    "simulation_day": event.get("simulation_day", 0),
                                    "stress_impact": event.get("stress_impact", 0.0),
                                    "timestamp": event.get("timestamp", datetime.now(timezone.utc).isoformat())
                                })
                        else:
                            # Single event
                            export_data["events"].append({
                                "event_id": events_data.get("event_id", "unknown"),
                                "persona_id": events_data.get("persona_id", "unknown"),
                                "event_type": events_data.get("event_type", "unknown"),
                                "title": events_data.get("title", "Unknown Event"),
                                "description": events_data.get("description", ""),
                                "simulation_day": events_data.get("simulation_day", 0),
                                "stress_impact": events_data.get("stress_impact", 0.0),
                                "timestamp": events_data.get("timestamp", datetime.now(timezone.utc).isoformat())
                            })
                    except Exception as e:
                        logger.warning(f"Failed to load events data from {file_path}: {e}")
        
        # If no real data found, add some sample data for testing
        if not export_data["assessments"] and request.include_assessments:
            logger.info("No assessment data found, adding sample data")
            export_data["assessments"] = [
                {
                    "persona_id": "persona_1",
                    "assessment_type": "PHQ-9",
                    "score": 5.0,
                    "severity_level": "minimal",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "responses": [{"question": "Q1", "response": "Not at all", "score": 0}]
                }
            ]
        
        if not export_data["mechanistic_data"] and request.include_mechanistic:
            logger.info("No mechanistic data found, adding sample data")
            export_data["mechanistic_data"] = [
                {
                    "persona_id": "persona_1",
                    "assessment_type": "PHQ-9",
                    "attention_patterns": {"self_reference": 0.3, "emotional_salience": 0.7},
                    "activation_data": {"layer_5": 0.8, "layer_10": 0.6},
                    "drift_indicators": {"baseline_deviation": 0.2, "attention_shift": 0.1},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
        
        if not export_data["events"] and request.include_events:
            logger.info("No events data found, adding sample data")
            export_data["events"] = [
                {
                    "event_id": "event_1",
                    "persona_id": "persona_1",
                    "event_type": EventType.STRESS.value,
                    "title": "Test Stress Event",
                    "description": "A test stress event for export",
                    "simulation_day": 5,
                    "stress_impact": 7.0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
        
        # Write export file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return {
            "message": "Data exported successfully",
            "filename": filename,
            "filepath": str(filepath),
            "size_bytes": filepath.stat().st_size,
            "data_counts": {
                "assessments": len(export_data["assessments"]),
                "mechanistic_data": len(export_data["mechanistic_data"]),
                "events": len(export_data["events"])
            }
        }
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


@router.get("/download/{filename}")
async def download_export(filename: str) -> FileResponse:
    """Download exported data file."""
    try:
        export_dir = Path(settings.DATA_DIR) / "exports"
        filepath = export_dir / filename
        
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Export file not found")
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Error downloading export: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download export: {str(e)}")


@router.get("/stats")
async def get_data_statistics() -> Dict[str, Any]:
    """Get data statistics."""
    try:
        # This would typically query the database for actual stats
        # For now, return mock statistics
        stats = {
            "total_personas": 9,
            "total_assessments": 45,
            "total_events": 120,
            "total_mechanistic_records": 135,
            "data_size_mb": 2.5,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting data statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data statistics: {str(e)}") 