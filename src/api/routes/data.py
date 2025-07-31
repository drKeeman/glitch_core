"""
Data access and export endpoints.
"""

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
        
        # Mock export data
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
        
        # Add mock data based on request
        if request.include_assessments:
            export_data["assessments"] = [
                {
                    "persona_id": "persona_1",
                    "assessment_type": "PHQ-9",
                    "score": 5.0,
                    "severity_level": "minimal",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
        
        if request.include_mechanistic:
            export_data["mechanistic_data"] = [
                {
                    "persona_id": "persona_1",
                    "assessment_type": "PHQ-9",
                    "attention_patterns": {"self_reference": 0.3},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
        
        if request.include_events:
            export_data["events"] = [
                {
                    "event_id": "event_1",
                    "persona_id": "persona_1",
                    "event_type": EventType.STRESS.value,
                    "title": "Test Event",
                    "simulation_day": 5
                }
            ]
        
        # Write export file
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return {
            "message": "Data exported successfully",
            "filename": filename,
            "filepath": str(filepath),
            "size_bytes": filepath.stat().st_size
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