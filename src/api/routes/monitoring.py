"""
Monitoring and health check endpoints.
"""

from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["monitoring"])


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    app_name: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    logger.info("Health check requested")
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=settings.VERSION,
        app_name=settings.APP_NAME,
    )


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Application status endpoint."""
    logger.info("Status check requested")
    
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "debug": settings.DEBUG,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "redis_url": settings.REDIS_URL,
            "qdrant_url": settings.QDRANT_URL,
            "model_path": settings.MODEL_PATH,
        }
    } 