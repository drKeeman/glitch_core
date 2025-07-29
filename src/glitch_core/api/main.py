"""
Main FastAPI application for Glitch Core.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

from glitch_core.config import get_settings
from glitch_core.config.logging import setup_logging, get_logger, api_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger = get_logger("startup")
    logger.info("application_starting", version="0.1.0")
    
    settings = get_settings()
    logger.info(
        "configuration_loaded",
        environment=settings.ENV,
        api_prefix=settings.API_PREFIX,
        log_level=settings.LOG_LEVEL
    )
    
    # Setup logging
    setup_logging(settings.LOG_LEVEL)
    logger.info("logging_configured", log_level=settings.LOG_LEVEL)
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Glitch Core",
        description="Temporal AI interpretability engine for tracking personality evolution and drift patterns",
        version="0.1.0",
        docs_url="/docs" if settings.ENV != "production" else None,
        redoc_url="/redoc" if settings.ENV != "production" else None,
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Log request start
        api_logger.request_start(
            method=request.method,
            path=str(request.url.path),
            client_ip=request.client.host if request.client else None
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request completion
        api_logger.request_complete(
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms
        )
        
        return response
    
    # Health check endpoint
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        logger = get_logger("health")
        logger.info("health_check_requested")
        
        # Check individual services
        checks = {}
        
        # Check Qdrant
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{settings.QDRANT_URL}/")
                checks["qdrant"] = response.status_code == 200
        except Exception:
            checks["qdrant"] = False
        
        # Check Redis
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url(settings.REDIS_URL)
            await redis_client.ping()
            checks["redis"] = True
            await redis_client.close()
        except Exception:
            checks["redis"] = False
        
        # Check Ollama
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{settings.OLLAMA_URL}/api/tags")
                checks["ollama"] = response.status_code == 200
        except Exception:
            checks["ollama"] = False
        
        # API is always healthy if we reach this point
        checks["api"] = True
        
        # Overall status
        overall_status = "healthy" if all(checks.values()) else "unhealthy"
        
        return {
            "status": overall_status,
            "checks": checks,
            "version": "0.1.0",
            "environment": settings.ENV,
            "timestamp": datetime.now().isoformat()
        }
    
    # Root endpoint
    @app.get("/")
    async def root() -> Dict[str, str]:
        """Root endpoint."""
        logger = get_logger("api")
        logger.info("root_endpoint_accessed")
        
        return {
            "message": "Glitch Core API",
            "version": "0.1.0",
            "docs": "/docs" if settings.ENV != "production" else "API documentation disabled in production",
        }
    
    # Include API v1 router
    from glitch_core.api.v1.router import router as api_v1_router
    app.include_router(api_v1_router, prefix=settings.API_PREFIX)
    
    # WebSocket endpoints
    from glitch_core.api.v1.websocket import websocket_endpoint, get_websocket_test_page
    
    @app.websocket("/ws/experiments/{experiment_id}")
    async def websocket_experiment_updates(websocket: WebSocket, experiment_id: str):
        """WebSocket endpoint for experiment real-time updates."""
        await websocket_endpoint(websocket, experiment_id)
    
    @app.get("/ws/test/{experiment_id}")
    async def websocket_test_page(experiment_id: str) -> HTMLResponse:
        """Test page for WebSocket connections."""
        html_content = get_websocket_test_page(experiment_id)
        return HTMLResponse(content=html_content)
    
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "glitch_core.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    ) 