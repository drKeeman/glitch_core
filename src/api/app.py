"""
FastAPI application factory and configuration.
"""

import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from src.api.routes import monitoring, simulation, data, websocket
from src.core.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="AI Personality Drift Simulation with Mechanistic Interpretability",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files for frontend (before API routes)
    frontend_path = Path("frontend")
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory="frontend"), name="frontend")
    
    # Root route to serve frontend
    @app.get("/")
    async def root():
        """Serve the frontend application."""
        frontend_path = Path("frontend/index.html")
        if frontend_path.exists():
            return FileResponse(frontend_path)
        return {"message": "Frontend not found"}
    
    # Include routers
    app.include_router(monitoring.router, prefix=settings.API_PREFIX)
    app.include_router(simulation.router, prefix=settings.API_PREFIX)
    app.include_router(data.router, prefix=settings.API_PREFIX)
    app.include_router(websocket.router, prefix=settings.API_PREFIX)
    
    @app.on_event("startup")
    async def startup_event():
        """Start background tasks on application startup."""
        # Start periodic status updates
        asyncio.create_task(websocket.periodic_status_updates())
    
    return app 