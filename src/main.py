"""
FastAPI application entry point for AI Personality Drift Simulation.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.app import create_app
from src.core.config import settings
from src.core.logging import setup_logging, get_logger

# Setup logging first, before creating any loggers
setup_logging()

logger = get_logger(__name__)
logger.info("Starting AI Personality Drift Simulation")

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    ) 