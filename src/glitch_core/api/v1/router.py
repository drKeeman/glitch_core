"""
API v1 router with all endpoints.
"""

from fastapi import APIRouter

from .personas import router as personas_router
from .experiments import router as experiments_router
from .analysis import router as analysis_router
from .interventions import router as interventions_router

# Create main v1 router
router = APIRouter()

# Include all sub-routers
router.include_router(personas_router, prefix="/personas", tags=["personas"])
router.include_router(experiments_router, prefix="/experiments", tags=["experiments"])
router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
router.include_router(interventions_router, prefix="/interventions", tags=["interventions"]) 