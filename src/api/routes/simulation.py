"""
Simulation control and management endpoints.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.core.config import settings
from src.core.logging import get_logger
from src.services.simulation_engine import SimulationEngine
from src.models.simulation import ExperimentalCondition, SimulationConfig

logger = get_logger(__name__)
router = APIRouter(prefix="/simulation", tags=["simulation"])


class SimulationStartRequest(BaseModel):
    """Request model for starting a simulation."""
    config_name: str = "experimental_design"
    experimental_condition: ExperimentalCondition = ExperimentalCondition.CONTROL
    duration_days: Optional[int] = None
    stress_event_frequency: Optional[float] = None
    neutral_event_frequency: Optional[float] = None


class SimulationStatusResponse(BaseModel):
    """Response model for simulation status."""
    simulation_id: Optional[str]
    status: str
    current_day: int
    total_days: int
    progress_percentage: float
    active_personas: int
    events_processed: int
    assessments_completed: int
    start_time: Optional[datetime]
    estimated_completion: Optional[datetime]
    is_running: bool
    is_paused: bool


class SimulationResultsResponse(BaseModel):
    """Response model for simulation results."""
    simulation_id: str
    status: str
    total_personas: int
    total_events: int
    total_assessments: int
    completion_time: Optional[datetime]
    results_summary: Dict[str, Any]


# Global simulation engine instance
simulation_engine = SimulationEngine()

# Import and connect WebSocket manager after it's created
def connect_websocket_manager():
    """Connect WebSocket manager to simulation engine."""
    try:
        from src.api.routes.websocket import websocket_manager
        simulation_engine.set_websocket_manager(websocket_manager)
        logger.info("WebSocket manager connected to simulation engine")
    except ImportError:
        logger.warning("WebSocket manager not available for simulation engine")

# Connect WebSocket manager when module is imported
connect_websocket_manager()


@router.post("/start", response_model=Dict[str, Any])
async def start_simulation(request: SimulationStartRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Start a new simulation."""
    try:
        logger.info(f"Starting simulation with condition: {request.experimental_condition}")
        
        # Initialize simulation
        success = await simulation_engine.initialize_simulation(
            config_name=request.config_name,
            experimental_condition=request.experimental_condition
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to initialize simulation")
        
        # Start simulation in background
        background_tasks.add_task(simulation_engine.run_simulation)
        
        return {
            "message": "Simulation started successfully",
            "simulation_id": simulation_engine.simulation_state.simulation_id if simulation_engine.simulation_state else None,
            "condition": request.experimental_condition.value,
            "status": "running"
        }
        
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start simulation: {str(e)}")


@router.post("/stop")
async def stop_simulation() -> Dict[str, Any]:
    """Stop the current simulation."""
    try:
        success = await simulation_engine.stop_simulation()
        
        if not success:
            raise HTTPException(status_code=400, detail="No active simulation to stop")
        
        return {
            "message": "Simulation stopped successfully",
            "status": "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error stopping simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop simulation: {str(e)}")


@router.post("/pause")
async def pause_simulation() -> Dict[str, Any]:
    """Pause the current simulation."""
    try:
        success = await simulation_engine.pause_simulation()
        
        if not success:
            raise HTTPException(status_code=400, detail="No active simulation to pause")
        
        return {
            "message": "Simulation paused successfully",
            "status": "paused"
        }
        
    except Exception as e:
        logger.error(f"Error pausing simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause simulation: {str(e)}")


@router.post("/resume")
async def resume_simulation() -> Dict[str, Any]:
    """Resume the current simulation."""
    try:
        success = await simulation_engine.resume_simulation()
        
        if not success:
            raise HTTPException(status_code=400, detail="No active simulation to resume")
        
        return {
            "message": "Simulation resumed successfully",
            "status": "running"
        }
        
    except Exception as e:
        logger.error(f"Error resuming simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume simulation: {str(e)}")


@router.get("/status", response_model=SimulationStatusResponse)
async def get_simulation_status() -> SimulationStatusResponse:
    """Get current simulation status."""
    try:
        status_data = await simulation_engine.get_simulation_status()
        
        if not status_data:
            # No active simulation
            return SimulationStatusResponse(
                simulation_id=None,
                status="no_simulation",
                current_day=0,
                total_days=0,
                progress_percentage=0.0,
                active_personas=0,
                events_processed=0,
                assessments_completed=0,
                start_time=None,
                estimated_completion=None,
                is_running=False,
                is_paused=False
            )
        
        return SimulationStatusResponse(**status_data)
        
    except Exception as e:
        logger.error(f"Error getting simulation status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get simulation status: {str(e)}")


@router.get("/results", response_model=SimulationResultsResponse)
async def get_simulation_results() -> SimulationResultsResponse:
    """Get simulation results."""
    try:
        results = await simulation_engine.get_simulation_results()
        
        if not results:
            raise HTTPException(status_code=404, detail="No simulation results available")
        
        return SimulationResultsResponse(**results)
        
    except Exception as e:
        logger.error(f"Error getting simulation results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get simulation results: {str(e)}")


@router.get("/configs")
async def get_available_configs() -> Dict[str, Any]:
    """Get available simulation configurations."""
    try:
        from src.core.config import config_manager
        
        configs = {
            "simulation_configs": config_manager.list_simulation_configs(),
            "persona_configs": config_manager.list_persona_configs(),
            "event_configs": config_manager.list_event_configs()
        }
        
        return configs
        
    except Exception as e:
        logger.error(f"Error getting available configs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configurations: {str(e)}") 