"""
Experiment management API endpoints.
"""

import asyncio
from typing import List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from .models import ExperimentRequest, ExperimentResponse, ErrorResponse
from glitch_core.config.logging import get_logger

# Create router
router = APIRouter()

# In-memory storage for experiments (in production, this would be a database)
experiments_db = {}
experiment_results_db = {}

logger = get_logger("experiments_api")


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    experiment_request: ExperimentRequest,
    background_tasks: BackgroundTasks
) -> ExperimentResponse:
    """Create and start a new drift simulation experiment."""
    try:
        experiment_id = uuid4()
        now = datetime.utcnow()
        
        # Create experiment record
        experiment = ExperimentResponse(
            id=experiment_id,
            persona_id=experiment_request.persona_id,
            drift_profile=experiment_request.drift_profile,
            epochs=experiment_request.epochs,
            events_per_epoch=experiment_request.events_per_epoch,
            seed=experiment_request.seed,
            status="running",
            current_epoch=0,
            created_at=now,
            started_at=now,
            completed_at=None
        )
        
        # Store in memory
        experiments_db[experiment_id] = experiment
        
        # Start simulation in background
        background_tasks.add_task(
            run_simulation_background,
            experiment_id,
            experiment_request
        )
        
        logger.info(
            "experiment_created",
            experiment_id=str(experiment_id),
            persona_id=str(experiment_request.persona_id),
            epochs=experiment_request.epochs
        )
        
        return experiment
        
    except Exception as e:
        logger.error("experiment_creation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create experiment"
        )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: UUID) -> ExperimentResponse:
    """Get experiment status and details."""
    if experiment_id not in experiments_db:
        logger.warning("experiment_not_found", experiment_id=str(experiment_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    logger.info("experiment_retrieved", experiment_id=str(experiment_id))
    return experiments_db[experiment_id]


@router.get("/", response_model=List[ExperimentResponse])
async def list_experiments() -> List[ExperimentResponse]:
    """List all experiments."""
    experiments = list(experiments_db.values())
    logger.info("experiments_listed", count=len(experiments))
    return experiments


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def stop_experiment(experiment_id: UUID):
    """Stop a running experiment."""
    if experiment_id not in experiments_db:
        logger.warning("experiment_not_found_for_stop", experiment_id=str(experiment_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    experiment = experiments_db[experiment_id]
    if experiment.status == "running":
        experiment.status = "stopped"
        experiment.completed_at = datetime.utcnow()
        
        logger.info("experiment_stopped", experiment_id=str(experiment_id))
    else:
        logger.warning("experiment_already_stopped", experiment_id=str(experiment_id))


@router.get("/{experiment_id}/results")
async def get_experiment_results(experiment_id: UUID) -> Dict[str, Any]:
    """Get experiment results and analysis."""
    if experiment_id not in experiments_db:
        logger.warning("experiment_not_found_for_results", experiment_id=str(experiment_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    if experiment_id not in experiment_results_db:
        logger.warning("experiment_results_not_found", experiment_id=str(experiment_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment results not available"
        )
    
    logger.info("experiment_results_retrieved", experiment_id=str(experiment_id))
    return experiment_results_db[experiment_id]


async def run_simulation_background(experiment_id: UUID, experiment_request: ExperimentRequest):
    """Run simulation in background task."""
    try:
        logger.info("simulation_started", experiment_id=str(experiment_id))
        
        # Simulate running epochs
        experiment = experiments_db[experiment_id]
        
        for epoch in range(experiment.epochs):
            # Update current epoch
            experiment.current_epoch = epoch
            
            # Simulate some processing time
            await asyncio.sleep(0.1)
            
            # Check if experiment was stopped
            if experiment.status == "stopped":
                logger.info("simulation_stopped", experiment_id=str(experiment_id))
                return
        
        # Mark as completed
        experiment.status = "completed"
        experiment.completed_at = datetime.utcnow()
        
        # Generate mock results
        results = generate_mock_results(experiment_id, experiment_request)
        experiment_results_db[experiment_id] = results
        
        logger.info("simulation_completed", experiment_id=str(experiment_id))
        
    except Exception as e:
        logger.error("simulation_failed", experiment_id=str(experiment_id), error=str(e))
        experiment = experiments_db[experiment_id]
        experiment.status = "failed"
        experiment.completed_at = datetime.utcnow()


def generate_mock_results(experiment_id: UUID, experiment_request: ExperimentRequest) -> Dict[str, Any]:
    """Generate mock simulation results for demonstration."""
    import random
    import numpy as np
    
    # Set seed for reproducibility
    if experiment_request.seed:
        random.seed(experiment_request.seed)
        np.random.seed(experiment_request.seed)
    
    epochs = experiment_request.epochs
    events_per_epoch = experiment_request.events_per_epoch
    
    # Generate mock epoch data
    epochs_data = []
    for epoch in range(epochs):
        # Generate emotional state
        emotional_state = {
            "joy": max(0, min(1, 0.5 + np.random.normal(0, 0.1))),
            "anxiety": max(0, min(1, 0.3 + np.random.normal(0, 0.15))),
            "anger": max(0, min(1, 0.2 + np.random.normal(0, 0.1))),
            "sadness": max(0, min(1, 0.2 + np.random.normal(0, 0.1))),
            "excitement": max(0, min(1, 0.4 + np.random.normal(0, 0.1)))
        }
        
        # Generate stability score
        stability_score = max(0, min(1, 0.7 + np.random.normal(0, 0.1)))
        
        # Generate attention metrics
        attention_metrics = {
            "focus": max(0, min(1, 0.6 + np.random.normal(0, 0.1))),
            "distraction": max(0, min(1, 0.3 + np.random.normal(0, 0.1))),
            "engagement": max(0, min(1, 0.5 + np.random.normal(0, 0.1)))
        }
        
        epochs_data.append({
            "epoch": epoch,
            "emotional_state": emotional_state,
            "stability_score": stability_score,
            "attention_metrics": attention_metrics,
            "events_processed": events_per_epoch,
            "reflections_generated": random.randint(1, 3)
        })
    
    # Generate mock interventions
    interventions = []
    if epochs > 20:
        # Add some interventions
        intervention_epochs = [epochs // 4, epochs // 2, 3 * epochs // 4]
        for i, epoch in enumerate(intervention_epochs):
            interventions.append({
                "id": str(uuid4()),
                "epoch": epoch,
                "event_type": ["trauma_injection", "stress_test", "recovery_event"][i % 3],
                "intensity": 5.0 + i * 1.5,
                "description": f"Intervention {i+1} at epoch {epoch}"
            })
    
    return {
        "experiment_id": str(experiment_id),
        "persona_id": str(experiment_request.persona_id),
        "drift_profile": experiment_request.drift_profile,
        "epochs": epochs_data,
        "interventions": interventions,
        "summary": {
            "total_epochs": epochs,
            "total_events": epochs * events_per_epoch,
            "final_stability": epochs_data[-1]["stability_score"] if epochs_data else 0.0,
            "avg_emotional_volatility": np.mean([
                np.std(list(epoch["emotional_state"].values())) 
                for epoch in epochs_data
            ]) if epochs_data else 0.0
        }
    } 