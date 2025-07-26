"""
Intervention management API endpoints.
"""

from typing import List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from .models import InterventionRequest, InterventionResponse, ErrorResponse
from glitch_core.config.logging import get_logger

# Create router
router = APIRouter()

# In-memory storage for interventions (in production, this would be a database)
interventions_db = {}

logger = get_logger("interventions_api")


@router.post("/", response_model=InterventionResponse, status_code=status.HTTP_201_CREATED)
async def create_intervention(intervention_request: InterventionRequest) -> InterventionResponse:
    """Inject an intervention into a running experiment."""
    try:
        # Import here to avoid circular imports
        from .experiments import experiments_db
        
        experiment_id = intervention_request.experiment_id
        
        # Check if experiment exists and is running
        if experiment_id not in experiments_db:
            logger.warning("experiment_not_found_for_intervention", experiment_id=str(experiment_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        experiment = experiments_db[experiment_id]
        if experiment.status != "running":
            logger.warning("experiment_not_running_for_intervention", 
                         experiment_id=str(experiment_id), status=experiment.status)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only inject interventions into running experiments"
            )
        
        # Create intervention record
        intervention_id = uuid4()
        now = datetime.utcnow()
        
        intervention = InterventionResponse(
            id=intervention_id,
            experiment_id=experiment_id,
            event_type=intervention_request.event_type,
            intensity=intervention_request.intensity,
            description=intervention_request.description,
            applied_at=now,
            impact_score=None  # Will be calculated after analysis
        )
        
        # Store intervention
        interventions_db[intervention_id] = intervention
        
        # Add intervention to experiment results
        await add_intervention_to_experiment(experiment_id, intervention)
        
        logger.info(
            "intervention_created",
            intervention_id=str(intervention_id),
            experiment_id=str(experiment_id),
            event_type=intervention_request.event_type,
            intensity=intervention_request.intensity
        )
        
        return intervention
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("intervention_creation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create intervention"
        )


@router.get("/{intervention_id}", response_model=InterventionResponse)
async def get_intervention(intervention_id: UUID) -> InterventionResponse:
    """Get intervention details by ID."""
    if intervention_id not in interventions_db:
        logger.warning("intervention_not_found", intervention_id=str(intervention_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Intervention not found"
        )
    
    logger.info("intervention_retrieved", intervention_id=str(intervention_id))
    return interventions_db[intervention_id]


@router.get("/", response_model=List[InterventionResponse])
async def list_interventions() -> List[InterventionResponse]:
    """List all interventions."""
    interventions = list(interventions_db.values())
    logger.info("interventions_listed", count=len(interventions))
    return interventions


@router.get("/experiment/{experiment_id}", response_model=List[InterventionResponse])
async def get_experiment_interventions(experiment_id: UUID) -> List[InterventionResponse]:
    """Get all interventions for a specific experiment."""
    experiment_interventions = [
        intervention for intervention in interventions_db.values()
        if intervention.experiment_id == experiment_id
    ]
    
    logger.info("experiment_interventions_retrieved", 
               experiment_id=str(experiment_id), count=len(experiment_interventions))
    return experiment_interventions


@router.delete("/{intervention_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_intervention(intervention_id: UUID):
    """Delete an intervention (for cleanup purposes)."""
    if intervention_id not in interventions_db:
        logger.warning("intervention_not_found_for_delete", intervention_id=str(intervention_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Intervention not found"
        )
    
    del interventions_db[intervention_id]
    logger.info("intervention_deleted", intervention_id=str(intervention_id))


@router.get("/types/available", response_model=List[str])
async def get_available_intervention_types() -> List[str]:
    """Get list of available intervention types."""
    available_types = [
        "trauma_injection",
        "stress_test", 
        "recovery_event",
        "cognitive_load",
        "emotional_shock",
        "social_rejection",
        "success_achievement",
        "failure_experience",
        "environmental_change",
        "relationship_breakdown"
    ]
    
    logger.info("available_intervention_types_retrieved", count=len(available_types))
    return available_types


@router.get("/{intervention_id}/impact")
async def get_intervention_impact(intervention_id: UUID) -> Dict[str, Any]:
    """Get detailed impact analysis for an intervention."""
    if intervention_id not in interventions_db:
        logger.warning("intervention_not_found_for_impact", intervention_id=str(intervention_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Intervention not found"
        )
    
    intervention = interventions_db[intervention_id]
    
    try:
        # Get experiment results and analyze impact
        from .experiments import experiment_results_db
        
        experiment_id = intervention.experiment_id
        if experiment_id not in experiment_results_db:
            logger.warning("experiment_results_not_found_for_impact", experiment_id=str(experiment_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment results not available for impact analysis"
            )
        
        # Calculate impact metrics
        impact_analysis = calculate_intervention_impact(intervention, experiment_id)
        
        logger.info("intervention_impact_analyzed", intervention_id=str(intervention_id))
        
        return {
            "intervention_id": str(intervention_id),
            "experiment_id": str(experiment_id),
            "impact_analysis": impact_analysis,
            "intervention_details": {
                "event_type": intervention.event_type,
                "intensity": intervention.intensity,
                "description": intervention.description,
                "applied_at": intervention.applied_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("intervention_impact_analysis_failed", intervention_id=str(intervention_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze intervention impact"
        )


async def add_intervention_to_experiment(experiment_id: UUID, intervention: InterventionResponse):
    """Add intervention to experiment results for analysis."""
    try:
        from .experiments import experiment_results_db
        
        if experiment_id in experiment_results_db:
            # Add intervention to existing results
            experiment_results_db[experiment_id]["interventions"].append({
                "id": str(intervention.id),
                "epoch": intervention.applied_at.timestamp(),  # Convert to epoch for simulation
                "event_type": intervention.event_type,
                "intensity": intervention.intensity,
                "description": intervention.description
            })
        else:
            # Create new intervention list for experiment
            experiment_results_db[experiment_id] = {
                "interventions": [{
                    "id": str(intervention.id),
                    "epoch": intervention.applied_at.timestamp(),
                    "event_type": intervention.event_type,
                    "intensity": intervention.intensity,
                    "description": intervention.description
                }]
            }
        
        logger.info("intervention_added_to_experiment", 
                   experiment_id=str(experiment_id), intervention_id=str(intervention.id))
        
    except Exception as e:
        logger.error("failed_to_add_intervention_to_experiment", 
                    experiment_id=str(experiment_id), error=str(e))


def calculate_intervention_impact(intervention: InterventionResponse, experiment_id: UUID) -> Dict[str, Any]:
    """Calculate impact metrics for an intervention."""
    try:
        from .experiments import experiment_results_db
        
        if experiment_id not in experiment_results_db:
            return {"error": "Experiment results not available"}
        
        simulation_data = experiment_results_db[experiment_id]
        epochs = simulation_data.get("epochs", [])
        
        if not epochs:
            return {"error": "No epoch data available"}
        
        # Find the epoch closest to when intervention was applied
        intervention_timestamp = intervention.applied_at.timestamp()
        closest_epoch = min(epochs, key=lambda e: abs(e.get("epoch", 0) - intervention_timestamp))
        
        # Calculate pre and post intervention metrics
        intervention_epoch = closest_epoch.get("epoch", 0)
        pre_intervention = epochs[:intervention_epoch] if intervention_epoch > 0 else []
        post_intervention = epochs[intervention_epoch:] if intervention_epoch < len(epochs) else []
        
        # Calculate impact metrics
        impact_metrics = {
            "intervention_epoch": intervention_epoch,
            "pre_intervention_stability": np.mean([e.get("stability_score", 0) for e in pre_intervention]) if pre_intervention else 0,
            "post_intervention_stability": np.mean([e.get("stability_score", 0) for e in post_intervention]) if post_intervention else 0,
            "stability_change": 0,
            "emotional_volatility_change": 0,
            "recovery_time": None
        }
        
        if pre_intervention and post_intervention:
            impact_metrics["stability_change"] = (
                impact_metrics["post_intervention_stability"] - impact_metrics["pre_intervention_stability"]
            )
            
            # Calculate emotional volatility change
            pre_volatility = np.mean([
                np.std(list(e.get("emotional_state", {}).values()))
                for e in pre_intervention
            ]) if pre_intervention else 0
            
            post_volatility = np.mean([
                np.std(list(e.get("emotional_state", {}).values()))
                for e in post_intervention
            ]) if post_intervention else 0
            
            impact_metrics["emotional_volatility_change"] = post_volatility - pre_volatility
            
            # Estimate recovery time (time to return to pre-intervention stability)
            for i, epoch in enumerate(post_intervention):
                if epoch.get("stability_score", 0) >= impact_metrics["pre_intervention_stability"]:
                    impact_metrics["recovery_time"] = i
                    break
        
        return impact_metrics
        
    except Exception as e:
        logger.error("intervention_impact_calculation_failed", error=str(e))
        return {"error": f"Failed to calculate impact: {str(e)}"}


# Import numpy for calculations
import numpy as np 