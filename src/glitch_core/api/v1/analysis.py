"""
Analysis API endpoints for retrieving interpretability insights.
"""

from typing import Dict, Any, List
from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from .models import AnalysisResult, ErrorResponse
from glitch_core.core.analysis import TemporalAnalyzer
from glitch_core.config.logging import get_logger

# Create router
router = APIRouter()

# Initialize temporal analyzer
temporal_analyzer = TemporalAnalyzer()

logger = get_logger("analysis_api")


@router.get("/{experiment_id}", response_model=AnalysisResult)
async def get_analysis(experiment_id: UUID) -> AnalysisResult:
    """Get comprehensive analysis results for an experiment."""
    try:
        # Import here to avoid circular imports
        from .experiments import experiment_results_db
        
        if experiment_id not in experiment_results_db:
            logger.warning("experiment_results_not_found", experiment_id=str(experiment_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment results not available for analysis"
            )
        
        # Get experiment results
        simulation_data = experiment_results_db[experiment_id]
        
        # Run temporal analysis
        analysis_result = temporal_analyzer.analyze_drift_patterns(simulation_data)
        
        # Convert to API response format
        api_response = AnalysisResult(
            experiment_id=experiment_id,
            emergence_points=analysis_result.emergence_points,
            stability_boundaries=analysis_result.stability_boundaries,
            intervention_leverage=analysis_result.intervention_leverage,
            attention_evolution=analysis_result.attention_evolution,
            drift_patterns=analysis_result.drift_patterns,
            created_at=analysis_result.created_at
        )
        
        logger.info("analysis_completed", experiment_id=str(experiment_id))
        
        return api_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("analysis_failed", experiment_id=str(experiment_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze experiment results"
        )


@router.get("/{experiment_id}/emergence")
async def get_emergence_points(experiment_id: UUID) -> Dict[str, Any]:
    """Get pattern emergence points for an experiment."""
    try:
        from .experiments import experiment_results_db
        
        if experiment_id not in experiment_results_db:
            logger.warning("experiment_results_not_found", experiment_id=str(experiment_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment results not available"
            )
        
        simulation_data = experiment_results_db[experiment_id]
        analysis_result = temporal_analyzer.analyze_drift_patterns(simulation_data)
        
        logger.info("emergence_points_retrieved", experiment_id=str(experiment_id))
        
        return {
            "experiment_id": str(experiment_id),
            "emergence_points": analysis_result.emergence_points,
            "total_emergence_points": len(analysis_result.emergence_points)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("emergence_analysis_failed", experiment_id=str(experiment_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze emergence points"
        )


@router.get("/{experiment_id}/stability")
async def get_stability_analysis(experiment_id: UUID) -> Dict[str, Any]:
    """Get stability boundary analysis for an experiment."""
    try:
        from .experiments import experiment_results_db
        
        if experiment_id not in experiment_results_db:
            logger.warning("experiment_results_not_found", experiment_id=str(experiment_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment results not available"
            )
        
        simulation_data = experiment_results_db[experiment_id]
        analysis_result = temporal_analyzer.analyze_drift_patterns(simulation_data)
        
        logger.info("stability_analysis_retrieved", experiment_id=str(experiment_id))
        
        return {
            "experiment_id": str(experiment_id),
            "stability_boundaries": analysis_result.stability_boundaries,
            "total_boundaries": len(analysis_result.stability_boundaries)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("stability_analysis_failed", experiment_id=str(experiment_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze stability boundaries"
        )


@router.get("/{experiment_id}/interventions")
async def get_intervention_analysis(experiment_id: UUID) -> Dict[str, Any]:
    """Get intervention impact analysis for an experiment."""
    try:
        from .experiments import experiment_results_db
        
        if experiment_id not in experiment_results_db:
            logger.warning("experiment_results_not_found", experiment_id=str(experiment_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment results not available"
            )
        
        simulation_data = experiment_results_db[experiment_id]
        analysis_result = temporal_analyzer.analyze_drift_patterns(simulation_data)
        
        logger.info("intervention_analysis_retrieved", experiment_id=str(experiment_id))
        
        return {
            "experiment_id": str(experiment_id),
            "intervention_leverage": analysis_result.intervention_leverage,
            "total_interventions": len(analysis_result.intervention_leverage)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("intervention_analysis_failed", experiment_id=str(experiment_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze intervention impacts"
        )


@router.get("/{experiment_id}/attention")
async def get_attention_analysis(experiment_id: UUID) -> Dict[str, Any]:
    """Get attention evolution analysis for an experiment."""
    try:
        from .experiments import experiment_results_db
        
        if experiment_id not in experiment_results_db:
            logger.warning("experiment_results_not_found", experiment_id=str(experiment_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment results not available"
            )
        
        simulation_data = experiment_results_db[experiment_id]
        analysis_result = temporal_analyzer.analyze_drift_patterns(simulation_data)
        
        logger.info("attention_analysis_retrieved", experiment_id=str(experiment_id))
        
        return {
            "experiment_id": str(experiment_id),
            "attention_evolution": analysis_result.attention_evolution,
            "total_evolution_points": len(analysis_result.attention_evolution)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("attention_analysis_failed", experiment_id=str(experiment_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze attention evolution"
        )


@router.get("/{experiment_id}/patterns")
async def get_drift_patterns(experiment_id: UUID) -> Dict[str, Any]:
    """Get overall drift pattern analysis for an experiment."""
    try:
        from .experiments import experiment_results_db
        
        if experiment_id not in experiment_results_db:
            logger.warning("experiment_results_not_found", experiment_id=str(experiment_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment results not available"
            )
        
        simulation_data = experiment_results_db[experiment_id]
        analysis_result = temporal_analyzer.analyze_drift_patterns(simulation_data)
        
        logger.info("drift_patterns_retrieved", experiment_id=str(experiment_id))
        
        return {
            "experiment_id": str(experiment_id),
            "drift_patterns": analysis_result.drift_patterns,
            "total_patterns": len(analysis_result.drift_patterns)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("drift_patterns_analysis_failed", experiment_id=str(experiment_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze drift patterns"
        )


@router.get("/{experiment_id}/summary")
async def get_analysis_summary(experiment_id: UUID) -> Dict[str, Any]:
    """Get a summary of all analysis results for an experiment."""
    try:
        from .experiments import experiment_results_db
        
        if experiment_id not in experiment_results_db:
            logger.warning("experiment_results_not_found", experiment_id=str(experiment_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment results not available"
            )
        
        simulation_data = experiment_results_db[experiment_id]
        analysis_result = temporal_analyzer.analyze_drift_patterns(simulation_data)
        
        # Create summary
        summary = {
            "experiment_id": str(experiment_id),
            "analysis_timestamp": analysis_result.created_at.isoformat(),
            "emergence_points_count": len(analysis_result.emergence_points),
            "stability_boundaries_count": len(analysis_result.stability_boundaries),
            "intervention_leverage_count": len(analysis_result.intervention_leverage),
            "attention_evolution_count": len(analysis_result.attention_evolution),
            "drift_patterns_count": len(analysis_result.drift_patterns),
            "key_insights": []
        }
        
        # Add key insights
        if analysis_result.emergence_points:
            summary["key_insights"].append({
                "type": "emergence",
                "count": len(analysis_result.emergence_points),
                "description": f"Detected {len(analysis_result.emergence_points)} pattern emergence points"
            })
        
        if analysis_result.stability_boundaries:
            summary["key_insights"].append({
                "type": "stability",
                "count": len(analysis_result.stability_boundaries),
                "description": f"Identified {len(analysis_result.stability_boundaries)} stability boundaries"
            })
        
        if analysis_result.intervention_leverage:
            summary["key_insights"].append({
                "type": "intervention",
                "count": len(analysis_result.intervention_leverage),
                "description": f"Analyzed {len(analysis_result.intervention_leverage)} intervention impacts"
            })
        
        logger.info("analysis_summary_retrieved", experiment_id=str(experiment_id))
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("analysis_summary_failed", experiment_id=str(experiment_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate analysis summary"
        ) 