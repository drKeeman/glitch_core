"""
Assessment service for psychiatric scale administration and scoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.models.persona import Persona
from src.models.assessment import AssessmentSession, PHQ9Result, GAD7Result, PSS10Result
from src.services.llm_service import llm_service
from src.services.persona_manager import persona_manager


logger = logging.getLogger(__name__)


class AssessmentService:
    """Service for conducting psychiatric assessments with personas."""
    
    def __init__(self):
        """Initialize assessment service."""
        self.assessment_types = ["phq9", "gad7", "pss10"]
        
    async def conduct_full_assessment(self, persona: Persona) -> Optional[AssessmentSession]:
        """Conduct complete assessment session with all scales."""
        try:
            logger.info(f"Starting full assessment for {persona.baseline.name}")
            
            # Create assessment session
            session = await persona_manager.create_assessment_session(persona)
            
            # Conduct each assessment type
            for assessment_type in self.assessment_types:
                try:
                    logger.info(f"Conducting {assessment_type.upper()} assessment...")
                    
                    # Conduct assessment
                    result = await llm_service.conduct_assessment(persona, assessment_type)
                    
                    if result:
                        # Update session with result
                        await persona_manager.update_assessment_session(session, result)
                        
                        # Update persona state with assessment results
                        await self._update_persona_with_assessment(persona, result)
                        
                        logger.info(f"Completed {assessment_type.upper()} assessment: {result.total_score}")
                    else:
                        logger.error(f"Failed to complete {assessment_type.upper()} assessment")
                        
                except Exception as e:
                    logger.error(f"Error in {assessment_type.upper()} assessment: {e}")
                    continue
                
                # Small delay between assessments
                await asyncio.sleep(0.5)
            
            # Save session to storage
            await persona_manager.save_assessment_session(session)
            
            # Update persona's last assessment day
            await persona_manager.update_persona_state(
                persona,
                last_assessment_day=persona.state.simulation_day
            )
            
            logger.info(f"Completed full assessment session for {persona.baseline.name}")
            return session
            
        except Exception as e:
            logger.error(f"Error conducting full assessment: {e}")
            return None
    
    async def conduct_single_assessment(self, persona: Persona, 
                                     assessment_type: str) -> Optional[PHQ9Result | GAD7Result | PSS10Result]:
        """Conduct single assessment type."""
        try:
            if assessment_type not in self.assessment_types:
                logger.error(f"Unknown assessment type: {assessment_type}")
                return None
            
            logger.info(f"Conducting {assessment_type.upper()} assessment for {persona.baseline.name}")
            
            # Conduct assessment
            result = await llm_service.conduct_assessment(persona, assessment_type)
            
            if result:
                # Update persona state with assessment results
                await self._update_persona_with_assessment(persona, result)
                
                logger.info(f"Completed {assessment_type.upper()} assessment: {result.total_score}")
                return result
            else:
                logger.error(f"Failed to complete {assessment_type.upper()} assessment")
                return None
                
        except Exception as e:
            logger.error(f"Error in {assessment_type.upper()} assessment: {e}")
            return None
    
    async def _update_persona_with_assessment(self, persona: Persona, 
                                            result: PHQ9Result | GAD7Result | PSS10Result) -> bool:
        """Update persona state with assessment results."""
        try:
            # Update clinical scores based on assessment type
            if isinstance(result, PHQ9Result):
                persona.state.current_phq9 = result.total_score
                
                # Update emotional state based on depression severity
                if result.total_score >= 15:  # Moderate to severe
                    await persona_manager.update_persona_state(
                        persona,
                        emotional_state="depressed"
                    )
                    
            elif isinstance(result, GAD7Result):
                persona.state.current_gad7 = result.total_score
                
                # Update emotional state based on anxiety severity
                if result.total_score >= 15:  # Moderate to severe
                    await persona_manager.update_persona_state(
                        persona,
                        emotional_state="anxious"
                    )
                    
            elif isinstance(result, PSS10Result):
                persona.state.current_pss10 = result.total_score
                
                # Update stress level based on PSS-10 score
                stress_level = min(10.0, result.total_score / 4.0)  # Scale 0-40 to 0-10
                await persona_manager.update_persona_state(
                    persona,
                    stress_level=stress_level
                )
            
            # Update last assessment day
            persona.state.last_assessment_day = persona.state.simulation_day
            
            logger.debug(f"Updated persona state with {result.assessment_type} results")
            return True
            
        except Exception as e:
            logger.error(f"Error updating persona with assessment results: {e}")
            return False
    
    async def get_assessment_summary(self, persona: Persona) -> Dict[str, Any]:
        """Get assessment summary for persona."""
        try:
            summary = {
                "persona_id": persona.state.persona_id,
                "persona_name": persona.baseline.name,
                "simulation_day": persona.state.simulation_day,
                "last_assessment_day": persona.state.last_assessment_day,
                "current_scores": {
                    "phq9": persona.state.current_phq9,
                    "gad7": persona.state.current_gad7,
                    "pss10": persona.state.current_pss10
                },
                "baseline_scores": {
                    "phq9": persona.baseline.baseline_phq9,
                    "gad7": persona.baseline.baseline_gad7,
                    "pss10": persona.baseline.baseline_pss10
                },
                "score_changes": {},
                "emotional_state": persona.state.emotional_state,
                "stress_level": persona.state.stress_level
            }
            
            # Calculate score changes
            for score_type in ["phq9", "gad7", "pss10"]:
                current = getattr(persona.state, f"current_{score_type}")
                baseline = getattr(persona.baseline, f"baseline_{score_type}")
                
                if current is not None:
                    summary["score_changes"][score_type] = current - baseline
                else:
                    summary["score_changes"][score_type] = None
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting assessment summary: {e}")
            return {}
    
    async def check_assessment_due(self, persona: Persona, 
                                 assessment_interval_days: int = 7) -> bool:
        """Check if assessment is due for persona."""
        return persona.is_assessment_due(assessment_interval_days)
    
    async def get_assessment_history(self, persona_id: str) -> List[AssessmentSession]:
        """Get assessment history for persona."""
        return await persona_manager.get_persona_assessment_history(persona_id)
    
    async def analyze_assessment_trends(self, persona_id: str) -> Dict[str, Any]:
        """Analyze assessment trends over time."""
        try:
            # Get assessment history
            sessions = await self.get_assessment_history(persona_id)
            
            if not sessions:
                return {"error": "No assessment history found"}
            
            # Extract time series data
            trends = {
                "simulation_days": [],
                "phq9_scores": [],
                "gad7_scores": [],
                "pss10_scores": [],
                "completion_rates": []
            }
            
            for session in sessions:
                trends["simulation_days"].append(session.simulation_day)
                trends["completion_rates"].append(session.completion_rate)
                
                # Extract scores
                if session.phq9_result:
                    trends["phq9_scores"].append(session.phq9_result.total_score)
                else:
                    trends["phq9_scores"].append(None)
                
                if session.gad7_result:
                    trends["gad7_scores"].append(session.gad7_result.total_score)
                else:
                    trends["gad7_scores"].append(None)
                
                if session.pss10_result:
                    trends["pss10_scores"].append(session.pss10_result.total_score)
                else:
                    trends["pss10_scores"].append(None)
            
            # Calculate trend statistics
            trend_stats = {
                "total_assessments": len(sessions),
                "average_completion_rate": sum(trends["completion_rates"]) / len(trends["completion_rates"]),
                "score_ranges": {},
                "trend_direction": {}
            }
            
            # Calculate score ranges and trends
            for score_type in ["phq9_scores", "gad7_scores", "pss10_scores"]:
                scores = [s for s in trends[score_type] if s is not None]
                if scores:
                    trend_stats["score_ranges"][score_type] = {
                        "min": min(scores),
                        "max": max(scores),
                        "mean": sum(scores) / len(scores)
                    }
                    
                    # Simple trend direction (first vs last score)
                    if len(scores) >= 2:
                        first_score = scores[0]
                        last_score = scores[-1]
                        if last_score > first_score:
                            trend_stats["trend_direction"][score_type] = "increasing"
                        elif last_score < first_score:
                            trend_stats["trend_direction"][score_type] = "decreasing"
                        else:
                            trend_stats["trend_direction"][score_type] = "stable"
            
            return {
                "trends": trends,
                "statistics": trend_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing assessment trends: {e}")
            return {"error": str(e)}
    
    async def validate_assessment_response(self, response: str, 
                                        assessment_type: str) -> Tuple[bool, Optional[int]]:
        """Validate assessment response and return score."""
        try:
            score = await llm_service.parse_assessment_response(response, assessment_type)
            return score is not None, score
            
        except Exception as e:
            logger.error(f"Error validating assessment response: {e}")
            return False, None
    
    async def get_assessment_questions(self, assessment_type: str) -> List[str]:
        """Get assessment questions for a specific type."""
        assessment_questions = {
            "phq9": [
                "Little interest or pleasure in doing things?",
                "Feeling down, depressed, or hopeless?",
                "Trouble falling/staying asleep, sleeping too much?",
                "Feeling tired or having little energy?",
                "Poor appetite or overeating?",
                "Feeling bad about yourself - or that you are a failure?",
                "Trouble concentrating on things?",
                "Moving or speaking slowly/being fidgety or restless?",
                "Thoughts that you would be better off dead or of hurting yourself?"
            ],
            "gad7": [
                "Feeling nervous, anxious, or on edge?",
                "Not being able to stop or control worrying?",
                "Worrying too much about different things?",
                "Trouble relaxing?",
                "Being so restless that it's hard to sit still?",
                "Becoming easily annoyed or irritable?",
                "Feeling afraid as if something awful might happen?"
            ],
            "pss10": [
                "In the last month, how often have you been upset because of something that happened unexpectedly?",
                "In the last month, how often have you felt that you were unable to control the important things in your life?",
                "In the last month, how often have you felt nervous and stressed?",
                "In the last month, how often have you felt confident about your ability to handle your personal problems?",
                "In the last month, how often have you felt that things were going your way?",
                "In the last month, how often have you found that you could not cope with all the things you had to do?",
                "In the last month, how often have you been able to control irritations in your life?",
                "In the last month, how often have you felt that you were on top of things?",
                "In the last month, how often have you been angered because of things that were outside of your control?",
                "In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?"
            ]
        }
        
        return assessment_questions.get(assessment_type, [])


# Global assessment service instance
assessment_service = AssessmentService() 