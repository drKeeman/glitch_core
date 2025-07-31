"""
Enhanced psychiatric scale implementations with clinical validation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from src.models.assessment import (
    PHQ9Result, GAD7Result, PSS10Result, SeverityLevel
)
from src.models.persona import Persona


logger = logging.getLogger(__name__)


class PsychiatricScaleValidator:
    """Validator for psychiatric scale responses and scoring."""
    
    # Clinical thresholds and validation rules
    PHQ9_THRESHOLDS = {
        "minimal": 5,
        "mild": 10, 
        "moderate": 15,
        "severe": 20
    }
    
    GAD7_THRESHOLDS = {
        "minimal": 5,
        "mild": 10,
        "moderate": 15, 
        "severe": 20
    }
    
    PSS10_THRESHOLDS = {
        "minimal": 13,
        "mild": 16,
        "moderate": 19,
        "severe": 22
    }
    
    # Reverse scoring items (higher score = better outcome)
    PSS10_REVERSE_ITEMS = [4, 5, 7, 8]  # 1-indexed
    
    @classmethod
    def validate_phq9_response(cls, response: str, question_index: int) -> Tuple[bool, Optional[int]]:
        """Validate PHQ-9 response and return score."""
        try:
            # Clean response
            response = response.strip().lower()
            
            # Extract numeric score
            import re
            numbers = re.findall(r'\b[0-3]\b', response)
            
            if numbers:
                score = int(numbers[0])
                if 0 <= score <= 3:
                    return True, score
            
            # Parse text responses
            score_map = {
                "not at all": 0, "never": 0, "0": 0,
                "several days": 1, "sometimes": 1, "1": 1,
                "more than half the days": 2, "often": 2, "2": 2,
                "nearly every day": 3, "always": 3, "3": 3
            }
            
            for text, score in score_map.items():
                if text in response:
                    return True, score
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error validating PHQ-9 response: {e}")
            return False, None
    
    @classmethod
    def validate_gad7_response(cls, response: str, question_index: int) -> Tuple[bool, Optional[int]]:
        """Validate GAD-7 response and return score."""
        try:
            # Clean response
            response = response.strip().lower()
            
            # Extract numeric score
            import re
            numbers = re.findall(r'\b[0-3]\b', response)
            
            if numbers:
                score = int(numbers[0])
                if 0 <= score <= 3:
                    return True, score
            
            # Parse text responses
            score_map = {
                "not at all": 0, "never": 0, "0": 0,
                "several days": 1, "sometimes": 1, "1": 1,
                "more than half the days": 2, "often": 2, "2": 2,
                "nearly every day": 3, "always": 3, "3": 3
            }
            
            for text, score in score_map.items():
                if text in response:
                    return True, score
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error validating GAD-7 response: {e}")
            return False, None
    
    @classmethod
    def validate_pss10_response(cls, response: str, question_index: int) -> Tuple[bool, Optional[int]]:
        """Validate PSS-10 response and return score."""
        try:
            # Clean response
            response = response.strip().lower()
            
            # Extract numeric score
            import re
            numbers = re.findall(r'\b[0-4]\b', response)
            
            if numbers:
                score = int(numbers[0])
                if 0 <= score <= 4:
                    return True, score
            
            # Parse text responses
            score_map = {
                "never": 0, "0": 0,
                "almost never": 1, "1": 1,
                "sometimes": 2, "2": 2,
                "fairly often": 3, "3": 3,
                "very often": 4, "4": 4
            }
            
            for text, score in score_map.items():
                if text in response:
                    return True, score
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error validating PSS-10 response: {e}")
            return False, None
    
    @classmethod
    def calculate_pss10_score(cls, raw_scores: List[int]) -> int:
        """Calculate PSS-10 score with reverse scoring."""
        if len(raw_scores) != 10:
            raise ValueError(f"PSS-10 requires exactly 10 scores, got {len(raw_scores)}")
        
        total_score = 0
        for i, score in enumerate(raw_scores):
            # Apply reverse scoring for items 4, 5, 7, 8 (0-indexed: 3, 4, 6, 7)
            if i in [3, 4, 6, 7]:  # Reverse scoring items
                total_score += (4 - score)  # Reverse: 0->4, 1->3, 2->2, 3->1, 4->0
            else:
                total_score += score
        
        return total_score


class ClinicalInterpreter:
    """Clinical interpretation of assessment results."""
    
    @classmethod
    def interpret_phq9_result(cls, result: PHQ9Result) -> Dict[str, Any]:
        """Provide clinical interpretation of PHQ-9 results."""
        interpretation = {
            "severity_level": result.severity_level.value,
            "total_score": result.total_score,
            "clinical_meaning": "",
            "recommendations": [],
            "risk_factors": [],
            "suicidal_risk": "low"
        }
        
        # Clinical interpretation based on score
        if result.severity_level == SeverityLevel.MINIMAL:
            interpretation["clinical_meaning"] = "Minimal depressive symptoms"
            interpretation["recommendations"] = ["Continue monitoring", "Maintain current routine"]
        elif result.severity_level == SeverityLevel.MILD:
            interpretation["clinical_meaning"] = "Mild depressive symptoms"
            interpretation["recommendations"] = ["Consider lifestyle changes", "Monitor for worsening"]
        elif result.severity_level == SeverityLevel.MODERATE:
            interpretation["clinical_meaning"] = "Moderate depressive symptoms"
            interpretation["recommendations"] = ["Consider professional evaluation", "Implement coping strategies"]
        else:  # Severe
            interpretation["clinical_meaning"] = "Severe depressive symptoms"
            interpretation["recommendations"] = ["Immediate professional evaluation recommended", "Safety assessment needed"]
        
        # Suicidal ideation assessment
        if result.suicidal_ideation_score >= 2:
            interpretation["suicidal_risk"] = "high"
            interpretation["risk_factors"].append("Suicidal ideation present")
            interpretation["recommendations"].insert(0, "Immediate safety assessment required")
        
        # Additional risk factors
        if result.total_score >= 20:
            interpretation["risk_factors"].append("High depression severity")
        
        return interpretation
    
    @classmethod
    def interpret_gad7_result(cls, result: GAD7Result) -> Dict[str, Any]:
        """Provide clinical interpretation of GAD-7 results."""
        interpretation = {
            "severity_level": result.severity_level.value,
            "total_score": result.total_score,
            "clinical_meaning": "",
            "recommendations": [],
            "risk_factors": []
        }
        
        # Clinical interpretation based on score
        if result.severity_level == SeverityLevel.MINIMAL:
            interpretation["clinical_meaning"] = "Minimal anxiety symptoms"
            interpretation["recommendations"] = ["Continue monitoring", "Maintain current routine"]
        elif result.severity_level == SeverityLevel.MILD:
            interpretation["clinical_meaning"] = "Mild anxiety symptoms"
            interpretation["recommendations"] = ["Consider stress management", "Monitor for worsening"]
        elif result.severity_level == SeverityLevel.MODERATE:
            interpretation["clinical_meaning"] = "Moderate anxiety symptoms"
            interpretation["recommendations"] = ["Consider professional evaluation", "Implement relaxation techniques"]
        else:  # Severe
            interpretation["clinical_meaning"] = "Severe anxiety symptoms"
            interpretation["recommendations"] = ["Immediate professional evaluation recommended", "Consider medication evaluation"]
        
        # Additional risk factors
        if result.total_score >= 20:
            interpretation["risk_factors"].append("High anxiety severity")
        
        return interpretation
    
    @classmethod
    def interpret_pss10_result(cls, result: PSS10Result) -> Dict[str, Any]:
        """Provide clinical interpretation of PSS-10 results."""
        interpretation = {
            "severity_level": result.severity_level.value,
            "total_score": result.total_score,
            "clinical_meaning": "",
            "recommendations": [],
            "risk_factors": []
        }
        
        # Clinical interpretation based on score
        if result.severity_level == SeverityLevel.MINIMAL:
            interpretation["clinical_meaning"] = "Minimal perceived stress"
            interpretation["recommendations"] = ["Continue current stress management", "Maintain healthy routines"]
        elif result.severity_level == SeverityLevel.MILD:
            interpretation["clinical_meaning"] = "Mild perceived stress"
            interpretation["recommendations"] = ["Consider stress management techniques", "Monitor stress levels"]
        elif result.severity_level == SeverityLevel.MODERATE:
            interpretation["clinical_meaning"] = "Moderate perceived stress"
            interpretation["recommendations"] = ["Implement stress reduction strategies", "Consider professional support"]
        else:  # Severe
            interpretation["clinical_meaning"] = "Severe perceived stress"
            interpretation["recommendations"] = ["Immediate stress management intervention", "Professional evaluation recommended"]
        
        # Additional risk factors
        if result.total_score >= 25:
            interpretation["risk_factors"].append("High stress levels")
        
        return interpretation


class AssessmentOrchestrator:
    """Orchestrates assessment administration and scheduling."""
    
    def __init__(self):
        """Initialize assessment orchestrator."""
        self.validator = PsychiatricScaleValidator()
        self.interpreter = ClinicalInterpreter()
        
        # Assessment schedules
        self.assessment_schedules = {
            "weekly": 7,
            "biweekly": 14,
            "monthly": 30
        }
    
    async def conduct_phq9_assessment(self, persona: Persona, 
                                    responses: List[str]) -> Optional[PHQ9Result]:
        """Conduct PHQ-9 assessment with validation."""
        try:
            # Validate all responses
            validated_scores = []
            for i, response in enumerate(responses):
                is_valid, score = self.validator.validate_phq9_response(response, i)
                if is_valid and score is not None:
                    validated_scores.append(score)
                else:
                    logger.warning(f"Invalid PHQ-9 response {i+1}: {response}")
                    validated_scores.append(0)  # Conservative fallback
            
            if len(validated_scores) != 9:
                logger.error(f"PHQ-9 requires 9 responses, got {len(validated_scores)}")
                return None
            
            # Calculate total score
            total_score = sum(validated_scores)
            
            # Create result
            result = PHQ9Result(
                assessment_id=f"{persona.state.persona_id}_phq9_{persona.state.simulation_day}",
                persona_id=persona.state.persona_id,
                assessment_type="phq9",
                simulation_day=persona.state.simulation_day,
                raw_responses=responses,
                parsed_scores=validated_scores,
                total_score=total_score,
                severity_level=PHQ9Result.calculate_severity(total_score),
                suicidal_ideation_score=validated_scores[8],  # Item 9
                depression_severity=PHQ9Result.calculate_severity(total_score)
            )
            
            # Add clinical interpretation
            result.clinical_interpretation = self.interpreter.interpret_phq9_result(result)
            
            logger.info(f"Completed PHQ-9 assessment for {persona.baseline.name}: {total_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error conducting PHQ-9 assessment: {e}")
            return None
    
    async def conduct_gad7_assessment(self, persona: Persona, 
                                    responses: List[str]) -> Optional[GAD7Result]:
        """Conduct GAD-7 assessment with validation."""
        try:
            # Validate all responses
            validated_scores = []
            for i, response in enumerate(responses):
                is_valid, score = self.validator.validate_gad7_response(response, i)
                if is_valid and score is not None:
                    validated_scores.append(score)
                else:
                    logger.warning(f"Invalid GAD-7 response {i+1}: {response}")
                    validated_scores.append(0)  # Conservative fallback
            
            if len(validated_scores) != 7:
                logger.error(f"GAD-7 requires 7 responses, got {len(validated_scores)}")
                return None
            
            # Calculate total score
            total_score = sum(validated_scores)
            
            # Create result
            result = GAD7Result(
                assessment_id=f"{persona.state.persona_id}_gad7_{persona.state.simulation_day}",
                persona_id=persona.state.persona_id,
                assessment_type="gad7",
                simulation_day=persona.state.simulation_day,
                raw_responses=responses,
                parsed_scores=validated_scores,
                total_score=total_score,
                severity_level=GAD7Result.calculate_severity(total_score),
                anxiety_severity=GAD7Result.calculate_severity(total_score)
            )
            
            # Add clinical interpretation
            result.clinical_interpretation = self.interpreter.interpret_gad7_result(result)
            
            logger.info(f"Completed GAD-7 assessment for {persona.baseline.name}: {total_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error conducting GAD-7 assessment: {e}")
            return None
    
    async def conduct_pss10_assessment(self, persona: Persona, 
                                     responses: List[str]) -> Optional[PSS10Result]:
        """Conduct PSS-10 assessment with validation and reverse scoring."""
        try:
            # Validate all responses
            validated_scores = []
            for i, response in enumerate(responses):
                is_valid, score = self.validator.validate_pss10_response(response, i)
                if is_valid and score is not None:
                    validated_scores.append(score)
                else:
                    logger.warning(f"Invalid PSS-10 response {i+1}: {response}")
                    validated_scores.append(2)  # Conservative fallback (middle score)
            
            if len(validated_scores) != 10:
                logger.error(f"PSS-10 requires 10 responses, got {len(validated_scores)}")
                return None
            
            # Calculate total score with reverse scoring
            total_score = self.validator.calculate_pss10_score(validated_scores)
            
            # Create result
            result = PSS10Result(
                assessment_id=f"{persona.state.persona_id}_pss10_{persona.state.simulation_day}",
                persona_id=persona.state.persona_id,
                assessment_type="pss10",
                simulation_day=persona.state.simulation_day,
                raw_responses=responses,
                parsed_scores=validated_scores,
                total_score=total_score,
                severity_level=PSS10Result.calculate_severity(total_score),
                stress_severity=PSS10Result.calculate_severity(total_score)
            )
            
            # Add clinical interpretation
            result.clinical_interpretation = self.interpreter.interpret_pss10_result(result)
            
            logger.info(f"Completed PSS-10 assessment for {persona.baseline.name}: {total_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error conducting PSS-10 assessment: {e}")
            return None
    
    def is_assessment_due(self, persona: Persona, schedule: str = "weekly") -> bool:
        """Check if assessment is due based on schedule."""
        interval_days = self.assessment_schedules.get(schedule, 7)
        days_since_last = persona.state.simulation_day - persona.state.last_assessment_day
        return days_since_last >= interval_days
    
    def get_next_assessment_day(self, persona: Persona, schedule: str = "weekly") -> int:
        """Get the next assessment day based on schedule."""
        interval_days = self.assessment_schedules.get(schedule, 7)
        return persona.state.last_assessment_day + interval_days


# Global instances
psychiatric_validator = PsychiatricScaleValidator()
clinical_interpreter = ClinicalInterpreter()
assessment_orchestrator = AssessmentOrchestrator() 