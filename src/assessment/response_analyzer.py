"""
Response analyzer for assessment consistency and validation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re

from src.models.assessment import AssessmentResult, PHQ9Result, GAD7Result, PSS10Result
from src.models.persona import Persona


logger = logging.getLogger(__name__)


class ResponseAnalyzer:
    """Analyzes assessment responses for consistency and validity."""
    
    def __init__(self):
        """Initialize response analyzer."""
        # Consistency thresholds
        self.consistency_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        # Response patterns for validation
        self.valid_response_patterns = {
            "phq9": {
                "numeric": r'\b[0-3]\b',
                "text": [
                    "not at all", "never", "0",
                    "several days", "sometimes", "1", 
                    "more than half the days", "often", "2",
                    "nearly every day", "always", "3"
                ]
            },
            "gad7": {
                "numeric": r'\b[0-3]\b',
                "text": [
                    "not at all", "never", "0",
                    "several days", "sometimes", "1",
                    "more than half the days", "often", "2", 
                    "nearly every day", "always", "3"
                ]
            },
            "pss10": {
                "numeric": r'\b[0-4]\b',
                "text": [
                    "never", "0",
                    "almost never", "1",
                    "sometimes", "2",
                    "fairly often", "3",
                    "very often", "4"
                ]
            }
        }
    
    def analyze_response_consistency(self, responses: List[str], 
                                  assessment_type: str) -> Dict[str, Any]:
        """Analyze response consistency across assessment."""
        try:
            analysis = {
                "consistency_score": 0.0,
                "consistency_level": "unknown",
                "response_pattern": "unknown",
                "valid_responses": 0,
                "invalid_responses": 0,
                "response_variability": 0.0,
                "issues": []
            }
            
            if not responses:
                analysis["issues"].append("No responses provided")
                return analysis
            
            # Count valid vs invalid responses
            valid_count = 0
            scores = []
            
            for i, response in enumerate(responses):
                is_valid, score = self._validate_single_response(response, assessment_type, i)
                if is_valid and score is not None:
                    valid_count += 1
                    scores.append(score)
                else:
                    analysis["issues"].append(f"Invalid response {i+1}: {response}")
            
            analysis["valid_responses"] = valid_count
            analysis["invalid_responses"] = len(responses) - valid_count
            
            # Calculate consistency score
            if len(responses) > 0:
                analysis["consistency_score"] = valid_count / len(responses)
            
            # Determine consistency level
            if analysis["consistency_score"] >= self.consistency_thresholds["high"]:
                analysis["consistency_level"] = "high"
            elif analysis["consistency_score"] >= self.consistency_thresholds["medium"]:
                analysis["consistency_level"] = "medium"
            elif analysis["consistency_score"] >= self.consistency_thresholds["low"]:
                analysis["consistency_level"] = "low"
            else:
                analysis["consistency_level"] = "poor"
            
            # Analyze response pattern
            if scores:
                analysis["response_pattern"] = self._analyze_response_pattern(scores, assessment_type)
                analysis["response_variability"] = self._calculate_variability(scores)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing response consistency: {e}")
            return {"error": str(e)}
    
    def _validate_single_response(self, response: str, assessment_type: str, 
                                question_index: int) -> Tuple[bool, Optional[int]]:
        """Validate a single response."""
        try:
            response = response.strip().lower()
            
            # Check numeric patterns
            patterns = self.valid_response_patterns.get(assessment_type, {})
            numeric_pattern = patterns.get("numeric", r'\b[0-9]\b')
            
            numbers = re.findall(numeric_pattern, response)
            if numbers:
                score = int(numbers[0])
                if assessment_type in ["phq9", "gad7"] and 0 <= score <= 3:
                    return True, score
                elif assessment_type == "pss10" and 0 <= score <= 4:
                    return True, score
            
            # Check text patterns
            text_patterns = patterns.get("text", [])
            for text in text_patterns:
                if text in response:
                    # Extract score from text
                    if assessment_type in ["phq9", "gad7"]:
                        if text in ["not at all", "never", "0"]:
                            return True, 0
                        elif text in ["several days", "sometimes", "1"]:
                            return True, 1
                        elif text in ["more than half the days", "often", "2"]:
                            return True, 2
                        elif text in ["nearly every day", "always", "3"]:
                            return True, 3
                    elif assessment_type == "pss10":
                        if text in ["never", "0"]:
                            return True, 0
                        elif text in ["almost never", "1"]:
                            return True, 1
                        elif text in ["sometimes", "2"]:
                            return True, 2
                        elif text in ["fairly often", "3"]:
                            return True, 3
                        elif text in ["very often", "4"]:
                            return True, 4
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error validating single response: {e}")
            return False, None
    
    def _analyze_response_pattern(self, scores: List[int], assessment_type: str) -> str:
        """Analyze the pattern of responses."""
        if not scores:
            return "no_scores"
        
        # Check for extreme responses
        if assessment_type in ["phq9", "gad7"]:
            all_zeros = all(score == 0 for score in scores)
            all_threes = all(score == 3 for score in scores)
            
            if all_zeros:
                return "all_minimal"
            elif all_threes:
                return "all_severe"
            elif all(score >= 2 for score in scores):
                return "mostly_high"
            elif all(score <= 1 for score in scores):
                return "mostly_low"
            else:
                return "mixed"
        
        elif assessment_type == "pss10":
            all_zeros = all(score == 0 for score in scores)
            all_fours = all(score == 4 for score in scores)
            
            if all_zeros:
                return "all_minimal"
            elif all_fours:
                return "all_severe"
            elif all(score >= 3 for score in scores):
                return "mostly_high"
            elif all(score <= 1 for score in scores):
                return "mostly_low"
            else:
                return "mixed"
        
        return "unknown"
    
    def _calculate_variability(self, scores: List[int]) -> float:
        """Calculate response variability."""
        if len(scores) <= 1:
            return 0.0
        
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        return variance ** 0.5  # Standard deviation
    
    def detect_response_anomalies(self, result: AssessmentResult) -> List[Dict[str, Any]]:
        """Detect anomalies in assessment responses."""
        anomalies = []
        
        try:
            # Check for missing responses
            if len(result.raw_responses) != len(result.parsed_scores):
                anomalies.append({
                    "type": "missing_responses",
                    "description": f"Response count mismatch: {len(result.raw_responses)} vs {len(result.parsed_scores)}",
                    "severity": "high"
                })
            
            # Check for invalid scores
            if isinstance(result, PHQ9Result) or isinstance(result, GAD7Result):
                invalid_scores = [score for score in result.parsed_scores if score < 0 or score > 3]
            elif isinstance(result, PSS10Result):
                invalid_scores = [score for score in result.parsed_scores if score < 0 or score > 4]
            else:
                invalid_scores = []
            
            if invalid_scores:
                anomalies.append({
                    "type": "invalid_scores",
                    "description": f"Invalid scores detected: {invalid_scores}",
                    "severity": "high"
                })
            
            # Check for extreme patterns
            if isinstance(result, PHQ9Result):
                # Check for suicidal ideation
                if result.suicidal_ideation_score >= 2:
                    anomalies.append({
                        "type": "suicidal_ideation",
                        "description": f"Suicidal ideation detected: score {result.suicidal_ideation_score}",
                        "severity": "critical"
                    })
                
                # Check for severe depression
                if result.total_score >= 20:
                    anomalies.append({
                        "type": "severe_depression",
                        "description": f"Severe depression score: {result.total_score}",
                        "severity": "high"
                    })
            
            # Check for response consistency
            consistency_analysis = self.analyze_response_consistency(
                result.raw_responses, result.assessment_type
            )
            
            if consistency_analysis.get("consistency_level") in ["low", "poor"]:
                anomalies.append({
                    "type": "low_consistency",
                    "description": f"Low response consistency: {consistency_analysis.get('consistency_score', 0):.2f}",
                    "severity": "medium"
                })
            
            # Check for unusual response patterns
            if consistency_analysis.get("response_pattern") in ["all_minimal", "all_severe"]:
                anomalies.append({
                    "type": "extreme_pattern",
                    "description": f"Extreme response pattern: {consistency_analysis.get('response_pattern')}",
                    "severity": "medium"
                })
            
        except Exception as e:
            logger.error(f"Error detecting response anomalies: {e}")
            anomalies.append({
                "type": "analysis_error",
                "description": f"Error in anomaly detection: {str(e)}",
                "severity": "low"
            })
        
        return anomalies
    
    def validate_assessment_result(self, result: AssessmentResult) -> Dict[str, Any]:
        """Comprehensive validation of assessment result."""
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "anomalies": [],
            "consistency_analysis": {},
            "recommendations": []
        }
        
        try:
            # Basic validation
            if not result.assessment_id:
                validation["issues"].append("Missing assessment ID")
                validation["is_valid"] = False
            
            if not result.persona_id:
                validation["issues"].append("Missing persona ID")
                validation["is_valid"] = False
            
            if result.simulation_day < 0:
                validation["issues"].append("Invalid simulation day")
                validation["is_valid"] = False
            
            # Score validation
            if result.total_score < 0:
                validation["issues"].append("Negative total score")
                validation["is_valid"] = False
            
            # Consistency analysis
            consistency_analysis = self.analyze_response_consistency(
                result.raw_responses, result.assessment_type
            )
            validation["consistency_analysis"] = consistency_analysis
            
            if consistency_analysis.get("consistency_level") in ["low", "poor"]:
                validation["warnings"].append("Low response consistency")
                validation["recommendations"].append("Consider re-administering assessment")
            
            # Anomaly detection
            anomalies = self.detect_response_anomalies(result)
            validation["anomalies"] = anomalies
            
            # Critical anomalies
            critical_anomalies = [a for a in anomalies if a.get("severity") == "critical"]
            if critical_anomalies:
                validation["is_valid"] = False
                validation["issues"].extend([a["description"] for a in critical_anomalies])
            
            # Recommendations based on analysis
            if validation["consistency_analysis"].get("invalid_responses", 0) > 0:
                validation["recommendations"].append("Review and correct invalid responses")
            
            if validation["consistency_analysis"].get("response_variability", 0) > 2.0:
                validation["recommendations"].append("High response variability - consider retesting")
            
        except Exception as e:
            logger.error(f"Error validating assessment result: {e}")
            validation["issues"].append(f"Validation error: {str(e)}")
            validation["is_valid"] = False
        
        return validation


# Global response analyzer instance
response_analyzer = ResponseAnalyzer() 