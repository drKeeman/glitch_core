"""
Clinical interpreter for assessment significance and recommendations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.models.assessment import (
    AssessmentResult, PHQ9Result, GAD7Result, PSS10Result, 
    AssessmentSession, SeverityLevel
)
from src.models.persona import Persona


logger = logging.getLogger(__name__)


class ClinicalInterpreter:
    """Provides clinical interpretation and significance assessment."""
    
    def __init__(self):
        """Initialize clinical interpreter."""
        # Clinical significance thresholds
        self.clinical_thresholds = {
            "phq9": {
                "minimal_change": 5.0,
                "moderate_change": 10.0,
                "severe_change": 15.0
            },
            "gad7": {
                "minimal_change": 5.0,
                "moderate_change": 10.0,
                "severe_change": 15.0
            },
            "pss10": {
                "minimal_change": 5.0,
                "moderate_change": 10.0,
                "severe_change": 15.0
            }
        }
        
        # Risk assessment criteria
        self.risk_criteria = {
            "suicidal_ideation": {
                "phq9_item_9": 2,  # Score of 2 or 3 on suicidal ideation item
                "critical_threshold": 2
            },
            "severe_symptoms": {
                "phq9_total": 20,
                "gad7_total": 20,
                "pss10_total": 25
            },
            "rapid_deterioration": {
                "weekly_increase": 10,  # 10+ point increase in a week
                "monthly_increase": 15   # 15+ point increase in a month
            }
        }
    
    def assess_clinical_significance(self, current_result: AssessmentResult, 
                                   baseline_result: Optional[AssessmentResult] = None,
                                   previous_results: Optional[List[AssessmentResult]] = None) -> Dict[str, Any]:
        """Assess clinical significance of assessment result."""
        try:
            assessment = {
                "is_clinically_significant": False,
                "significance_level": "none",
                "change_magnitude": 0.0,
                "trend_direction": "stable",
                "risk_level": "low",
                "clinical_recommendations": [],
                "risk_factors": [],
                "monitoring_priority": "routine"
            }
            
            # Calculate change from baseline
            if baseline_result:
                change = current_result.total_score - baseline_result.total_score
                assessment["change_magnitude"] = abs(change)
                assessment["trend_direction"] = "increasing" if change > 0 else "decreasing" if change < 0 else "stable"
                
                # Determine significance level
                thresholds = self.clinical_thresholds.get(current_result.assessment_type, {})
                if abs(change) >= thresholds.get("severe_change", 15):
                    assessment["significance_level"] = "severe"
                    assessment["is_clinically_significant"] = True
                elif abs(change) >= thresholds.get("moderate_change", 10):
                    assessment["significance_level"] = "moderate"
                    assessment["is_clinically_significant"] = True
                elif abs(change) >= thresholds.get("minimal_change", 5):
                    assessment["significance_level"] = "minimal"
                    assessment["is_clinically_significant"] = True
            
            # Assess risk level
            risk_assessment = self._assess_risk_level(current_result, previous_results)
            assessment["risk_level"] = risk_assessment["risk_level"]
            assessment["risk_factors"] = risk_assessment["risk_factors"]
            
            # Generate clinical recommendations
            recommendations = self._generate_clinical_recommendations(
                current_result, assessment, risk_assessment
            )
            assessment["clinical_recommendations"] = recommendations
            
            # Determine monitoring priority
            assessment["monitoring_priority"] = self._determine_monitoring_priority(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing clinical significance: {e}")
            return {"error": str(e)}
    
    def _assess_risk_level(self, result: AssessmentResult, 
                          previous_results: Optional[List[AssessmentResult]] = None) -> Dict[str, Any]:
        """Assess risk level based on current and previous results."""
        risk_assessment = {
            "risk_level": "low",
            "risk_factors": [],
            "risk_score": 0
        }
        
        try:
            # Check for critical risk factors
            if isinstance(result, PHQ9Result):
                # Suicidal ideation assessment
                if result.suicidal_ideation_score >= self.risk_criteria["suicidal_ideation"]["critical_threshold"]:
                    risk_assessment["risk_level"] = "critical"
                    risk_assessment["risk_factors"].append("Suicidal ideation present")
                    risk_assessment["risk_score"] += 10
                
                # Severe depression
                if result.total_score >= self.risk_criteria["severe_symptoms"]["phq9_total"]:
                    risk_assessment["risk_factors"].append("Severe depression symptoms")
                    risk_assessment["risk_score"] += 5
            
            elif isinstance(result, GAD7Result):
                # Severe anxiety
                if result.total_score >= self.risk_criteria["severe_symptoms"]["gad7_total"]:
                    risk_assessment["risk_factors"].append("Severe anxiety symptoms")
                    risk_assessment["risk_score"] += 5
            
            elif isinstance(result, PSS10Result):
                # Severe stress
                if result.total_score >= self.risk_criteria["severe_symptoms"]["pss10_total"]:
                    risk_assessment["risk_factors"].append("Severe stress symptoms")
                    risk_assessment["risk_score"] += 3
            
            # Check for rapid deterioration
            if previous_results and len(previous_results) >= 2:
                recent_change = result.total_score - previous_results[-1].total_score
                if recent_change >= self.risk_criteria["rapid_deterioration"]["weekly_increase"]:
                    risk_assessment["risk_factors"].append("Rapid symptom deterioration")
                    risk_assessment["risk_score"] += 3
            
            # Determine risk level based on score
            if risk_assessment["risk_score"] >= 10:
                risk_assessment["risk_level"] = "critical"
            elif risk_assessment["risk_score"] >= 5:
                risk_assessment["risk_level"] = "high"
            elif risk_assessment["risk_score"] >= 2:
                risk_assessment["risk_level"] = "medium"
            
        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            risk_assessment["risk_factors"].append(f"Risk assessment error: {str(e)}")
        
        return risk_assessment
    
    def _generate_clinical_recommendations(self, result: AssessmentResult, 
                                         assessment: Dict[str, Any],
                                         risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on assessment."""
        recommendations = []
        
        try:
            # Critical risk recommendations
            if risk_assessment["risk_level"] == "critical":
                recommendations.append("Immediate clinical evaluation required")
                recommendations.append("Safety assessment and monitoring")
                if "Suicidal ideation present" in risk_assessment["risk_factors"]:
                    recommendations.append("Crisis intervention services recommended")
            
            # High risk recommendations
            elif risk_assessment["risk_level"] == "high":
                recommendations.append("Urgent clinical evaluation recommended")
                recommendations.append("Consider medication evaluation")
                recommendations.append("Weekly monitoring required")
            
            # Moderate risk recommendations
            elif risk_assessment["risk_level"] == "medium":
                recommendations.append("Clinical evaluation within 1-2 weeks")
                recommendations.append("Implement coping strategies")
                recommendations.append("Bi-weekly monitoring")
            
            # Significance-based recommendations
            if assessment["is_clinically_significant"]:
                if assessment["significance_level"] == "severe":
                    recommendations.append("Immediate intervention recommended")
                elif assessment["significance_level"] == "moderate":
                    recommendations.append("Clinical evaluation within 1 week")
                elif assessment["significance_level"] == "minimal":
                    recommendations.append("Monitor for continued changes")
            
            # Assessment-specific recommendations
            if isinstance(result, PHQ9Result):
                if result.severity_level in [SeverityLevel.MODERATE, SeverityLevel.SEVERE]:
                    recommendations.append("Consider antidepressant medication evaluation")
                    recommendations.append("Psychotherapy referral recommended")
                
                if result.total_score >= 15:
                    recommendations.append("Functional impairment likely - assess daily activities")
            
            elif isinstance(result, GAD7Result):
                if result.severity_level in [SeverityLevel.MODERATE, SeverityLevel.SEVERE]:
                    recommendations.append("Consider anti-anxiety medication evaluation")
                    recommendations.append("Cognitive behavioral therapy recommended")
                
                if result.total_score >= 15:
                    recommendations.append("Assess impact on work and relationships")
            
            elif isinstance(result, PSS10Result):
                if result.severity_level in [SeverityLevel.MODERATE, SeverityLevel.SEVERE]:
                    recommendations.append("Stress management intervention recommended")
                    recommendations.append("Lifestyle modification counseling")
                
                if result.total_score >= 20:
                    recommendations.append("Assess coping mechanisms and support systems")
            
            # Trend-based recommendations
            if assessment["trend_direction"] == "increasing":
                recommendations.append("Monitor for continued deterioration")
                if assessment["change_magnitude"] >= 10:
                    recommendations.append("Consider medication adjustment")
            
            elif assessment["trend_direction"] == "decreasing":
                recommendations.append("Continue current treatment plan")
                recommendations.append("Monitor for sustained improvement")
            
            # Default recommendations
            if not recommendations:
                recommendations.append("Continue routine monitoring")
                recommendations.append("Maintain current treatment plan")
            
        except Exception as e:
            logger.error(f"Error generating clinical recommendations: {e}")
            recommendations.append("Clinical consultation recommended")
        
        return recommendations
    
    def _determine_monitoring_priority(self, assessment: Dict[str, Any]) -> str:
        """Determine monitoring priority based on assessment."""
        if assessment["risk_level"] == "critical":
            return "immediate"
        elif assessment["risk_level"] == "high":
            return "urgent"
        elif assessment["is_clinically_significant"]:
            return "elevated"
        else:
            return "routine"
    
    def analyze_longitudinal_trends(self, results: List[AssessmentResult]) -> Dict[str, Any]:
        """Analyze longitudinal trends across multiple assessments."""
        try:
            if len(results) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            # Sort by simulation day
            sorted_results = sorted(results, key=lambda x: x.simulation_day)
            
            analysis = {
                "trend_direction": "stable",
                "trend_magnitude": 0.0,
                "stability_score": 0.0,
                "change_points": [],
                "periods_of_change": [],
                "overall_trajectory": "stable"
            }
            
            # Calculate trend
            first_score = sorted_results[0].total_score
            last_score = sorted_results[-1].total_score
            score_change = last_score - first_score
            
            analysis["trend_magnitude"] = abs(score_change)
            
            if score_change > 5:
                analysis["trend_direction"] = "increasing"
                analysis["overall_trajectory"] = "deteriorating"
            elif score_change < -5:
                analysis["trend_direction"] = "decreasing"
                analysis["overall_trajectory"] = "improving"
            
            # Calculate stability
            scores = [r.total_score for r in sorted_results]
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            analysis["stability_score"] = 1.0 / (1.0 + variance)  # Higher = more stable
            
            # Detect change points
            for i in range(1, len(sorted_results)):
                change = sorted_results[i].total_score - sorted_results[i-1].total_score
                if abs(change) >= 5:  # Significant change
                    analysis["change_points"].append({
                        "day": sorted_results[i].simulation_day,
                        "change_magnitude": change,
                        "previous_score": sorted_results[i-1].total_score,
                        "current_score": sorted_results[i].total_score
                    })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing longitudinal trends: {e}")
            return {"error": str(e)}
    
    def generate_clinical_summary(self, session: AssessmentSession) -> Dict[str, Any]:
        """Generate comprehensive clinical summary for assessment session."""
        try:
            summary = {
                "session_id": session.session_id,
                "persona_id": session.persona_id,
                "simulation_day": session.simulation_day,
                "overall_severity": session.get_overall_severity().value,
                "composite_scores": session.get_composite_score(),
                "clinical_interpretations": {},
                "risk_assessment": {
                    "overall_risk": "low",
                    "risk_factors": [],
                    "critical_alerts": []
                },
                "recommendations": [],
                "monitoring_plan": "routine"
            }
            
            # Analyze each assessment
            for result in session.get_all_results():
                if isinstance(result, PHQ9Result):
                    summary["clinical_interpretations"]["depression"] = {
                        "score": result.total_score,
                        "severity": result.severity_level.value,
                        "suicidal_risk": "high" if result.suicidal_ideation_score >= 2 else "low"
                    }
                    
                    if result.suicidal_ideation_score >= 2:
                        summary["risk_assessment"]["critical_alerts"].append("Suicidal ideation present")
                        summary["risk_assessment"]["overall_risk"] = "critical"
                
                elif isinstance(result, GAD7Result):
                    summary["clinical_interpretations"]["anxiety"] = {
                        "score": result.total_score,
                        "severity": result.severity_level.value
                    }
                
                elif isinstance(result, PSS10Result):
                    summary["clinical_interpretations"]["stress"] = {
                        "score": result.total_score,
                        "severity": result.severity_level.value
                    }
            
            # Generate overall recommendations
            if summary["overall_severity"] in ["moderate", "severe"]:
                summary["recommendations"].append("Comprehensive clinical evaluation recommended")
                summary["monitoring_plan"] = "elevated"
            
            if summary["risk_assessment"]["overall_risk"] == "critical":
                summary["recommendations"].insert(0, "Immediate clinical intervention required")
                summary["monitoring_plan"] = "immediate"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating clinical summary: {e}")
            return {"error": str(e)}
    
    def interpret_phq9_result(self, result: PHQ9Result) -> Dict[str, Any]:
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
    
    def interpret_gad7_result(self, result: GAD7Result) -> Dict[str, Any]:
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
    
    def interpret_pss10_result(self, result: PSS10Result) -> Dict[str, Any]:
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


# Global clinical interpreter instance
clinical_interpreter = ClinicalInterpreter() 