"""
Unit tests for assessment scale models.
"""

import pytest
from datetime import datetime

from src.models.assessment import (
    AssessmentResult, PHQ9Result, GAD7Result, PSS10Result, 
    AssessmentSession, SeverityLevel
)


class TestAssessmentResult:
    """Test base AssessmentResult model."""
    
    def test_valid_assessment_result(self):
        """Test creating a valid assessment result."""
        result = AssessmentResult(
            assessment_id="test_assessment",
            persona_id="test_persona",
            assessment_type="test",
            simulation_day=7,
            total_score=10.0,
            severity_level=SeverityLevel.MILD,
            raw_responses=["Response 1", "Response 2"],
            parsed_scores=[2, 3]
        )
        
        assert result.assessment_id == "test_assessment"
        assert result.total_score == 10.0
        assert result.severity_level == SeverityLevel.MILD
    
    def test_get_score_change(self):
        """Test score change calculation."""
        result = AssessmentResult(
            assessment_id="test",
            persona_id="test",
            assessment_type="test",
            simulation_day=7,
            total_score=12.0,
            severity_level=SeverityLevel.MILD
        )
        
        change = result.get_score_change(8.0)
        assert change == 4.0  # 12.0 - 8.0
    
    def test_is_clinically_significant(self):
        """Test clinical significance checking."""
        result = AssessmentResult(
            assessment_id="test",
            persona_id="test",
            assessment_type="test",
            simulation_day=7,
            total_score=15.0,
            severity_level=SeverityLevel.MODERATE
        )
        
        # Change of 7 points (15 - 8) >= threshold of 5
        assert result.is_clinically_significant(8.0, threshold=5.0) is True
        
        # Change of 3 points (15 - 12) < threshold of 5
        assert result.is_clinically_significant(12.0, threshold=5.0) is False


class TestPHQ9Result:
    """Test PHQ-9 assessment result."""
    
    def test_valid_phq9_result(self):
        """Test creating a valid PHQ-9 result."""
        result = PHQ9Result(
            assessment_id="test_phq9",
            persona_id="test_persona",
            simulation_day=7,
            total_score=12.0,
            severity_level=SeverityLevel.MILD,
            depression_severity=SeverityLevel.MILD,
            suicidal_ideation_score=1
        )
        
        assert result.assessment_type == "phq9"
        assert result.depression_severity == SeverityLevel.MILD
        assert result.suicidal_ideation_score == 1
    
    def test_calculate_severity(self):
        """Test PHQ-9 severity calculation."""
        # Minimal severity
        assert PHQ9Result.calculate_severity(4) == SeverityLevel.MINIMAL
        
        # Mild severity
        assert PHQ9Result.calculate_severity(8) == SeverityLevel.MILD
        
        # Moderate severity
        assert PHQ9Result.calculate_severity(12) == SeverityLevel.MODERATE
        
        # Severe severity
        assert PHQ9Result.calculate_severity(18) == SeverityLevel.SEVERE
    
    def test_has_suicidal_ideation(self):
        """Test suicidal ideation detection."""
        result = PHQ9Result(
            assessment_id="test",
            persona_id="test",
            simulation_day=7,
            total_score=10.0,
            severity_level=SeverityLevel.MILD,
            depression_severity=SeverityLevel.MILD,
            suicidal_ideation_score=2
        )
        
        assert result.has_suicidal_ideation() is True
        
        result.suicidal_ideation_score = 1
        assert result.has_suicidal_ideation() is False


class TestGAD7Result:
    """Test GAD-7 assessment result."""
    
    def test_valid_gad7_result(self):
        """Test creating a valid GAD-7 result."""
        result = GAD7Result(
            assessment_id="test_gad7",
            persona_id="test_persona",
            simulation_day=7,
            total_score=8.0,
            severity_level=SeverityLevel.MILD,
            anxiety_severity=SeverityLevel.MILD,
            worry_duration="several months"
        )
        
        assert result.assessment_type == "gad7"
        assert result.anxiety_severity == SeverityLevel.MILD
        assert result.worry_duration == "several months"
    
    def test_calculate_severity(self):
        """Test GAD-7 severity calculation."""
        # Minimal severity
        assert GAD7Result.calculate_severity(3) == SeverityLevel.MINIMAL
        
        # Mild severity
        assert GAD7Result.calculate_severity(8) == SeverityLevel.MILD
        
        # Moderate severity
        assert GAD7Result.calculate_severity(12) == SeverityLevel.MODERATE
        
        # Severe severity
        assert GAD7Result.calculate_severity(18) == SeverityLevel.SEVERE


class TestPSS10Result:
    """Test PSS-10 assessment result."""
    
    def test_valid_pss10_result(self):
        """Test creating a valid PSS-10 result."""
        result = PSS10Result(
            assessment_id="test_pss10",
            persona_id="test_persona",
            simulation_day=7,
            total_score=16.0,
            severity_level=SeverityLevel.MILD,
            stress_severity=SeverityLevel.MILD,
            coping_effectiveness=0.6
        )
        
        assert result.assessment_type == "pss10"
        assert result.stress_severity == SeverityLevel.MILD
        assert result.coping_effectiveness == 0.6
    
    def test_calculate_severity(self):
        """Test PSS-10 severity calculation."""
        # Minimal severity
        assert PSS10Result.calculate_severity(10) == SeverityLevel.MINIMAL
        
        # Mild severity
        assert PSS10Result.calculate_severity(14) == SeverityLevel.MILD
        
        # Moderate severity
        assert PSS10Result.calculate_severity(18) == SeverityLevel.MODERATE
        
        # Severe severity
        assert PSS10Result.calculate_severity(25) == SeverityLevel.SEVERE


class TestAssessmentSession:
    """Test AssessmentSession model."""
    
    def test_valid_session(self):
        """Test creating a valid assessment session."""
        session = AssessmentSession(
            session_id="test_session",
            persona_id="test_persona",
            simulation_day=7
        )
        
        assert session.session_id == "test_session"
        assert session.persona_id == "test_persona"
        assert session.simulation_day == 7
        assert session.is_complete() is False
    
    def test_get_all_results(self):
        """Test getting all assessment results."""
        session = AssessmentSession(
            session_id="test",
            persona_id="test",
            simulation_day=7
        )
        
        # Initially no results
        assert len(session.get_all_results()) == 0
        
        # Add PHQ-9 result
        phq9_result = PHQ9Result(
            assessment_id="phq9_test",
            persona_id="test",
            simulation_day=7,
            total_score=10.0,
            severity_level=SeverityLevel.MILD,
            depression_severity=SeverityLevel.MILD,
            suicidal_ideation_score=0
        )
        session.phq9_result = phq9_result
        
        assert len(session.get_all_results()) == 1
        assert session.get_all_results()[0] == phq9_result
    
    def test_get_composite_score(self):
        """Test getting composite scores."""
        session = AssessmentSession(
            session_id="test",
            persona_id="test",
            simulation_day=7
        )
        
        # Add multiple results
        session.phq9_result = PHQ9Result(
            assessment_id="phq9",
            persona_id="test",
            simulation_day=7,
            total_score=10.0,
            severity_level=SeverityLevel.MILD,
            depression_severity=SeverityLevel.MILD,
            suicidal_ideation_score=0
        )
        
        session.gad7_result = GAD7Result(
            assessment_id="gad7",
            persona_id="test",
            simulation_day=7,
            total_score=8.0,
            severity_level=SeverityLevel.MILD,
            anxiety_severity=SeverityLevel.MILD,
            worry_duration="several months"
        )
        
        composite = session.get_composite_score()
        assert composite["depression"] == 10.0
        assert composite["anxiety"] == 8.0
        assert "stress" not in composite
    
    def test_get_overall_severity(self):
        """Test overall severity calculation."""
        session = AssessmentSession(
            session_id="test",
            persona_id="test",
            simulation_day=7
        )
        
        # No results - should be minimal
        assert session.get_overall_severity() == SeverityLevel.MINIMAL
        
        # Add mild result
        session.phq9_result = PHQ9Result(
            assessment_id="phq9",
            persona_id="test",
            simulation_day=7,
            total_score=10.0,
            severity_level=SeverityLevel.MILD,
            depression_severity=SeverityLevel.MILD,
            suicidal_ideation_score=0
        )
        
        assert session.get_overall_severity() == SeverityLevel.MILD
        
        # Add severe result - should override
        session.gad7_result = GAD7Result(
            assessment_id="gad7",
            persona_id="test",
            simulation_day=7,
            total_score=20.0,
            severity_level=SeverityLevel.SEVERE,
            anxiety_severity=SeverityLevel.SEVERE,
            worry_duration="several months"
        )
        
        assert session.get_overall_severity() == SeverityLevel.SEVERE
    
    def test_mark_completed(self):
        """Test marking session as completed."""
        session = AssessmentSession(
            session_id="test",
            persona_id="test",
            simulation_day=7
        )
        
        assert session.is_complete() is False
        assert session.completed_at is None
        
        session.mark_completed()
        
        assert session.is_complete() is True
        assert session.completed_at is not None
        assert session.session_duration is not None 