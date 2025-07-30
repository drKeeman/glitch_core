"""
Unit tests for psychiatric scales implementation.
"""

import pytest
from datetime import datetime

from src.assessment.psychiatric_scales import (
    PsychiatricScaleValidator,
    AssessmentOrchestrator
)
from src.assessment.clinical_interpreter import ClinicalInterpreter
from src.models.assessment import (
    PHQ9Result, GAD7Result, PSS10Result, SeverityLevel
)
from src.models.persona import Persona, PersonaBaseline, PersonaState


class TestPsychiatricScaleValidator:
    """Test psychiatric scale validator."""
    
    def test_validate_phq9_response_valid_numeric(self):
        """Test PHQ-9 response validation with valid numeric input."""
        validator = PsychiatricScaleValidator()
        
        # Test valid numeric responses
        assert validator.validate_phq9_response("2", 0) == (True, 2)
        assert validator.validate_phq9_response("0", 0) == (True, 0)
        assert validator.validate_phq9_response("3", 0) == (True, 3)
    
    def test_validate_phq9_response_valid_text(self):
        """Test PHQ-9 response validation with valid text input."""
        validator = PsychiatricScaleValidator()
        
        # Test valid text responses
        assert validator.validate_phq9_response("not at all", 0) == (True, 0)
        assert validator.validate_phq9_response("several days", 0) == (True, 1)
        assert validator.validate_phq9_response("more than half the days", 0) == (True, 2)
        assert validator.validate_phq9_response("nearly every day", 0) == (True, 3)
    
    def test_validate_phq9_response_invalid(self):
        """Test PHQ-9 response validation with invalid input."""
        validator = PsychiatricScaleValidator()
        
        # Test invalid responses
        assert validator.validate_phq9_response("invalid", 0) == (False, None)
        assert validator.validate_phq9_response("5", 0) == (False, None)
        assert validator.validate_phq9_response("", 0) == (False, None)
    
    def test_validate_gad7_response_valid(self):
        """Test GAD-7 response validation."""
        validator = PsychiatricScaleValidator()
        
        # Test valid responses
        assert validator.validate_gad7_response("1", 0) == (True, 1)
        assert validator.validate_gad7_response("sometimes", 0) == (True, 1)
        assert validator.validate_gad7_response("often", 0) == (True, 2)
    
    def test_validate_pss10_response_valid(self):
        """Test PSS-10 response validation."""
        validator = PsychiatricScaleValidator()
        
        # Test valid responses
        assert validator.validate_pss10_response("2", 0) == (True, 2)
        assert validator.validate_pss10_response("sometimes", 0) == (True, 2)
        assert validator.validate_pss10_response("very often", 0) == (True, 4)
    
    def test_calculate_pss10_score(self):
        """Test PSS-10 score calculation with reverse scoring."""
        validator = PsychiatricScaleValidator()
        
        # Test normal scoring (items 1, 2, 3, 6, 9, 10)
        normal_scores = [2, 3, 1, 0, 2, 1, 3, 2, 1, 2]  # 10 items
        total_score = validator.calculate_pss10_score(normal_scores)
        
        # Manual calculation: 2+3+1+0+2+1+3+2+1+2 = 17
        # Reverse scoring items (4,5,7,8): 0->4, 0->4, 3->1, 2->2
        # So: 2+3+1+4+2+1+1+2+1+2 = 19
        expected_score = 19
        assert total_score == expected_score
    
    def test_calculate_pss10_score_invalid_length(self):
        """Test PSS-10 score calculation with invalid number of items."""
        validator = PsychiatricScaleValidator()
        
        with pytest.raises(ValueError):
            validator.calculate_pss10_score([1, 2, 3])  # Too few items


class TestClinicalInterpreter:
    """Test clinical interpreter."""
    
    def test_interpret_phq9_result_minimal(self):
        """Test PHQ-9 interpretation for minimal severity."""
        interpreter = ClinicalInterpreter()
        
        result = PHQ9Result(
            assessment_id="test",
            persona_id="test",
            simulation_day=7,
            total_score=4,
            severity_level=SeverityLevel.MINIMAL,
            suicidal_ideation_score=0,
            depression_severity=SeverityLevel.MINIMAL
        )
        
        interpretation = interpreter.interpret_phq9_result(result)
        
        assert interpretation["severity_level"] == "minimal"
        assert interpretation["clinical_meaning"] == "Minimal depressive symptoms"
        assert interpretation["suicidal_risk"] == "low"
        assert len(interpretation["recommendations"]) > 0
    
    def test_interpret_phq9_result_severe_with_suicidal_ideation(self):
        """Test PHQ-9 interpretation for severe depression with suicidal ideation."""
        interpreter = ClinicalInterpreter()
        
        result = PHQ9Result(
            assessment_id="test",
            persona_id="test",
            simulation_day=7,
            total_score=22,
            severity_level=SeverityLevel.SEVERE,
            suicidal_ideation_score=3,
            depression_severity=SeverityLevel.SEVERE
        )
        
        interpretation = interpreter.interpret_phq9_result(result)
        
        assert interpretation["severity_level"] == "severe"
        assert interpretation["clinical_meaning"] == "Severe depressive symptoms"
        assert interpretation["suicidal_risk"] == "high"
        assert "Suicidal ideation present" in interpretation["risk_factors"]
        assert "Immediate safety assessment required" in interpretation["recommendations"]
    
    def test_interpret_gad7_result(self):
        """Test GAD-7 interpretation."""
        interpreter = ClinicalInterpreter()
        
        result = GAD7Result(
            assessment_id="test",
            persona_id="test",
            simulation_day=7,
            total_score=15,
            severity_level=SeverityLevel.MODERATE,
            anxiety_severity=SeverityLevel.MODERATE
        )
        
        interpretation = interpreter.interpret_gad7_result(result)
        
        assert interpretation["severity_level"] == "moderate"
        assert interpretation["clinical_meaning"] == "Moderate anxiety symptoms"
        assert len(interpretation["recommendations"]) > 0
    
    def test_interpret_pss10_result(self):
        """Test PSS-10 interpretation."""
        interpreter = ClinicalInterpreter()
        
        result = PSS10Result(
            assessment_id="test",
            persona_id="test",
            simulation_day=7,
            total_score=25,
            severity_level=SeverityLevel.SEVERE,
            stress_severity=SeverityLevel.SEVERE
        )
        
        interpretation = interpreter.interpret_pss10_result(result)
        
        assert interpretation["severity_level"] == "severe"
        assert interpretation["clinical_meaning"] == "Severe perceived stress"
        assert "High stress levels" in interpretation["risk_factors"]


class TestAssessmentOrchestrator:
    """Test assessment orchestrator."""
    
    @pytest.fixture
    def sample_persona(self):
        """Create a sample persona for testing."""
        baseline = PersonaBaseline(
            name="Test Person",
            age=30,
            occupation="Software Engineer",
            background="Test background",
            core_memories=["Test memory"],
            baseline_phq9=5.0,
            baseline_gad7=4.0,
            baseline_pss10=12.0,
            openness=0.7,
            conscientiousness=0.8,
            extraversion=0.6,
            agreeableness=0.7,
            neuroticism=0.4
        )
        
        state = PersonaState(
            persona_id="test_persona",
            simulation_day=7,
            last_assessment_day=0,
            current_phq9=5.0,
            current_gad7=4.0,
            current_pss10=12.0,
            emotional_state="neutral",
            stress_level=5.0
        )
        
        return Persona(baseline=baseline, state=state)
    
    @pytest.mark.asyncio
    async def test_conduct_phq9_assessment_valid(self, sample_persona):
        """Test PHQ-9 assessment with valid responses."""
        orchestrator = AssessmentOrchestrator()
        
        responses = ["0", "1", "2", "1", "0", "1", "2", "1", "0"]  # 9 responses
        
        result = await orchestrator.conduct_phq9_assessment(sample_persona, responses)
        
        assert result is not None
        assert result.assessment_type == "phq9"
        assert result.total_score == 8  # 0+1+2+1+0+1+2+1+0
        assert result.severity_level == SeverityLevel.MILD
        assert result.suicidal_ideation_score == 0
        assert "clinical_interpretation" in result.__dict__
    
    @pytest.mark.asyncio
    async def test_conduct_phq9_assessment_invalid_responses(self, sample_persona):
        """Test PHQ-9 assessment with invalid responses."""
        orchestrator = AssessmentOrchestrator()
        
        responses = ["invalid", "1", "2", "invalid", "0", "1", "2", "1", "0"]
        
        result = await orchestrator.conduct_phq9_assessment(sample_persona, responses)
        
        assert result is not None
        # Should use fallback scores (0) for invalid responses
        assert result.total_score == 7  # 0+1+2+0+0+1+2+1+0
    
    @pytest.mark.asyncio
    async def test_conduct_gad7_assessment(self, sample_persona):
        """Test GAD-7 assessment."""
        orchestrator = AssessmentOrchestrator()
        
        responses = ["1", "2", "1", "0", "1", "2", "1"]  # 7 responses
        
        result = await orchestrator.conduct_gad7_assessment(sample_persona, responses)
        
        assert result is not None
        assert result.assessment_type == "gad7"
        assert result.total_score == 8  # 1+2+1+0+1+2+1
        assert result.severity_level == SeverityLevel.MILD
    
    @pytest.mark.asyncio
    async def test_conduct_pss10_assessment(self, sample_persona):
        """Test PSS-10 assessment with reverse scoring."""
        orchestrator = AssessmentOrchestrator()
        
        responses = ["2", "3", "1", "0", "1", "2", "3", "2", "1", "2"]  # 10 responses
        
        result = await orchestrator.conduct_pss10_assessment(sample_persona, responses)
        
        assert result is not None
        assert result.assessment_type == "pss10"
        # Manual calculation with reverse scoring: 
        # Normal items (1,2,3,6,9,10): 2+3+1+2+1+2 = 11
        # Reverse items (4,5,7,8): 0->4, 1->3, 3->1, 2->2 = 4+3+1+2 = 10
        # Total: 11 + 10 = 21
        assert result.total_score == 21
    
    def test_is_assessment_due(self, sample_persona):
        """Test assessment due checking."""
        orchestrator = AssessmentOrchestrator()
        
        # Test weekly schedule
        assert orchestrator.is_assessment_due(sample_persona, "weekly") is True  # Day 7, last assessment day 0
        
        # Test biweekly schedule
        assert orchestrator.is_assessment_due(sample_persona, "biweekly") is False  # Day 7, last assessment day 0
        
        # Test monthly schedule
        assert orchestrator.is_assessment_due(sample_persona, "monthly") is False  # Day 7, last assessment day 0
    
    def test_get_next_assessment_day(self, sample_persona):
        """Test next assessment day calculation."""
        orchestrator = AssessmentOrchestrator()
        
        next_day = orchestrator.get_next_assessment_day(sample_persona, "weekly")
        assert next_day == 7  # 0 + 7
        
        next_day = orchestrator.get_next_assessment_day(sample_persona, "biweekly")
        assert next_day == 14  # 0 + 14


class TestAssessmentIntegration:
    """Integration tests for assessment system."""
    
    @pytest.fixture
    def sample_persona(self):
        """Create a sample persona for integration testing."""
        baseline = PersonaBaseline(
            name="Integration Test Person",
            age=35,
            occupation="Researcher",
            background="Test background for integration",
            core_memories=["Integration test memory"],
            baseline_phq9=6.0,
            baseline_gad7=5.0,
            baseline_pss10=14.0,
            openness=0.8,
            conscientiousness=0.7,
            extraversion=0.5,
            agreeableness=0.8,
            neuroticism=0.3
        )
        
        state = PersonaState(
            persona_id="integration_test_persona",
            simulation_day=14,
            last_assessment_day=7,
            current_phq9=6.0,
            current_gad7=5.0,
            current_pss10=14.0,
            emotional_state="neutral",
            stress_level=5.0
        )
        
        return Persona(baseline=baseline, state=state)
    
    @pytest.mark.asyncio
    async def test_full_assessment_workflow(self, sample_persona):
        """Test complete assessment workflow."""
        validator = PsychiatricScaleValidator()
        interpreter = ClinicalInterpreter()
        orchestrator = AssessmentOrchestrator()
        
        # Test PHQ-9 workflow
        phq9_responses = ["1", "2", "1", "0", "1", "2", "1", "0", "1"]
        phq9_result = await orchestrator.conduct_phq9_assessment(sample_persona, phq9_responses)
        
        assert phq9_result is not None
        assert phq9_result.total_score == 9
        
        # Test clinical interpretation
        interpretation = interpreter.interpret_phq9_result(phq9_result)
        assert interpretation["severity_level"] == "mild"
        assert interpretation["suicidal_risk"] == "low"
        
        # Test response validation
        for i, response in enumerate(phq9_responses):
            is_valid, score = validator.validate_phq9_response(response, i)
            assert is_valid is True
            assert score is not None
    
    @pytest.mark.asyncio
    async def test_assessment_with_clinical_significance(self, sample_persona):
        """Test assessment with clinical significance analysis."""
        interpreter = ClinicalInterpreter()
        
        # Create baseline result
        baseline_result = PHQ9Result(
            assessment_id="baseline",
            persona_id=sample_persona.state.persona_id,
            simulation_day=0,
            total_score=6.0,
            severity_level=SeverityLevel.MILD,
            suicidal_ideation_score=0,
            depression_severity=SeverityLevel.MILD
        )
        
        # Create current result with significant change
        current_result = PHQ9Result(
            assessment_id="current",
            persona_id=sample_persona.state.persona_id,
            simulation_day=14,
            total_score=18.0,  # Significant increase
            severity_level=SeverityLevel.MODERATE,
            suicidal_ideation_score=1,
            depression_severity=SeverityLevel.MODERATE
        )
        
        # Assess clinical significance
        significance = interpreter.assess_clinical_significance(
            current_result, baseline_result
        )
        
        assert significance["is_clinically_significant"] is True
        assert significance["significance_level"] == "moderate"
        assert significance["change_magnitude"] == 12.0
        assert significance["trend_direction"] == "increasing"
        assert len(significance["clinical_recommendations"]) > 0 