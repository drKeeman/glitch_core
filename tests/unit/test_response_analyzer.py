"""
Unit tests for response analyzer.
"""

import pytest
from datetime import datetime

from src.assessment.response_analyzer import ResponseAnalyzer
from src.models.assessment import (
    AssessmentResult, PHQ9Result, GAD7Result, PSS10Result, SeverityLevel
)


class TestResponseAnalyzer:
    """Test response analyzer functionality."""
    
    def test_analyze_response_consistency_valid(self):
        """Test response consistency analysis with valid responses."""
        analyzer = ResponseAnalyzer()
        
        responses = ["1", "2", "1", "0", "1", "2", "1", "0", "1"]  # PHQ-9 responses
        
        analysis = analyzer.analyze_response_consistency(responses, "phq9")
        
        assert analysis["consistency_score"] == 1.0
        assert analysis["consistency_level"] == "high"
        assert analysis["valid_responses"] == 9
        assert analysis["invalid_responses"] == 0
        assert analysis["response_pattern"] == "mixed"
        assert len(analysis["issues"]) == 0
    
    def test_analyze_response_consistency_invalid(self):
        """Test response consistency analysis with invalid responses."""
        analyzer = ResponseAnalyzer()
        
        responses = ["invalid", "2", "1", "invalid", "1", "2", "1", "0", "1"]
        
        analysis = analyzer.analyze_response_consistency(responses, "phq9")
        
        assert analysis["consistency_score"] == 7/9  # 7 valid out of 9
        assert analysis["consistency_level"] == "medium"
        assert analysis["valid_responses"] == 7
        assert analysis["invalid_responses"] == 2
        assert len(analysis["issues"]) == 2
    
    def test_analyze_response_consistency_extreme_patterns(self):
        """Test response consistency analysis with extreme patterns."""
        analyzer = ResponseAnalyzer()
        
        # All minimal responses
        all_minimal = ["0", "0", "0", "0", "0", "0", "0", "0", "0"]
        analysis = analyzer.analyze_response_consistency(all_minimal, "phq9")
        assert analysis["response_pattern"] == "all_minimal"
        
        # All severe responses
        all_severe = ["3", "3", "3", "3", "3", "3", "3", "3", "3"]
        analysis = analyzer.analyze_response_consistency(all_severe, "phq9")
        assert analysis["response_pattern"] == "all_severe"
    
    def test_validate_single_response_phq9(self):
        """Test single response validation for PHQ-9."""
        analyzer = ResponseAnalyzer()
        
        # Test valid numeric responses
        assert analyzer._validate_single_response("0", "phq9", 0) == (True, 0)
        assert analyzer._validate_single_response("1", "phq9", 0) == (True, 1)
        assert analyzer._validate_single_response("2", "phq9", 0) == (True, 2)
        assert analyzer._validate_single_response("3", "phq9", 0) == (True, 3)
        
        # Test valid text responses
        assert analyzer._validate_single_response("not at all", "phq9", 0) == (True, 0)
        assert analyzer._validate_single_response("several days", "phq9", 0) == (True, 1)
        assert analyzer._validate_single_response("more than half the days", "phq9", 0) == (True, 2)
        assert analyzer._validate_single_response("nearly every day", "phq9", 0) == (True, 3)
        
        # Test invalid responses
        assert analyzer._validate_single_response("invalid", "phq9", 0) == (False, None)
        assert analyzer._validate_single_response("4", "phq9", 0) == (False, None)
        assert analyzer._validate_single_response("", "phq9", 0) == (False, None)
    
    def test_validate_single_response_gad7(self):
        """Test single response validation for GAD-7."""
        analyzer = ResponseAnalyzer()
        
        # Test valid responses
        assert analyzer._validate_single_response("1", "gad7", 0) == (True, 1)
        assert analyzer._validate_single_response("sometimes", "gad7", 0) == (True, 1)
        assert analyzer._validate_single_response("often", "gad7", 0) == (True, 2)
        
        # Test invalid responses
        assert analyzer._validate_single_response("invalid", "gad7", 0) == (False, None)
        assert analyzer._validate_single_response("4", "gad7", 0) == (False, None)
    
    def test_validate_single_response_pss10(self):
        """Test single response validation for PSS-10."""
        analyzer = ResponseAnalyzer()
        
        # Test valid responses
        assert analyzer._validate_single_response("2", "pss10", 0) == (True, 2)
        assert analyzer._validate_single_response("sometimes", "pss10", 0) == (True, 2)
        assert analyzer._validate_single_response("very often", "pss10", 0) == (True, 4)
        
        # Test invalid responses
        assert analyzer._validate_single_response("invalid", "pss10", 0) == (False, None)
        assert analyzer._validate_single_response("5", "pss10", 0) == (False, None)
    
    def test_analyze_response_pattern(self):
        """Test response pattern analysis."""
        analyzer = ResponseAnalyzer()
        
        # Test PHQ-9 patterns
        scores = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert analyzer._analyze_response_pattern(scores, "phq9") == "all_minimal"
        
        scores = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        assert analyzer._analyze_response_pattern(scores, "phq9") == "all_severe"
        
        scores = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        assert analyzer._analyze_response_pattern(scores, "phq9") == "mostly_high"
        
        scores = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert analyzer._analyze_response_pattern(scores, "phq9") == "mostly_low"
        
        scores = [0, 1, 2, 1, 0, 1, 2, 1, 0]
        assert analyzer._analyze_response_pattern(scores, "phq9") == "mixed"
        
        # Test PSS-10 patterns
        scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert analyzer._analyze_response_pattern(scores, "pss10") == "all_minimal"
        
        scores = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        assert analyzer._analyze_response_pattern(scores, "pss10") == "all_severe"
    
    def test_calculate_variability(self):
        """Test response variability calculation."""
        analyzer = ResponseAnalyzer()
        
        # Test with no variability
        scores = [2, 2, 2, 2, 2]
        variability = analyzer._calculate_variability(scores)
        assert variability == 0.0
        
        # Test with some variability
        scores = [1, 2, 3, 2, 1]
        variability = analyzer._calculate_variability(scores)
        assert variability > 0.0
        
        # Test with single score
        scores = [2]
        variability = analyzer._calculate_variability(scores)
        assert variability == 0.0
    
    def test_detect_response_anomalies_phq9(self):
        """Test anomaly detection for PHQ-9 results."""
        analyzer = ResponseAnalyzer()
        
        # Create PHQ-9 result with suicidal ideation
        result = PHQ9Result(
            assessment_id="test",
            persona_id="test",
            simulation_day=7,
            total_score=22,
            severity_level=SeverityLevel.SEVERE,
            suicidal_ideation_score=3,  # Critical
            depression_severity=SeverityLevel.SEVERE,
            raw_responses=["3", "3", "3", "3", "3", "3", "3", "3", "3"],
            parsed_scores=[3, 3, 3, 3, 3, 3, 3, 3, 3]
        )
        
        anomalies = analyzer.detect_response_anomalies(result)
        
        # Should detect suicidal ideation and severe depression
        anomaly_types = [a["type"] for a in anomalies]
        assert "suicidal_ideation" in anomaly_types
        assert "severe_depression" in anomaly_types
        
        # Check for critical severity
        critical_anomalies = [a for a in anomalies if a["severity"] == "critical"]
        assert len(critical_anomalies) > 0
    
    def test_detect_response_anomalies_invalid_scores(self):
        """Test anomaly detection for invalid scores."""
        analyzer = ResponseAnalyzer()
        
        # Create result with invalid scores
        result = PHQ9Result(
            assessment_id="test",
            persona_id="test",
            simulation_day=7,
            total_score=15,
            severity_level=SeverityLevel.MODERATE,
            suicidal_ideation_score=0,
            depression_severity=SeverityLevel.MODERATE,
            raw_responses=["1", "2", "1", "0", "1", "2", "1", "0", "1"],
            parsed_scores=[1, 2, 1, 0, 1, 2, 1, 0, 1]  # Valid scores
        )
        
        anomalies = analyzer.detect_response_anomalies(result)
        
        # Should not have invalid score anomalies
        anomaly_types = [a["type"] for a in anomalies]
        assert "invalid_scores" not in anomaly_types
    
    def test_validate_assessment_result_valid(self):
        """Test assessment result validation with valid result."""
        analyzer = ResponseAnalyzer()
        
        result = PHQ9Result(
            assessment_id="test",
            persona_id="test",
            simulation_day=7,
            total_score=10,
            severity_level=SeverityLevel.MILD,
            suicidal_ideation_score=0,
            depression_severity=SeverityLevel.MILD,
            raw_responses=["1", "2", "1", "0", "1", "2", "1", "0", "1"],
            parsed_scores=[1, 2, 1, 0, 1, 2, 1, 0, 1]
        )
        
        validation = analyzer.validate_assessment_result(result)
        
        assert validation["is_valid"] is True
        assert len(validation["issues"]) == 0
        assert validation["consistency_analysis"]["consistency_level"] == "high"
        assert len(validation["anomalies"]) == 0
    
    def test_validate_assessment_result_invalid(self):
        """Test assessment result validation with invalid result."""
        analyzer = ResponseAnalyzer()
        
        # Create result with missing assessment ID
        result = PHQ9Result(
            assessment_id="",  # Invalid
            persona_id="test",
            simulation_day=7,
            total_score=10,
            severity_level=SeverityLevel.MILD,
            suicidal_ideation_score=0,
            depression_severity=SeverityLevel.MILD,
            raw_responses=["1", "2", "1", "0", "1", "2", "1", "0", "1"],
            parsed_scores=[1, 2, 1, 0, 1, 2, 1, 0, 1]
        )
        
        validation = analyzer.validate_assessment_result(result)
        
        assert validation["is_valid"] is False
        assert "Missing assessment ID" in validation["issues"]
    
    def test_validate_assessment_result_low_consistency(self):
        """Test assessment result validation with low consistency."""
        analyzer = ResponseAnalyzer()
        
        result = PHQ9Result(
            assessment_id="test",
            persona_id="test",
            simulation_day=7,
            total_score=10,
            severity_level=SeverityLevel.MILD,
            suicidal_ideation_score=0,
            depression_severity=SeverityLevel.MILD,
            raw_responses=["invalid", "invalid", "1", "0", "1", "2", "1", "0", "1"],
            parsed_scores=[0, 0, 1, 0, 1, 2, 1, 0, 1]  # Fallback scores
        )
        
        validation = analyzer.validate_assessment_result(result)
        
        assert validation["is_valid"] is True  # Still valid
        assert validation["consistency_analysis"]["consistency_level"] in ["low", "poor"]
        assert len(validation["warnings"]) > 0
        assert "Consider re-administering assessment" in validation["recommendations"]


class TestResponseAnalyzerIntegration:
    """Integration tests for response analyzer."""
    
    def test_full_analysis_workflow(self):
        """Test complete response analysis workflow."""
        analyzer = ResponseAnalyzer()
        
        # Create a realistic assessment result
        result = PHQ9Result(
            assessment_id="test_workflow",
            persona_id="test_persona",
            simulation_day=14,
            total_score=12,
            severity_level=SeverityLevel.MILD,
            suicidal_ideation_score=1,
            depression_severity=SeverityLevel.MILD,
            raw_responses=["1", "2", "1", "0", "1", "2", "1", "0", "1"],
            parsed_scores=[1, 2, 1, 0, 1, 2, 1, 0, 1]
        )
        
        # Analyze consistency
        consistency = analyzer.analyze_response_consistency(
            result.raw_responses, result.assessment_type
        )
        
        assert consistency["consistency_score"] == 1.0
        assert consistency["consistency_level"] == "high"
        assert consistency["response_pattern"] == "mixed"
        
        # Detect anomalies
        anomalies = analyzer.detect_response_anomalies(result)
        
        # Should not have critical anomalies for this result
        critical_anomalies = [a for a in anomalies if a["severity"] == "critical"]
        assert len(critical_anomalies) == 0
        
        # Validate result
        validation = analyzer.validate_assessment_result(result)
        
        assert validation["is_valid"] is True
        assert validation["consistency_analysis"]["consistency_level"] == "high"
        assert len(validation["anomalies"]) == 0
    
    def test_analysis_with_extreme_case(self):
        """Test analysis with extreme case (severe depression with suicidal ideation)."""
        analyzer = ResponseAnalyzer()
        
        # Create severe case
        result = PHQ9Result(
            assessment_id="extreme_case",
            persona_id="test_persona",
            simulation_day=14,
            total_score=25,
            severity_level=SeverityLevel.SEVERE,
            suicidal_ideation_score=3,  # Critical
            depression_severity=SeverityLevel.SEVERE,
            raw_responses=["3", "3", "3", "3", "3", "3", "3", "3", "3"],
            parsed_scores=[3, 3, 3, 3, 3, 3, 3, 3, 3]
        )
        
        # Analyze consistency
        consistency = analyzer.analyze_response_consistency(
            result.raw_responses, result.assessment_type
        )
        
        assert consistency["response_pattern"] == "all_severe"
        assert consistency["consistency_score"] == 1.0
        
        # Detect anomalies
        anomalies = analyzer.detect_response_anomalies(result)
        
        # Should have critical anomalies
        critical_anomalies = [a for a in anomalies if a["severity"] == "critical"]
        assert len(critical_anomalies) > 0
        
        # Validate result
        validation = analyzer.validate_assessment_result(result)
        
        # Should be valid but with critical anomalies
        assert validation["is_valid"] is False  # Critical anomalies make it invalid
        assert len(validation["anomalies"]) > 0
        assert any(a["severity"] == "critical" for a in validation["anomalies"]) 