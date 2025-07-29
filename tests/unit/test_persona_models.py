"""
Unit tests for persona models.
"""

import pytest
from datetime import datetime

from src.models.persona import Persona, PersonaBaseline, PersonaState, PersonalityTrait


class TestPersonaBaseline:
    """Test PersonaBaseline model."""
    
    def test_valid_baseline(self):
        """Test creating a valid persona baseline."""
        baseline = PersonaBaseline(
            name="Test Persona",
            age=30,
            occupation="Software Engineer",
            background="Test background",
            openness=0.7,
            conscientiousness=0.8,
            extraversion=0.4,
            agreeableness=0.6,
            neuroticism=0.5,
            baseline_phq9=5.0,
            baseline_gad7=4.0,
            baseline_pss10=12.0
        )
        
        assert baseline.name == "Test Persona"
        assert baseline.age == 30
        assert baseline.get_trait(PersonalityTrait.OPENNESS) == 0.7
        assert len(baseline.get_traits_dict()) == 5
    
    def test_invalid_age(self):
        """Test that invalid age raises validation error."""
        with pytest.raises(ValueError):
            PersonaBaseline(
                name="Test",
                age=15,  # Too young
                occupation="Test",
                background="Test",
                openness=0.5,
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5,
                baseline_phq9=5.0,
                baseline_gad7=4.0,
                baseline_pss10=12.0
            )
    
    def test_invalid_trait_values(self):
        """Test that invalid trait values raise validation error."""
        with pytest.raises(ValueError):
            PersonaBaseline(
                name="Test",
                age=30,
                occupation="Test",
                background="Test",
                openness=1.5,  # Too high
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5,
                baseline_phq9=5.0,
                baseline_gad7=4.0,
                baseline_pss10=12.0
            )


class TestPersonaState:
    """Test PersonaState model."""
    
    def test_valid_state(self):
        """Test creating a valid persona state."""
        state = PersonaState(
            persona_id="test_persona",
            simulation_day=5,
            current_phq9=6.0,
            current_gad7=5.0,
            current_pss10=13.0
        )
        
        assert state.persona_id == "test_persona"
        assert state.simulation_day == 5
        assert state.current_phq9 == 6.0
        assert state.drift_magnitude == 0.0
    
    def test_add_event(self):
        """Test adding events to state."""
        state = PersonaState(persona_id="test")
        state.add_event("Test event")
        
        assert len(state.recent_events) == 1
        assert state.recent_events[0] == "Test event"
    
    def test_update_stress_level(self):
        """Test updating stress level."""
        state = PersonaState(persona_id="test")
        state.update_stress_level(7.5)
        
        assert state.stress_level == 7.5
    
    def test_stress_level_bounds(self):
        """Test stress level bounds enforcement."""
        state = PersonaState(persona_id="test")
        state.update_stress_level(15.0)  # Too high
        assert state.stress_level == 10.0
        
        state.update_stress_level(-5.0)  # Too low
        assert state.stress_level == 0.0


class TestPersona:
    """Test Persona model."""
    
    def test_valid_persona(self):
        """Test creating a valid persona."""
        baseline = PersonaBaseline(
            name="Test Persona",
            age=30,
            occupation="Test",
            background="Test",
            openness=0.7,
            conscientiousness=0.8,
            extraversion=0.4,
            agreeableness=0.6,
            neuroticism=0.5,
            baseline_phq9=5.0,
            baseline_gad7=4.0,
            baseline_pss10=12.0
        )
        
        state = PersonaState(persona_id="test_persona")
        
        persona = Persona(baseline=baseline, state=state)
        
        assert persona.baseline.name == "Test Persona"
        assert persona.state.persona_id == "persona_test_persona"
    
    def test_get_current_traits(self):
        """Test getting current traits with drift."""
        baseline = PersonaBaseline(
            name="Test",
            age=30,
            occupation="Test",
            background="Test",
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
            baseline_phq9=5.0,
            baseline_gad7=4.0,
            baseline_pss10=12.0
        )
        
        state = PersonaState(
            persona_id="test",
            trait_changes={"openness": 0.1, "neuroticism": -0.1}
        )
        
        persona = Persona(baseline=baseline, state=state)
        current_traits = persona.get_current_traits()
        
        assert current_traits["openness"] == 0.6  # 0.5 + 0.1
        assert current_traits["neuroticism"] == 0.4  # 0.5 - 0.1
        assert current_traits["extraversion"] == 0.5  # No change
    
    def test_calculate_drift_magnitude(self):
        """Test drift magnitude calculation."""
        baseline = PersonaBaseline(
            name="Test",
            age=30,
            occupation="Test",
            background="Test",
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
            baseline_phq9=5.0,
            baseline_gad7=4.0,
            baseline_pss10=12.0
        )
        
        state = PersonaState(
            persona_id="test",
            trait_changes={"openness": 0.2, "neuroticism": -0.1, "extraversion": 0.0}
        )
        
        persona = Persona(baseline=baseline, state=state)
        drift_magnitude = persona.calculate_drift_magnitude()
        
        # Average of |0.2| + |-0.1| + |0.0| + |0.0| + |0.0| = 0.06 (5 traits total)
        assert drift_magnitude == pytest.approx(0.06, abs=1e-6)
    
    def test_is_assessment_due(self):
        """Test assessment due checking."""
        baseline = PersonaBaseline(
            name="Test",
            age=30,
            occupation="Test",
            background="Test",
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
            baseline_phq9=5.0,
            baseline_gad7=4.0,
            baseline_pss10=12.0
        )
        
        # Last assessment was 5 days ago, current day is 12
        state = PersonaState(
            persona_id="test",
            simulation_day=12,
            last_assessment_day=5
        )
        
        persona = Persona(baseline=baseline, state=state)
        
        # 7 days have passed, assessment is due
        assert persona.is_assessment_due(assessment_interval_days=7) is True
        
        # Change to 6 days ago
        state.last_assessment_day = 6
        assert persona.is_assessment_due(assessment_interval_days=7) is False
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        baseline = PersonaBaseline(
            name="Test",
            age=30,
            occupation="Test",
            background="Test",
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
            baseline_phq9=5.0,
            baseline_gad7=4.0,
            baseline_pss10=12.0
        )
        
        state = PersonaState(persona_id="test")
        persona = Persona(baseline=baseline, state=state)
        
        # Convert to dict
        persona_dict = persona.to_dict()
        
        # Convert back from dict
        restored_persona = Persona.from_dict(persona_dict)
        
        assert restored_persona.baseline.name == persona.baseline.name
        assert restored_persona.state.persona_id == persona.state.persona_id
        assert restored_persona.version == persona.version 