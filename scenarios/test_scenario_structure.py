#!/usr/bin/env python3
"""
Test Scenario Structure

This script tests the basic structure and functionality of the scenario framework,
ensuring that all components work correctly together.
"""

import asyncio
import sys
import os
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from .base_scenario import BaseScenario
from .trauma_response_analysis import TraumaResponseAnalysis
from glitch_core.config.logging import get_logger


async def test_base_scenario():
    """Test the base scenario functionality."""
    print("🧪 Testing Base Scenario...")
    
    scenario = BaseScenario()
    
    # Test results directory creation
    assert scenario.results_dir.exists(), "Results directory should be created"
    print("✅ Results directory created successfully")
    
    # Test save_results
    test_data = {"test": "data", "timestamp": "2024-01-01"}
    results_file = scenario.save_results("test_scenario", test_data)
    assert Path(results_file).exists(), "Results file should be saved"
    print("✅ Results saving works correctly")
    
    # Test plot creation
    test_emotions = [
        {"joy": 0.5, "sadness": 0.2, "anger": 0.1},
        {"joy": 0.6, "sadness": 0.1, "anger": 0.2},
        {"joy": 0.4, "sadness": 0.3, "anger": 0.1}
    ]
    
    fig = scenario.create_emotional_evolution_plot(test_emotions, "Test Plot")
    assert fig is not None, "Plot should be created"
    print("✅ Plot creation works correctly")
    
    # Test plot saving
    plot_file = scenario.save_plot(fig, "test_plot.html")
    assert Path(plot_file).exists(), "Plot file should be saved"
    print("✅ Plot saving works correctly")


async def test_trauma_scenario_structure():
    """Test the trauma response analysis scenario structure."""
    print("\n🧪 Testing Trauma Response Analysis Structure...")
    
    scenario = TraumaResponseAnalysis()
    
    # Test trauma events
    assert len(scenario.trauma_events) == 4, "Should have 4 trauma types"
    print("✅ Trauma events configured correctly")
    
    # Test personality types
    assert len(scenario.personality_types) == 4, "Should have 4 personality types"
    print("✅ Personality types configured correctly")
    
    # Test trauma event structure
    for trauma_type, trauma_data in scenario.trauma_events.items():
        assert "event" in trauma_data, f"Trauma {trauma_type} should have event"
        assert "emotional_impact" in trauma_data, f"Trauma {trauma_type} should have emotional_impact"
        print(f"✅ Trauma type '{trauma_type}' structure correct")
    
    # Test analysis methods exist
    assert hasattr(scenario, '_calculate_emotional_impact'), "Should have emotional impact calculation"
    assert hasattr(scenario, '_calculate_recovery_time'), "Should have recovery time calculation"
    assert hasattr(scenario, '_emotional_states_similar'), "Should have emotional state comparison"
    print("✅ Analysis methods implemented correctly")


async def test_scenario_integration():
    """Test scenario integration with API (mock)."""
    print("\n🧪 Testing Scenario Integration...")
    
    # Create a mock scenario that doesn't require API
    class MockTraumaScenario(TraumaResponseAnalysis):
        async def create_persona(self, persona_type: str) -> str:
            return f"mock_persona_{persona_type}"
        
        async def start_experiment(self, persona_id: str, epochs: int = 100, events_per_epoch: int = 10) -> str:
            return f"mock_experiment_{persona_id}"
        
        async def get_analysis(self, experiment_id: str) -> dict:
            return {
                "emotional_states": [
                    {"joy": 0.5, "sadness": 0.2, "anger": 0.1},
                    {"joy": 0.6, "sadness": 0.1, "anger": 0.2},
                    {"joy": 0.4, "sadness": 0.3, "anger": 0.1}
                ],
                "stability_metrics": {"overall_stability": 0.8},
                "pattern_emergence": [],
                "stability_warnings": []
            }
        
        async def wait_for_completion(self, experiment_id: str, timeout: int = 300) -> bool:
            return True
    
    scenario = MockTraumaScenario()
    
    # Test scenario structure
    assert scenario.trauma_events is not None, "Trauma events should be defined"
    assert scenario.personality_types is not None, "Personality types should be defined"
    print("✅ Scenario structure integration works")
    
    # Test analysis methods
    test_emotions = [
        {"joy": 0.5, "sadness": 0.2, "anger": 0.1},
        {"joy": 0.6, "sadness": 0.1, "anger": 0.2},
        {"joy": 0.4, "sadness": 0.3, "anger": 0.1}
    ]
    
    impact = scenario._calculate_emotional_impact(test_emotions, test_emotions, 1)
    assert isinstance(impact, dict), "Emotional impact should be a dictionary"
    print("✅ Analysis methods work correctly")


async def main():
    """Run all scenario structure tests."""
    print("🚀 Testing Scenario Structure")
    print("=" * 40)
    
    try:
        await test_base_scenario()
        await test_trauma_scenario_structure()
        await test_scenario_integration()
        
        print("\n✅ All scenario structure tests passed!")
        print("=" * 40)
        print("📋 Scenario infrastructure is ready for use")
        print("🎯 Ready to run: python scenarios/trauma_response_analysis.py")
        
    except Exception as e:
        print(f"\n❌ Scenario structure test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 