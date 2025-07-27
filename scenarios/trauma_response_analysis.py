"""
Scenario 1: Trauma Response Analysis

This scenario demonstrates how different personality types respond to traumatic events,
showing the temporal evolution of emotional states and stability patterns.
"""

import asyncio
import sys
import os
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from .base_scenario import BaseScenario
from glitch_core.config.logging import get_logger


class TraumaResponseAnalysis(BaseScenario):
    """
    Trauma Response Analysis Scenario
    
    This scenario demonstrates:
    1. Baseline personality establishment
    2. Trauma injection at specific epochs
    3. Before/after pattern analysis
    4. Stability boundary detection
    5. Recovery pattern identification
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        super().__init__(api_base_url)
        self.logger = get_logger(__name__)
        
        # Define trauma events with different intensities
        self.trauma_events = {
            "mild_social_rejection": {
                "event": "Experienced mild social rejection from a group",
                "emotional_impact": {"anxiety": 0.4, "sadness": 0.3, "anger": 0.2}
            },
            "moderate_failure": {
                "event": "Failed an important project despite best efforts",
                "emotional_impact": {"anxiety": 0.6, "sadness": 0.5, "anger": 0.3, "shame": 0.4}
            },
            "severe_betrayal": {
                "event": "Discovered close friend betrayed trust in significant way",
                "emotional_impact": {"anger": 0.8, "sadness": 0.7, "anxiety": 0.6, "shame": 0.5}
            },
            "extreme_loss": {
                "event": "Experienced devastating personal loss",
                "emotional_impact": {"sadness": 0.9, "anxiety": 0.7, "anger": 0.4, "despair": 0.8}
            }
        }
        
        # Define personality types to test
        self.personality_types = [
            "resilient_optimist",
            "anxious_overthinker", 
            "stoic_philosopher",
            "creative_volatile"
        ]
    
    async def run_baseline_simulation(self, persona_id: str, epochs: int = 50) -> Dict[str, Any]:
        """Run baseline simulation without trauma."""
        self.logger.info("Running baseline simulation...")
        
        experiment_id = await self.start_experiment(
            persona_id=persona_id,
            epochs=epochs,
            events_per_epoch=10
        )
        
        # Wait for completion
        success = await self.wait_for_completion(experiment_id)
        if not success:
            raise RuntimeError(f"Baseline simulation failed for experiment {experiment_id}")
        
        # Get analysis results
        analysis = await self.get_analysis(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "analysis": analysis,
            "epochs": epochs
        }
    
    async def run_trauma_simulation(self, persona_id: str, trauma_type: str, 
                                  injection_epoch: int = 25, epochs: int = 50) -> Dict[str, Any]:
        """Run simulation with trauma injection."""
        self.logger.info(f"Running trauma simulation with {trauma_type}...")
        
        experiment_id = await self.start_experiment(
            persona_id=persona_id,
            epochs=epochs,
            events_per_epoch=10
        )
        
        # Wait a bit for the simulation to start
        await asyncio.sleep(5)
        
        # Inject trauma at specified epoch
        trauma_event = self.trauma_events[trauma_type]
        intervention = await self.inject_intervention(
            experiment_id=experiment_id,
            event=trauma_event["event"],
            emotional_impact=trauma_event["emotional_impact"]
        )
        
        self.logger.info(f"Injected trauma at epoch {injection_epoch}")
        
        # Wait for completion
        success = await self.wait_for_completion(experiment_id)
        if not success:
            raise RuntimeError(f"Trauma simulation failed for experiment {experiment_id}")
        
        # Get analysis results
        analysis = await self.get_analysis(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "analysis": analysis,
            "trauma_type": trauma_type,
            "injection_epoch": injection_epoch,
            "epochs": epochs,
            "intervention": intervention
        }
    
    async def run_scenario(self) -> Dict[str, Any]:
        """Run the complete trauma response analysis scenario."""
        self.logger.info("Starting Trauma Response Analysis Scenario")
        
        results = {
            "scenario": "trauma_response_analysis",
            "timestamp": datetime.now().isoformat(),
            "personalities": {},
            "comparisons": {},
            "insights": {}
        }
        
        # Test each personality type
        for persona_type in self.personality_types:
            self.logger.info(f"Testing personality: {persona_type}")
            
            # Create persona
            persona_id = await self.create_persona(persona_type)
            
            persona_results = {
                "persona_id": persona_id,
                "persona_type": persona_type,
                "baseline": {},
                "trauma_tests": {}
            }
            
            # Run baseline simulation
            baseline = await self.run_baseline_simulation(persona_id, epochs=50)
            persona_results["baseline"] = baseline
            
            # Test different trauma types
            for trauma_type in self.trauma_events.keys():
                self.logger.info(f"Testing {persona_type} with {trauma_type}")
                
                # Create new persona for trauma test
                trauma_persona_id = await self.create_persona(persona_type)
                
                trauma_result = await self.run_trauma_simulation(
                    persona_id=trauma_persona_id,
                    trauma_type=trauma_type,
                    injection_epoch=25,
                    epochs=50
                )
                
                persona_results["trauma_tests"][trauma_type] = trauma_result
            
            results["personalities"][persona_type] = persona_results
        
        # Save results
        results_file = self.save_results("trauma_response_analysis", results)
        results["results_file"] = results_file
        
        return results
    
    async def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the trauma response results."""
        self.logger.info("Analyzing trauma response results...")
        
        analysis = {
            "scenario": "trauma_response_analysis",
            "timestamp": datetime.now().isoformat(),
            "personality_insights": {},
            "trauma_insights": {},
            "stability_analysis": {},
            "recovery_patterns": {},
            "visualizations": {}
        }
        
        # Analyze each personality type
        for persona_type, persona_data in results["personalities"].items():
            self.logger.info(f"Analyzing {persona_type}...")
            
            baseline_analysis = persona_data["baseline"]["analysis"]
            personality_insights = {
                "baseline_stability": baseline_analysis.get("stability_metrics", {}),
                "baseline_patterns": baseline_analysis.get("pattern_emergence", []),
                "trauma_responses": {}
            }
            
            # Analyze trauma responses
            for trauma_type, trauma_data in persona_data["trauma_tests"].items():
                trauma_analysis = trauma_data["analysis"]
                
                # Calculate stability changes
                baseline_stability = baseline_analysis.get("stability_metrics", {}).get("overall_stability", 1.0)
                trauma_stability = trauma_analysis.get("stability_metrics", {}).get("overall_stability", 1.0)
                stability_change = trauma_stability - baseline_stability
                
                # Calculate emotional impact
                baseline_emotions = baseline_analysis.get("emotional_states", [])
                trauma_emotions = trauma_analysis.get("emotional_states", [])
                
                emotional_impact = self._calculate_emotional_impact(
                    baseline_emotions, trauma_emotions, trauma_data["injection_epoch"]
                )
                
                personality_insights["trauma_responses"][trauma_type] = {
                    "stability_change": stability_change,
                    "emotional_impact": emotional_impact,
                    "breakdown_points": trauma_analysis.get("stability_warnings", []),
                    "recovery_time": self._calculate_recovery_time(trauma_emotions, trauma_data["injection_epoch"])
                }
            
            analysis["personality_insights"][persona_type] = personality_insights
            
            # Create visualizations
            self._create_personality_visualizations(persona_data, persona_type, analysis)
        
        # Cross-personality analysis
        analysis["trauma_insights"] = self._analyze_trauma_effects(results)
        analysis["stability_analysis"] = self._analyze_stability_patterns(results)
        analysis["recovery_patterns"] = self._analyze_recovery_patterns(results)
        
        # Save analysis
        analysis_file = self.save_results("trauma_response_analysis_analysis", analysis)
        analysis["analysis_file"] = analysis_file
        
        return analysis
    
    def _calculate_emotional_impact(self, baseline_emotions: List[Dict], 
                                  trauma_emotions: List[Dict], 
                                  injection_epoch: int) -> Dict[str, float]:
        """Calculate emotional impact of trauma."""
        if not baseline_emotions or not trauma_emotions:
            return {}
        
        # Get baseline emotional averages
        baseline_df = pd.DataFrame(baseline_emotions)
        baseline_avg = baseline_df.mean()
        
        # Get post-trauma emotional averages
        post_trauma_df = pd.DataFrame(trauma_emotions[injection_epoch:])
        post_trauma_avg = post_trauma_df.mean()
        
        # Calculate impact
        impact = {}
        for emotion in baseline_avg.index:
            if emotion in post_trauma_avg.index:
                impact[emotion] = post_trauma_avg[emotion] - baseline_avg[emotion]
        
        return impact
    
    def _calculate_recovery_time(self, emotional_states: List[Dict], 
                                injection_epoch: int) -> int:
        """Calculate recovery time in epochs."""
        if len(emotional_states) <= injection_epoch:
            return -1
        
        # Get baseline emotional state
        baseline_state = emotional_states[injection_epoch - 1]
        
        # Find when emotional state returns to baseline
        for i, state in enumerate(emotional_states[injection_epoch:], injection_epoch):
            if self._emotional_states_similar(state, baseline_state, threshold=0.1):
                return i - injection_epoch
        
        return len(emotional_states) - injection_epoch
    
    def _emotional_states_similar(self, state1: Dict, state2: Dict, threshold: float = 0.1) -> bool:
        """Check if two emotional states are similar."""
        for emotion in state1:
            if emotion in state2:
                if abs(state1[emotion] - state2[emotion]) > threshold:
                    return False
        return True
    
    def _create_personality_visualizations(self, persona_data: Dict, 
                                         persona_type: str, 
                                         analysis: Dict) -> None:
        """Create visualizations for a personality type."""
        baseline_emotions = persona_data["baseline"]["analysis"].get("emotional_states", [])
        
        # Create baseline emotional evolution plot
        if baseline_emotions:
            fig = self.create_emotional_evolution_plot(
                baseline_emotions,
                title=f"{persona_type.replace('_', ' ').title()} - Baseline Emotional Evolution"
            )
            plot_file = self.save_plot(fig, f"{persona_type}_baseline_emotions.html")
            analysis["visualizations"][f"{persona_type}_baseline"] = plot_file
        
        # Create trauma comparison plots
        for trauma_type, trauma_data in persona_data["trauma_tests"].items():
            trauma_emotions = trauma_data["analysis"].get("emotional_states", [])
            
            if baseline_emotions and trauma_emotions:
                fig = self.create_comparison_plot(
                    baseline_emotions,
                    trauma_emotions,
                    title=f"{persona_type.replace('_', ' ').title()} - {trauma_type.replace('_', ' ').title()}"
                )
                plot_file = self.save_plot(fig, f"{persona_type}_{trauma_type}_comparison.html")
                analysis["visualizations"][f"{persona_type}_{trauma_type}"] = plot_file
    
    def _analyze_trauma_effects(self, results: Dict) -> Dict:
        """Analyze the effects of different trauma types across personalities."""
        trauma_effects = {}
        
        for trauma_type in self.trauma_events.keys():
            trauma_effects[trauma_type] = {
                "personality_responses": {},
                "most_affected": None,
                "least_affected": None,
                "average_stability_change": 0.0
            }
            
            stability_changes = []
            
            for persona_type, persona_data in results["personalities"].items():
                if trauma_type in persona_data["trauma_tests"]:
                    trauma_response = persona_data["trauma_tests"][trauma_type]
                    stability_change = trauma_response["analysis"].get("stability_metrics", {}).get("overall_stability", 1.0)
                    stability_changes.append((persona_type, stability_change))
            
            if stability_changes:
                # Sort by stability change (most negative = most affected)
                stability_changes.sort(key=lambda x: x[1])
                trauma_effects[trauma_type]["most_affected"] = stability_changes[0][0]
                trauma_effects[trauma_type]["least_affected"] = stability_changes[-1][0]
                trauma_effects[trauma_type]["average_stability_change"] = sum(x[1] for x in stability_changes) / len(stability_changes)
        
        return trauma_effects
    
    def _analyze_stability_patterns(self, results: Dict) -> Dict:
        """Analyze stability patterns across personalities."""
        stability_patterns = {}
        
        for persona_type, persona_data in results["personalities"].items():
            baseline_stability = persona_data["baseline"]["analysis"].get("stability_metrics", {}).get("overall_stability", 1.0)
            
            stability_patterns[persona_type] = {
                "baseline_stability": baseline_stability,
                "trauma_sensitivity": {},
                "recovery_capacity": {}
            }
            
            for trauma_type, trauma_data in persona_data["trauma_tests"].items():
                trauma_stability = trauma_data["analysis"].get("stability_metrics", {}).get("overall_stability", 1.0)
                stability_patterns[persona_type]["trauma_sensitivity"][trauma_type] = baseline_stability - trauma_stability
        
        return stability_patterns
    
    def _analyze_recovery_patterns(self, results: Dict) -> Dict:
        """Analyze recovery patterns across personalities."""
        recovery_patterns = {}
        
        for persona_type, persona_data in results["personalities"].items():
            recovery_patterns[persona_type] = {
                "recovery_times": {},
                "recovery_quality": {}
            }
            
            for trauma_type, trauma_data in persona_data["trauma_tests"].items():
                emotional_states = trauma_data["analysis"].get("emotional_states", [])
                injection_epoch = trauma_data["injection_epoch"]
                
                recovery_time = self._calculate_recovery_time(emotional_states, injection_epoch)
                recovery_patterns[persona_type]["recovery_times"][trauma_type] = recovery_time
        
        return recovery_patterns


async def main():
    """Run the Trauma Response Analysis scenario."""
    print("🚀 Trauma Response Analysis Scenario")
    print("=" * 50)
    
    # Use Docker service name when running in container, localhost otherwise
    api_url = "http://api:8000" if os.getenv("DOCKER_ENV") else "http://localhost:8000"
    scenario = TraumaResponseAnalysis(api_base_url=api_url)
    
    try:
        # Run the scenario
        results = await scenario.run_scenario()
        print("✅ Scenario completed successfully!")
        
        # Analyze results
        analysis = await scenario.analyze_results(results)
        print("✅ Analysis completed!")
        
        # Print key insights
        print("\n📊 Key Insights:")
        print("=" * 30)
        
        for persona_type, insights in analysis["personality_insights"].items():
            print(f"\n{persona_type.replace('_', ' ').title()}:")
            baseline_stability = insights["baseline_stability"].get("overall_stability", 1.0)
            print(f"  Baseline Stability: {baseline_stability:.3f}")
            
            for trauma_type, response in insights["trauma_responses"].items():
                stability_change = response["stability_change"]
                recovery_time = response["recovery_time"]
                print(f"  {trauma_type}: Stability Δ={stability_change:.3f}, Recovery={recovery_time} epochs")
        
        print(f"\n📁 Results saved to: {results['results_file']}")
        print(f"📊 Analysis saved to: {analysis['analysis_file']}")
        
    except Exception as e:
        print(f"❌ Scenario failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 