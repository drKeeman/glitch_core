#!/usr/bin/env python3
"""
Run All Demo Scenarios

This script runs all available demo scenarios and generates a comprehensive report
comparing their results and insights.
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

from .trauma_response_analysis import TraumaResponseAnalysis
from glitch_core.config.logging import get_logger


class ScenarioRunner:
    """Master runner for all demo scenarios."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.logger = get_logger(__name__)
        self.results_dir = Path("scenarios/results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Define all scenarios
        self.scenarios = {
            "trauma_response_analysis": TraumaResponseAnalysis(api_base_url)
        }
    
    async def run_all_scenarios(self) -> Dict[str, Any]:
        """Run all scenarios and collect results."""
        self.logger.info("Starting all demo scenarios...")
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "scenarios": {},
            "summary": {},
            "research_insights": {}
        }
        
        # Run each scenario
        for scenario_name, scenario in self.scenarios.items():
            self.logger.info(f"Running scenario: {scenario_name}")
            
            try:
                # Run the scenario
                results = await scenario.run_scenario()
                
                # Analyze results
                analysis = await scenario.analyze_results(results)
                
                all_results["scenarios"][scenario_name] = {
                    "results": results,
                    "analysis": analysis
                }
                
                self.logger.info(f"✅ Completed scenario: {scenario_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Scenario {scenario_name} failed: {e}")
                all_results["scenarios"][scenario_name] = {
                    "error": str(e)
                }
        
        # Generate summary
        all_results["summary"] = self._generate_summary(all_results["scenarios"])
        
        # Generate research insights
        all_results["research_insights"] = self._generate_research_insights(all_results["scenarios"])
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"all_scenarios_{timestamp}.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        all_results["results_file"] = str(results_file)
        
        return all_results
    
    def _generate_summary(self, scenarios: Dict) -> Dict[str, Any]:
        """Generate a summary of all scenarios."""
        summary = {
            "total_scenarios": len(scenarios),
            "completed_scenarios": 0,
            "failed_scenarios": 0,
            "scenario_status": {}
        }
        
        for scenario_name, scenario_data in scenarios.items():
            if "error" in scenario_data:
                summary["failed_scenarios"] += 1
                summary["scenario_status"][scenario_name] = "failed"
            else:
                summary["completed_scenarios"] += 1
                summary["scenario_status"][scenario_name] = "completed"
        
        return summary
    
    def _generate_research_insights(self, scenarios: Dict) -> Dict[str, Any]:
        """Generate research insights from all scenarios."""
        insights = {
            "temporal_patterns": {},
            "stability_insights": {},
            "intervention_effects": {},
            "personality_comparisons": {}
        }
        
        # Analyze trauma response scenario if available
        if "trauma_response_analysis" in scenarios:
            trauma_data = scenarios["trauma_response_analysis"]
            if "analysis" in trauma_data:
                analysis = trauma_data["analysis"]
                
                # Extract key insights
                if "personality_insights" in analysis:
                    insights["personality_comparisons"] = self._extract_personality_insights(
                        analysis["personality_insights"]
                    )
                
                if "trauma_insights" in analysis:
                    insights["intervention_effects"] = analysis["trauma_insights"]
                
                if "stability_analysis" in analysis:
                    insights["stability_insights"] = analysis["stability_analysis"]
        
        return insights
    
    def _extract_personality_insights(self, personality_insights: Dict) -> Dict[str, Any]:
        """Extract key personality comparison insights."""
        insights = {
            "most_resilient": None,
            "least_resilient": None,
            "recovery_patterns": {},
            "trauma_sensitivity": {}
        }
        
        # Find most and least resilient personalities
        baseline_stabilities = {}
        for persona_type, data in personality_insights.items():
            baseline_stability = data.get("baseline_stability", {}).get("overall_stability", 1.0)
            baseline_stabilities[persona_type] = baseline_stability
        
        if baseline_stabilities:
            sorted_stabilities = sorted(baseline_stabilities.items(), key=lambda x: x[1], reverse=True)
            insights["most_resilient"] = sorted_stabilities[0][0]
            insights["least_resilient"] = sorted_stabilities[-1][0]
        
        # Analyze recovery patterns
        for persona_type, data in personality_insights.items():
            if "trauma_responses" in data:
                recovery_times = []
                for trauma_type, response in data["trauma_responses"].items():
                    recovery_time = response.get("recovery_time", -1)
                    if recovery_time >= 0:
                        recovery_times.append((trauma_type, recovery_time))
                
                insights["recovery_patterns"][persona_type] = recovery_times
        
        return insights
    
    def generate_research_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive research report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"research_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Glitch Core Research Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = results["summary"]
            f.write(f"- **Total Scenarios:** {summary['total_scenarios']}\n")
            f.write(f"- **Completed:** {summary['completed_scenarios']}\n")
            f.write(f"- **Failed:** {summary['failed_scenarios']}\n\n")
            
            # Scenario Results
            f.write("## Scenario Results\n\n")
            for scenario_name, scenario_data in results["scenarios"].items():
                status = summary["scenario_status"].get(scenario_name, "unknown")
                f.write(f"### {scenario_name.replace('_', ' ').title()}\n")
                f.write(f"**Status:** {status}\n\n")
                
                if "error" in scenario_data:
                    f.write(f"**Error:** {scenario_data['error']}\n\n")
                else:
                    f.write("**Results:** Available in JSON format\n\n")
            
            # Research Insights
            f.write("## Research Insights\n\n")
            insights = results["research_insights"]
            
            if "personality_comparisons" in insights:
                personality_insights = insights["personality_comparisons"]
                f.write("### Personality Comparisons\n\n")
                
                if personality_insights.get("most_resilient"):
                    f.write(f"- **Most Resilient:** {personality_insights['most_resilient']}\n")
                if personality_insights.get("least_resilient"):
                    f.write(f"- **Least Resilient:** {personality_insights['least_resilient']}\n")
                
                f.write("\n### Recovery Patterns\n\n")
                for persona_type, recovery_times in personality_insights.get("recovery_patterns", {}).items():
                    f.write(f"- **{persona_type.replace('_', ' ').title()}:**\n")
                    for trauma_type, recovery_time in recovery_times:
                        f.write(f"  - {trauma_type}: {recovery_time} epochs\n")
                    f.write("\n")
            
            # Technical Details
            f.write("## Technical Details\n\n")
            f.write(f"- **Results File:** {results.get('results_file', 'N/A')}\n")
            f.write(f"- **API Base URL:** {self.api_base_url}\n")
            f.write(f"- **Timestamp:** {results['timestamp']}\n\n")
            
            # Next Steps
            f.write("## Next Steps\n\n")
            f.write("1. Review generated visualizations in the results directory\n")
            f.write("2. Analyze temporal patterns in the JSON results\n")
            f.write("3. Compare intervention effects across personality types\n")
            f.write("4. Identify stability boundaries and breakdown points\n")
            f.write("5. Document research findings for publication\n\n")
        
        self.logger.info(f"Research report saved to: {report_file}")
        return str(report_file)


async def main():
    """Run all scenarios and generate research report."""
    print("🚀 Glitch Core - All Scenarios Runner")
    print("=" * 50)
    
    # Use Docker service name when running in container, localhost otherwise
    api_url = "http://api:8000" if os.getenv("DOCKER_ENV") else "http://localhost:8000"
    runner = ScenarioRunner(api_base_url=api_url)
    
    try:
        # Run all scenarios
        results = await runner.run_all_scenarios()
        print("✅ All scenarios completed!")
        
        # Generate research report
        report_file = runner.generate_research_report(results)
        print(f"📊 Research report generated: {report_file}")
        
        # Print summary
        summary = results["summary"]
        print(f"\n📋 Summary:")
        print(f"  Total scenarios: {summary['total_scenarios']}")
        print(f"  Completed: {summary['completed_scenarios']}")
        print(f"  Failed: {summary['failed_scenarios']}")
        
        # Print key insights
        if "research_insights" in results:
            insights = results["research_insights"]
            if "personality_comparisons" in insights:
                personality_insights = insights["personality_comparisons"]
                print(f"\n🔍 Key Insights:")
                if personality_insights.get("most_resilient"):
                    print(f"  Most resilient: {personality_insights['most_resilient']}")
                if personality_insights.get("least_resilient"):
                    print(f"  Least resilient: {personality_insights['least_resilient']}")
        
        print(f"\n📁 All results saved to: {results['results_file']}")
        
    except Exception as e:
        print(f"❌ Scenario runner failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 