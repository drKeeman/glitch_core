"""
Base Scenario Class

Provides common functionality for all demo scenarios including API integration,
data collection, analysis, and visualization.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import httpx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from glitch_core.config.logging import get_logger


class BaseScenario:
    """Base class for all demo scenarios."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.logger = get_logger(__name__)
        self.results_dir = Path("scenarios/results")
        self.results_dir.mkdir(exist_ok=True)
        
    async def create_persona(self, persona_type: str) -> str:
        """Create a persona via API and return the ID."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_base_url}/api/v1/personas/",
                json={
                    "name": f"test_{persona_type}",
                    "persona_type": persona_type
                }
            )
            response.raise_for_status()
            return response.json()["id"]
    
    async def start_experiment(self, persona_id: str, epochs: int = 100, events_per_epoch: int = 10) -> str:
        """Start an experiment via API and return the experiment ID."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_base_url}/api/v1/experiments/",
                json={
                    "persona_id": persona_id,
                    "drift_profile": "baseline",  # Default drift profile
                    "epochs": epochs,
                    "events_per_epoch": events_per_epoch
                }
            )
            response.raise_for_status()
            return response.json()["id"]
    
    async def inject_intervention(self, experiment_id: str, event: str, emotional_impact: Dict[str, float]) -> Dict[str, Any]:
        """Inject an intervention into a running experiment."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_base_url}/api/v1/interventions/",
                json={
                    "experiment_id": experiment_id,
                    "event_type": event,
                    "intensity": max(emotional_impact.values()) if emotional_impact else 1.0,
                    "description": f"Intervention: {event}"
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def get_analysis(self, experiment_id: str) -> Dict[str, Any]:
        """Get analysis results for an experiment."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_base_url}/api/v1/analysis/{experiment_id}"
            )
            response.raise_for_status()
            return response.json()
    
    async def wait_for_completion(self, experiment_id: str, timeout: int = 300) -> bool:
        """Wait for experiment completion with timeout."""
        start_time = datetime.now()
        async with httpx.AsyncClient() as client:
            while (datetime.now() - start_time).seconds < timeout:
                try:
                    response = await client.get(
                        f"{self.api_base_url}/api/v1/experiments/{experiment_id}"
                    )
                    response.raise_for_status()
                    status = response.json()["status"]
                    if status == "completed":
                        return True
                    elif status == "failed":
                        return False
                    await asyncio.sleep(2)
                except Exception as e:
                    self.logger.warning(f"Error checking experiment status: {e}")
                    await asyncio.sleep(5)
        return False
    
    def save_results(self, scenario_name: str, data: Dict[str, Any]) -> str:
        """Save scenario results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scenario_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
        return str(filepath)
    
    def create_emotional_evolution_plot(self, emotional_states: List[Dict[str, float]], 
                                      title: str = "Emotional Evolution") -> go.Figure:
        """Create an interactive plot of emotional evolution."""
        df = pd.DataFrame(emotional_states)
        
        fig = go.Figure()
        
        for emotion in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[emotion],
                mode='lines+markers',
                name=emotion.title(),
                line=dict(width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Emotional Intensity",
            hovermode='x unified',
            template="plotly_white"
        )
        
        return fig
    
    def create_stability_analysis_plot(self, stability_scores: List[float], 
                                     breakdown_points: List[int],
                                     title: str = "Stability Analysis") -> go.Figure:
        """Create a plot showing stability scores and breakdown points."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(stability_scores))),
            y=stability_scores,
            mode='lines+markers',
            name='Stability Score',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        if breakdown_points:
            fig.add_trace(go.Scatter(
                x=breakdown_points,
                y=[stability_scores[i] for i in breakdown_points],
                mode='markers',
                name='Breakdown Points',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Stability Score",
            template="plotly_white"
        )
        
        return fig
    
    def create_comparison_plot(self, before_data: List[Dict[str, float]], 
                              after_data: List[Dict[str, float]],
                              title: str = "Before vs After Comparison") -> go.Figure:
        """Create a comparison plot of before/after emotional states."""
        before_df = pd.DataFrame(before_data)
        after_df = pd.DataFrame(after_data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Before Intervention", "After Intervention")
        )
        
        for emotion in before_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=before_df.index,
                    y=before_df[emotion],
                    mode='lines+markers',
                    name=f"{emotion.title()} (Before)",
                    line=dict(width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=after_df.index,
                    y=after_df[emotion],
                    mode='lines+markers',
                    name=f"{emotion.title()} (After)",
                    line=dict(width=2)
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=True
        )
        
        return fig
    
    def save_plot(self, fig: go.Figure, filename: str) -> str:
        """Save a plotly figure to HTML file."""
        filepath = self.results_dir / filename
        fig.write_html(str(filepath))
        self.logger.info(f"Plot saved to {filepath}")
        return str(filepath)
    
    async def run_scenario(self) -> Dict[str, Any]:
        """Override this method in subclasses to implement specific scenarios."""
        raise NotImplementedError("Subclasses must implement run_scenario")
    
    async def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method in subclasses to implement specific analysis."""
        raise NotImplementedError("Subclasses must implement analyze_results") 