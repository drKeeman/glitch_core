"""
Real-time mechanistic visualization for attention patterns and neural activations.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.models.mechanistic import (
    MechanisticAnalysis,
    AttentionCapture,
    ActivationCapture,
    DriftDetection
)
from src.models.persona import Persona


logger = logging.getLogger(__name__)


class MechanisticVisualizer:
    """Real-time visualization for mechanistic analysis."""
    
    def __init__(self):
        """Initialize mechanistic visualizer."""
        self.visualization_history: Dict[str, List[MechanisticAnalysis]] = {}
        self.attention_heatmaps: Dict[str, np.ndarray] = {}
        self.activation_plots: Dict[str, List[float]] = {}
        self.drift_timelines: Dict[str, List[DriftDetection]] = {}
        
        # Visualization configuration
        self.max_history_length = 100
        self.update_frequency = 1.0  # seconds
        self.auto_save_plots = True
        
        # Performance tracking
        self.total_visualizations = 0
        self.total_processing_time = 0.0
    
    async def add_analysis(self, persona_id: str, analysis: MechanisticAnalysis):
        """Add mechanistic analysis to visualization history."""
        try:
            if persona_id not in self.visualization_history:
                self.visualization_history[persona_id] = []
            
            self.visualization_history[persona_id].append(analysis)
            
            # Keep only recent history
            if len(self.visualization_history[persona_id]) > self.max_history_length:
                self.visualization_history[persona_id] = self.visualization_history[persona_id][-self.max_history_length:]
            
            # Update visualizations
            await self._update_attention_heatmap(persona_id, analysis)
            await self._update_activation_plot(persona_id, analysis)
            
            logger.debug(f"Added analysis to visualization for {persona_id}")
            
        except Exception as e:
            logger.error(f"Error adding analysis to visualization: {e}")
    
    async def add_drift_detection(self, persona_id: str, detection: DriftDetection):
        """Add drift detection to visualization timeline."""
        try:
            if persona_id not in self.drift_timelines:
                self.drift_timelines[persona_id] = []
            
            self.drift_timelines[persona_id].append(detection)
            
            # Keep only recent history
            if len(self.drift_timelines[persona_id]) > self.max_history_length:
                self.drift_timelines[persona_id] = self.drift_timelines[persona_id][-self.max_history_length:]
            
            logger.debug(f"Added drift detection to visualization for {persona_id}")
            
        except Exception as e:
            logger.error(f"Error adding drift detection to visualization: {e}")
    
    async def _update_attention_heatmap(self, persona_id: str, analysis: MechanisticAnalysis):
        """Update attention heatmap for a persona."""
        if not analysis.attention_capture:
            return
        
        try:
            attention_weights = analysis.attention_capture.attention_weights
            if not attention_weights:
                return
            
            # Convert to numpy array
            attention_matrix = np.array(attention_weights)
            
            # Store heatmap data
            self.attention_heatmaps[persona_id] = attention_matrix
            
        except Exception as e:
            logger.error(f"Error updating attention heatmap: {e}")
    
    async def _update_activation_plot(self, persona_id: str, analysis: MechanisticAnalysis):
        """Update activation plot for a persona."""
        if not analysis.activation_capture:
            return
        
        try:
            if persona_id not in self.activation_plots:
                self.activation_plots[persona_id] = []
            
            # Extract activation magnitude
            activation_magnitude = analysis.activation_capture.activation_magnitude
            self.activation_plots[persona_id].append(activation_magnitude)
            
            # Keep only recent history
            if len(self.activation_plots[persona_id]) > self.max_history_length:
                self.activation_plots[persona_id] = self.activation_plots[persona_id][-self.max_history_length:]
            
        except Exception as e:
            logger.error(f"Error updating activation plot: {e}")
    
    async def create_attention_heatmap(self, persona_id: str) -> Optional[go.Figure]:
        """Create attention heatmap visualization."""
        if persona_id not in self.attention_heatmaps:
            return None
        
        try:
            attention_matrix = self.attention_heatmaps[persona_id]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=attention_matrix,
                colorscale='Viridis',
                showscale=True,
                zmid=0.5
            ))
            
            fig.update_layout(
                title=f"Attention Heatmap - {persona_id}",
                xaxis_title="Token Position",
                yaxis_title="Token Position",
                width=600,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating attention heatmap: {e}")
            return None
    
    async def create_activation_timeline(self, persona_id: str) -> Optional[go.Figure]:
        """Create activation timeline visualization."""
        if persona_id not in self.activation_plots:
            return None
        
        try:
            activations = self.activation_plots[persona_id]
            timestamps = list(range(len(activations)))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=activations,
                mode='lines+markers',
                name='Activation Magnitude',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f"Activation Timeline - {persona_id}",
                xaxis_title="Time Step",
                yaxis_title="Activation Magnitude",
                width=800,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating activation timeline: {e}")
            return None
    
    async def create_drift_visualization(self, persona_id: str) -> Optional[go.Figure]:
        """Create drift detection visualization."""
        if persona_id not in self.drift_timelines:
            return None
        
        try:
            detections = self.drift_timelines[persona_id]
            if not detections:
                return None
            
            # Extract drift data
            days = [d.current_day for d in detections]
            magnitudes = [d.drift_magnitude for d in detections]
            directions = [d.drift_direction for d in detections]
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Drift Magnitude Over Time', 'Drift Direction'),
                vertical_spacing=0.1
            )
            
            # Drift magnitude plot
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=magnitudes,
                    mode='lines+markers',
                    name='Drift Magnitude',
                    line=dict(color='red', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Add threshold lines
            fig.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                         annotation_text="Detection Threshold", row=1, col=1)
            fig.add_hline(y=0.05, line_dash="dash", line_color="yellow", 
                         annotation_text="Significance Threshold", row=1, col=1)
            
            # Drift direction plot
            direction_colors = {
                'positive': 'green',
                'negative': 'red',
                'neutral': 'gray'
            }
            
            for direction in set(directions):
                direction_days = [d for i, d in enumerate(days) if directions[i] == direction]
                direction_magnitudes = [m for i, m in enumerate(magnitudes) if directions[i] == direction]
                
                fig.add_trace(
                    go.Scatter(
                        x=direction_days,
                        y=direction_magnitudes,
                        mode='markers',
                        name=f'{direction.title()} Drift',
                        marker=dict(color=direction_colors.get(direction, 'blue'), size=10)
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title=f"Personality Drift Analysis - {persona_id}",
                width=800,
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating drift visualization: {e}")
            return None
    
    async def create_comprehensive_dashboard(self, persona_id: str) -> Optional[go.Figure]:
        """Create comprehensive dashboard with all visualizations."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Attention Heatmap',
                    'Activation Timeline', 
                    'Drift Magnitude',
                    'Circuit Activity'
                ),
                specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Attention heatmap
            attention_fig = await self.create_attention_heatmap(persona_id)
            if attention_fig:
                attention_data = attention_fig.data[0]
                fig.add_trace(attention_data, row=1, col=1)
            
            # Activation timeline
            activation_fig = await self.create_activation_timeline(persona_id)
            if activation_fig:
                activation_data = activation_fig.data[0]
                fig.add_trace(activation_data, row=1, col=2)
            
            # Drift visualization
            drift_fig = await self.create_drift_visualization(persona_id)
            if drift_fig:
                for trace in drift_fig.data:
                    fig.add_trace(trace, row=2, col=1)
            
            # Circuit activity (placeholder)
            circuit_data = await self._get_circuit_activity_data(persona_id)
            if circuit_data:
                fig.add_trace(
                    go.Scatter(
                        x=list(circuit_data.keys()),
                        y=list(circuit_data.values()),
                        mode='lines+markers',
                        name='Circuit Activity',
                        line=dict(color='purple', width=2)
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title=f"Mechanistic Analysis Dashboard - {persona_id}",
                width=1200,
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard: {e}")
            return None
    
    async def _get_circuit_activity_data(self, persona_id: str) -> Dict[str, float]:
        """Get circuit activity data for visualization."""
        # This would integrate with CircuitTracker in a full implementation
        # For now, return placeholder data
        return {
            "self_reference": 0.6,
            "emotional": 0.4,
            "memory": 0.7,
            "attention": 0.5,
            "language": 0.8,
            "reasoning": 0.3
        }
    
    async def create_mechanistic_summary(self, persona_id: str) -> Dict[str, Any]:
        """Create mechanistic analysis summary."""
        try:
            summary = {
                "persona_id": persona_id,
                "total_analyses": 0,
                "attention_patterns": {},
                "activation_patterns": {},
                "drift_indicators": {},
                "circuit_activity": {}
            }
            
            # Analysis count
            if persona_id in self.visualization_history:
                summary["total_analyses"] = len(self.visualization_history[persona_id])
            
            # Attention patterns
            if persona_id in self.attention_heatmaps:
                attention_matrix = self.attention_heatmaps[persona_id]
                summary["attention_patterns"] = {
                    "matrix_shape": attention_matrix.shape,
                    "max_attention": float(np.max(attention_matrix)),
                    "mean_attention": float(np.mean(attention_matrix)),
                    "attention_entropy": float(self._calculate_entropy(attention_matrix))
                }
            
            # Activation patterns
            if persona_id in self.activation_plots:
                activations = self.activation_plots[persona_id]
                summary["activation_patterns"] = {
                    "total_measurements": len(activations),
                    "mean_activation": float(np.mean(activations)),
                    "activation_trend": float(self._calculate_trend(activations)),
                    "activation_stability": float(self._calculate_stability(activations))
                }
            
            # Drift indicators
            if persona_id in self.drift_timelines:
                detections = self.drift_timelines[persona_id]
                if detections:
                    latest_detection = detections[-1]
                    summary["drift_indicators"] = {
                        "drift_detected": latest_detection.drift_detected,
                        "significant_drift": latest_detection.significant_drift,
                        "drift_magnitude": latest_detection.drift_magnitude,
                        "drift_direction": latest_detection.drift_direction,
                        "affected_circuits": latest_detection.affected_circuits
                    }
            
            # Circuit activity
            circuit_data = await self._get_circuit_activity_data(persona_id)
            summary["circuit_activity"] = circuit_data
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating mechanistic summary: {e}")
            return {}
    
    def _calculate_entropy(self, matrix: np.ndarray) -> float:
        """Calculate entropy of attention matrix."""
        try:
            # Flatten matrix and normalize
            flat_matrix = matrix.flatten()
            if np.sum(flat_matrix) == 0:
                return 0.0
            
            normalized = flat_matrix / np.sum(flat_matrix)
            entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
            return entropy
        except Exception:
            return 0.0
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in time series."""
        try:
            if len(values) < 2:
                return 0.0
            
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return slope
        except Exception:
            return 0.0
    
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability of time series."""
        try:
            if len(values) < 2:
                return 1.0
            
            variance = np.var(values)
            stability = 1.0 / (1.0 + variance)
            return stability
        except Exception:
            return 1.0
    
    async def export_visualization(self, persona_id: str, format: str = "html") -> Optional[str]:
        """Export visualization to file."""
        try:
            dashboard = await self.create_comprehensive_dashboard(persona_id)
            if not dashboard:
                return None
            
            filename = f"mechanistic_dashboard_{persona_id}_{int(time.time())}.{format}"
            
            if format == "html":
                dashboard.write_html(filename)
            elif format == "png":
                dashboard.write_image(filename)
            elif format == "pdf":
                dashboard.write_image(filename)
            
            logger.info(f"Exported visualization to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting visualization: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_visualizations": self.total_visualizations,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / max(self.total_visualizations, 1),
            "personas_tracked": len(self.visualization_history),
            "max_history_length": self.max_history_length
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.visualization_history.clear()
        self.attention_heatmaps.clear()
        self.activation_plots.clear()
        self.drift_timelines.clear()
        logger.info("Mechanistic visualizer cleaned up") 