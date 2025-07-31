"""
Visualization toolkit for AI personality drift research.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt

# Use absolute imports instead of relative imports
try:
    from models import Persona, AssessmentResult, PHQ9Result, GAD7Result, PSS10Result
    from models.persona import PersonalityTrait
except ImportError:
    # Fallback for when running from notebooks
    from src.models import Persona, AssessmentResult, PHQ9Result, GAD7Result, PSS10Result
    from src.models.persona import PersonalityTrait

logger = logging.getLogger(__name__)


class VisualizationToolkit:
    """Visualization toolkit for personality drift research."""
    
    def __init__(self, style: str = "publication"):
        """
        Initialize the visualization toolkit.
        
        Args:
            style: Plot style ("publication", "default", "dark")
        """
        self.logger = logging.getLogger(__name__)
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Setup plotting style."""
        if self.style == "publication":
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9
            plt.rcParams['legend.fontsize'] = 9
            plt.rcParams['figure.titlesize'] = 14
        elif self.style == "dark":
            plt.style.use('dark_background')
            sns.set_palette("husl")
    
    def plot_personality_traits(self, personas: List[Persona], 
                               traits: Optional[List[str]] = None,
                               plot_type: str = "radar") -> plt.Figure:
        """
        Plot personality traits for multiple personas.
        
        Args:
            personas: List of personas to plot
            traits: List of traits to include (if None, uses all Big Five)
            plot_type: Plot type ("radar", "bar", "heatmap")
            
        Returns:
            Matplotlib figure
        """
        if traits is None:
            traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        
        if plot_type == "radar":
            return self._plot_radar_traits(personas, traits)
        elif plot_type == "bar":
            return self._plot_bar_traits(personas, traits)
        elif plot_type == "heatmap":
            return self._plot_heatmap_traits(personas, traits)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _plot_radar_traits(self, personas: List[Persona], traits: List[str]) -> plt.Figure:
        """Create radar plot of personality traits."""
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each persona
        colors = plt.cm.Set3(np.linspace(0, 1, len(personas)))
        
        for i, persona in enumerate(personas):
            values = [persona.baseline.get_trait(PersonalityTrait(trait)) for trait in traits]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=persona.baseline.name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([trait.title() for trait in traits])
        ax.set_ylim(0, 1)
        ax.set_title("Personality Traits Comparison", size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        return fig
    
    def _plot_bar_traits(self, personas: List[Persona], traits: List[str]) -> plt.Figure:
        """Create bar plot of personality traits."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        names = [p.baseline.name for p in personas]
        x = np.arange(len(traits))
        width = 0.8 / len(personas)
        
        # Plot each persona
        for i, persona in enumerate(personas):
            values = [persona.baseline.get_trait(PersonalityTrait(trait)) for trait in traits]
            ax.bar(x + i * width, values, width, label=persona.baseline.name, alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('Personality Traits')
        ax.set_ylabel('Trait Value')
        ax.set_title('Personality Traits by Persona')
        ax.set_xticks(x + width * (len(personas) - 1) / 2)
        ax.set_xticklabels([trait.title() for trait in traits])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_heatmap_traits(self, personas: List[Persona], traits: List[str]) -> plt.Figure:
        """Create heatmap of personality traits."""
        # Prepare data
        data = []
        names = []
        
        for persona in personas:
            values = [persona.baseline.get_trait(PersonalityTrait(trait)) for trait in traits]
            data.append(values)
            names.append(persona.baseline.name)
        
        df = pd.DataFrame(data, index=names, columns=[trait.title() for trait in traits])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df, annot=True, cmap='RdYlBu_r', center=0.5, 
                   cbar_kws={'label': 'Trait Value'}, ax=ax)
        ax.set_title('Personality Traits Heatmap')
        
        return fig
    
    def plot_assessment_trends(self, assessment_data: pd.DataFrame,
                              assessment_type: str = "phq9",
                              plot_type: str = "line") -> plt.Figure:
        """
        Plot assessment score trends over time.
        
        Args:
            assessment_data: DataFrame with assessment results
            assessment_type: Type of assessment ("phq9", "gad7", "pss10")
            plot_type: Plot type ("line", "box", "violin")
            
        Returns:
            Matplotlib figure
        """
        if plot_type == "line":
            return self._plot_line_trends(assessment_data, assessment_type)
        elif plot_type == "box":
            return self._plot_box_trends(assessment_data, assessment_type)
        elif plot_type == "violin":
            return self._plot_violin_trends(assessment_data, assessment_type)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _plot_line_trends(self, assessment_data: pd.DataFrame, assessment_type: str) -> plt.Figure:
        """Create line plot of assessment trends."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by persona and time
        for persona_id in assessment_data['persona_id'].unique():
            persona_data = assessment_data[assessment_data['persona_id'] == persona_id]
            ax.plot(persona_data['simulation_day'], persona_data['total_score'], 
                   marker='o', label=persona_id, linewidth=2, markersize=4)
        
        ax.set_xlabel('Simulation Day')
        ax.set_ylabel(f'{assessment_type.upper()} Score')
        ax.set_title(f'{assessment_type.upper()} Score Trends Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_box_trends(self, assessment_data: pd.DataFrame, assessment_type: str) -> plt.Figure:
        """Create box plot of assessment trends."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create box plot
        assessment_data.boxplot(column='total_score', by='simulation_day', ax=ax)
        ax.set_xlabel('Simulation Day')
        ax.set_ylabel(f'{assessment_type.upper()} Score')
        ax.set_title(f'{assessment_type.upper()} Score Distribution Over Time')
        
        return fig
    
    def _plot_violin_trends(self, assessment_data: pd.DataFrame, assessment_type: str) -> plt.Figure:
        """Create violin plot of assessment trends."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create violin plot
        sns.violinplot(data=assessment_data, x='simulation_day', y='total_score', ax=ax)
        ax.set_xlabel('Simulation Day')
        ax.set_ylabel(f'{assessment_type.upper()} Score')
        ax.set_title(f'{assessment_type.upper()} Score Distribution Over Time')
        
        return fig
    
    def plot_mechanistic_analysis(self, mechanistic_data: pd.DataFrame,
                                 analysis_type: str = "attention",
                                 plot_type: str = "heatmap") -> plt.Figure:
        """
        Plot mechanistic analysis results.
        
        Args:
            mechanistic_data: DataFrame with mechanistic data
            analysis_type: Type of analysis ("attention", "activation", "drift")
            plot_type: Plot type ("heatmap", "line", "scatter")
            
        Returns:
            Matplotlib figure
        """
        if plot_type == "heatmap":
            return self._plot_mechanistic_heatmap(mechanistic_data, analysis_type)
        elif plot_type == "line":
            return self._plot_mechanistic_line(mechanistic_data, analysis_type)
        elif plot_type == "scatter":
            return self._plot_mechanistic_scatter(mechanistic_data, analysis_type)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _plot_mechanistic_heatmap(self, mechanistic_data: pd.DataFrame, analysis_type: str) -> plt.Figure:
        """Create heatmap of mechanistic data."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        if analysis_type == "attention":
            pivot_data = mechanistic_data.pivot_table(
                values='attention_weight', 
                index='layer', 
                columns='simulation_day', 
                aggfunc='mean'
            )
        elif analysis_type == "activation":
            pivot_data = mechanistic_data.pivot_table(
                values='activation_value', 
                index='layer', 
                columns='simulation_day', 
                aggfunc='mean'
            )
        else:
            pivot_data = mechanistic_data.pivot_table(
                values='drift_magnitude', 
                index='persona_id', 
                columns='simulation_day', 
                aggfunc='mean'
            )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=False, cmap='viridis', ax=ax)
        ax.set_title(f'{analysis_type.title()} Analysis Heatmap')
        
        return fig
    
    def _plot_mechanistic_line(self, mechanistic_data: pd.DataFrame, analysis_type: str) -> plt.Figure:
        """Create line plot of mechanistic data."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if analysis_type == "attention":
            for layer in mechanistic_data['layer'].unique():
                layer_data = mechanistic_data[mechanistic_data['layer'] == layer]
                ax.plot(layer_data['simulation_day'], layer_data['attention_weight'], 
                       label=f'Layer {layer}', marker='o')
        elif analysis_type == "activation":
            for layer in mechanistic_data['layer'].unique():
                layer_data = mechanistic_data[mechanistic_data['layer'] == layer]
                ax.plot(layer_data['simulation_day'], layer_data['activation_value'], 
                       label=f'Layer {layer}', marker='o')
        else:
            for persona_id in mechanistic_data['persona_id'].unique():
                persona_data = mechanistic_data[mechanistic_data['persona_id'] == persona_id]
                ax.plot(persona_data['simulation_day'], persona_data['drift_magnitude'], 
                       label=persona_id, marker='o')
        
        ax.set_xlabel('Simulation Day')
        ax.set_ylabel(f'{analysis_type.title()} Value')
        ax.set_title(f'{analysis_type.title()} Analysis Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_mechanistic_scatter(self, mechanistic_data: pd.DataFrame, analysis_type: str) -> plt.Figure:
        """Create scatter plot of mechanistic data."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if analysis_type == "attention":
            scatter = ax.scatter(mechanistic_data['simulation_day'], 
                               mechanistic_data['attention_weight'],
                               c=mechanistic_data['layer'], cmap='viridis', alpha=0.6)
        elif analysis_type == "activation":
            scatter = ax.scatter(mechanistic_data['simulation_day'], 
                               mechanistic_data['activation_value'],
                               c=mechanistic_data['layer'], cmap='viridis', alpha=0.6)
        else:
            scatter = ax.scatter(mechanistic_data['simulation_day'], 
                               mechanistic_data['drift_magnitude'],
                               c=mechanistic_data['persona_id'], cmap='Set1', alpha=0.6)
        
        ax.set_xlabel('Simulation Day')
        ax.set_ylabel(f'{analysis_type.title()} Value')
        ax.set_title(f'{analysis_type.title()} Analysis Scatter Plot')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax)
        
        return fig
    
    def create_interactive_dashboard(self, assessment_data: pd.DataFrame,
                                   mechanistic_data: pd.DataFrame,
                                   persona_data: pd.DataFrame) -> go.Figure:
        """
        Create interactive dashboard with multiple plots.
        
        Args:
            assessment_data: Assessment results data
            mechanistic_data: Mechanistic analysis data
            persona_data: Persona information data (can be empty)
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Assessment Trends', 'Personality Drift', 
                          'Mechanistic Analysis', 'Trait Changes'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Assessment trends
        for assessment_type in ['phq9', 'gad7', 'pss10']:
            type_data = assessment_data[assessment_data['assessment_type'] == assessment_type]
            if not type_data.empty:
                fig.add_trace(
                    go.Scatter(x=type_data['simulation_day'], y=type_data['total_score'],
                              mode='lines+markers', name=f'{assessment_type.upper()}',
                              showlegend=True),
                    row=1, col=1
                )
        
        # Personality drift
        for persona_id in mechanistic_data['persona_id'].unique():
            persona_drift = mechanistic_data[mechanistic_data['persona_id'] == persona_id]
            if not persona_drift.empty:
                fig.add_trace(
                    go.Scatter(x=persona_drift['simulation_day'], y=persona_drift['drift_magnitude'],
                              mode='lines+markers', name=f'Drift {persona_id}',
                              showlegend=True),
                    row=1, col=2
                )
        
        # Mechanistic analysis
        attention_data = mechanistic_data[mechanistic_data['analysis_type'] == 'attention']
        if not attention_data.empty:
            try:
                pivot_data = attention_data.pivot_table(values='attention_weight', 
                                                      index='layer', columns='simulation_day')
                fig.add_trace(
                    go.Heatmap(z=pivot_data.values,
                              x=pivot_data.columns,
                              y=pivot_data.index,
                              colorscale='Viridis', name='Attention Weights'),
                    row=2, col=1
                )
            except Exception as e:
                self.logger.warning(f"Could not create attention heatmap: {e}")
        
        # Trait changes (only if persona_data has the expected structure)
        if not persona_data.empty and 'trait' in persona_data.columns and 'trait_value' in persona_data.columns:
            for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                trait_data = persona_data[persona_data['trait'] == trait]
                if not trait_data.empty:
                    fig.add_trace(
                        go.Scatter(x=trait_data['simulation_day'], y=trait_data['trait_value'],
                                  mode='lines+markers', name=trait.title(),
                                  showlegend=True),
                        row=2, col=2
                    )
        else:
            # Alternative: show mechanistic drift over time
            for persona_id in mechanistic_data['persona_id'].unique():
                persona_mechanistic = mechanistic_data[mechanistic_data['persona_id'] == persona_id]
                if not persona_mechanistic.empty:
                    fig.add_trace(
                        go.Scatter(x=persona_mechanistic['simulation_day'], 
                                  y=persona_mechanistic['drift_magnitude'],
                                  mode='lines+markers', name=f'Mechanistic {persona_id}',
                                  showlegend=True),
                        row=2, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title_text="AI Personality Drift Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                              variables: Optional[List[str]] = None,
                              method: str = "pearson") -> plt.Figure:
        """
        Create correlation matrix plot.
        
        Args:
            data: DataFrame with variables
            variables: List of variables to include (if None, uses all numeric)
            method: Correlation method ("pearson", "spearman", "kendall")
            
        Returns:
            Matplotlib figure
        """
        if variables is None:
            variables = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = data[variables].corr(method=method)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title(f'Correlation Matrix ({method.title()})')
        
        return fig
    
    def plot_distribution_comparison(self, data: pd.DataFrame,
                                   variable: str,
                                   group_by: str,
                                   plot_type: str = "histogram") -> plt.Figure:
        """
        Compare distributions across groups.
        
        Args:
            data: DataFrame with data
            variable: Variable to plot
            group_by: Grouping variable
            plot_type: Plot type ("histogram", "box", "violin", "kde")
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "histogram":
            for group in data[group_by].unique():
                group_data = data[data[group_by] == group][variable]
                ax.hist(group_data, alpha=0.7, label=group, bins=20)
        elif plot_type == "box":
            data.boxplot(column=variable, by=group_by, ax=ax)
        elif plot_type == "violin":
            sns.violinplot(data=data, x=group_by, y=variable, ax=ax)
        elif plot_type == "kde":
            for group in data[group_by].unique():
                group_data = data[data[group_by] == group][variable]
                group_data.plot.kde(ax=ax, label=group)
        
        ax.set_title(f'Distribution of {variable} by {group_by}')
        if plot_type != "box":
            ax.legend()
        
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, 
                   dpi: int = 300, format: str = "png") -> None:
        """
        Save figure with publication-quality settings.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution
            format: Output format
        """
        fig.savefig(filename, dpi=dpi, format=format, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
    
    def create_altair_chart(self, data: pd.DataFrame,
                           x: str, y: str,
                           color: Optional[str] = None,
                           chart_type: str = "line") -> alt.Chart:
        """
        Create Altair chart for interactive visualization.
        
        Args:
            data: DataFrame with data
            x: X-axis variable
            y: Y-axis variable
            color: Color variable
            chart_type: Chart type ("line", "scatter", "bar", "area")
            
        Returns:
            Altair chart
        """
        if chart_type == "line":
            chart = alt.Chart(data).mark_line().encode(
                x=x, y=y, color=color if color else alt.value('steelblue')
            )
        elif chart_type == "scatter":
            chart = alt.Chart(data).mark_circle().encode(
                x=x, y=y, color=color if color else alt.value('steelblue')
            )
        elif chart_type == "bar":
            chart = alt.Chart(data).mark_bar().encode(
                x=x, y=y, color=color if color else alt.value('steelblue')
            )
        elif chart_type == "area":
            chart = alt.Chart(data).mark_area().encode(
                x=x, y=y, color=color if color else alt.value('steelblue')
            )
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        return chart 