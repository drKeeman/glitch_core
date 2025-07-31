"""
Longitudinal analysis tools for AI personality drift research.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress
import pingouin as pg

# Use absolute imports instead of relative imports
try:
    from models import Persona, AssessmentResult, PHQ9Result, GAD7Result, PSS10Result
    from models.persona import PersonalityTrait
    from analysis.statistical_analyzer import StatisticalAnalyzer
except ImportError:
    # Fallback for when running from notebooks
    from src.models import Persona, AssessmentResult, PHQ9Result, GAD7Result, PSS10Result
    from src.models.persona import PersonalityTrait
    from src.analysis.statistical_analyzer import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class LongitudinalAnalyzer:
    """Longitudinal analysis tools for personality drift research."""
    
    def __init__(self):
        """Initialize the longitudinal analyzer."""
        self.logger = logging.getLogger(__name__)
        self.stats_analyzer = StatisticalAnalyzer()
    
    def analyze_personality_trajectories(self, personas: List[Persona],
                                       traits: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze personality trait trajectories over time.
        
        Args:
            personas: List of personas with state data
            traits: List of traits to analyze (if None, uses all Big Five)
            
        Returns:
            Dictionary with trajectory analysis results
        """
        if traits is None:
            traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        
        trajectory_results = {}
        
        for trait in traits:
            trait_trajectories = []
            
            for persona in personas:
                # Get baseline trait value
                baseline_value = persona.baseline.get_trait(PersonalityTrait(trait))
                
                # Get current trait value (baseline + changes)
                current_value = baseline_value + persona.state.trait_changes.get(trait, 0.0)
                
                trajectory = {
                    "persona_id": persona.state.persona_id,
                    "trait": trait,
                    "baseline_value": baseline_value,
                    "current_value": current_value,
                    "change_magnitude": current_value - baseline_value,
                    "change_percentage": ((current_value - baseline_value) / baseline_value) * 100 if baseline_value > 0 else 0,
                    "simulation_day": persona.state.simulation_day,
                    "drift_magnitude": persona.state.drift_magnitude
                }
                trait_trajectories.append(trajectory)
            
            trajectory_results[trait] = trait_trajectories
        
        return trajectory_results
    
    def analyze_assessment_trajectories(self, assessment_data: pd.DataFrame,
                                      assessment_type: str = "phq9") -> Dict[str, Any]:
        """
        Analyze assessment score trajectories over time.
        
        Args:
            assessment_data: DataFrame with assessment results
            assessment_type: Type of assessment to analyze
            
        Returns:
            Dictionary with trajectory analysis results
        """
        # Filter data for specific assessment type
        type_data = assessment_data[assessment_data['assessment_type'] == assessment_type].copy()
        
        if type_data.empty:
            return {"error": f"No data found for assessment type: {assessment_type}"}
        
        # Group by persona
        trajectory_results = {}
        
        for persona_id in type_data['persona_id'].unique():
            persona_data = type_data[type_data['persona_id'] == persona_id].sort_values('simulation_day')
            
            if len(persona_data) < 2:
                continue
            
            # Calculate trajectory metrics
            trajectory = {
                "persona_id": persona_id,
                "assessment_type": assessment_type,
                "baseline_score": persona_data.iloc[0]['total_score'],
                "final_score": persona_data.iloc[-1]['total_score'],
                "score_change": persona_data.iloc[-1]['total_score'] - persona_data.iloc[0]['total_score'],
                "score_change_percentage": ((persona_data.iloc[-1]['total_score'] - persona_data.iloc[0]['total_score']) / 
                                         persona_data.iloc[0]['total_score']) * 100 if persona_data.iloc[0]['total_score'] > 0 else 0,
                "assessment_count": len(persona_data),
                "simulation_days": persona_data['simulation_day'].max() - persona_data['simulation_day'].min(),
                "score_trajectory": persona_data[['simulation_day', 'total_score']].to_dict('records'),
                "severity_progression": persona_data['severity_level'].tolist()
            }
            
            # Calculate trend analysis
            trend_analysis = self.stats_analyzer.calculate_trend_analysis(
                persona_data['total_score'].values,
                persona_data['simulation_day'].values
            )
            trajectory["trend_analysis"] = trend_analysis
            
            # Calculate change points
            change_points = self.stats_analyzer.detect_change_points(
                persona_data['total_score'].values
            )
            trajectory["change_points"] = change_points
            
            trajectory_results[persona_id] = trajectory
        
        return trajectory_results
    
    def analyze_mechanistic_trajectories(self, mechanistic_data: pd.DataFrame,
                                       analysis_type: str = "attention") -> Dict[str, Any]:
        """
        Analyze mechanistic data trajectories over time.
        
        Args:
            mechanistic_data: DataFrame with mechanistic data
            analysis_type: Type of mechanistic analysis
            
        Returns:
            Dictionary with trajectory analysis results
        """
        # Filter data for specific analysis type
        type_data = mechanistic_data[mechanistic_data['analysis_type'] == analysis_type].copy()
        
        if type_data.empty:
            return {"error": f"No data found for analysis type: {analysis_type}"}
        
        trajectory_results = {}
        
        # Group by persona and layer
        for persona_id in type_data['persona_id'].unique():
            persona_data = type_data[type_data['persona_id'] == persona_id]
            
            persona_trajectory = {
                "persona_id": persona_id,
                "analysis_type": analysis_type,
                "layers": {}
            }
            
            for layer in persona_data['layer'].unique():
                layer_data = persona_data[persona_data['layer'] == layer].sort_values('simulation_day')
                
                if len(layer_data) < 2:
                    continue
                
                # Get value column based on analysis type
                if analysis_type == "attention":
                    value_col = "attention_weight"
                elif analysis_type == "activation":
                    value_col = "activation_value"
                else:
                    value_col = "drift_magnitude"
                
                layer_trajectory = {
                    "layer": layer,
                    "baseline_value": layer_data.iloc[0][value_col],
                    "final_value": layer_data.iloc[-1][value_col],
                    "value_change": layer_data.iloc[-1][value_col] - layer_data.iloc[0][value_col],
                    "value_trajectory": layer_data[['simulation_day', value_col]].to_dict('records')
                }
                
                # Calculate trend analysis
                trend_analysis = self.stats_analyzer.calculate_trend_analysis(
                    layer_data[value_col].values,
                    layer_data['simulation_day'].values
                )
                layer_trajectory["trend_analysis"] = trend_analysis
                
                persona_trajectory["layers"][layer] = layer_trajectory
            
            trajectory_results[persona_id] = persona_trajectory
        
        return trajectory_results
    
    def calculate_growth_rates(self, assessment_data: pd.DataFrame,
                             assessment_type: str = "phq9") -> Dict[str, Any]:
        """
        Calculate growth rates for assessment scores over time.
        
        Args:
            assessment_data: DataFrame with assessment results
            assessment_type: Type of assessment to analyze
            
        Returns:
            Dictionary with growth rate analysis
        """
        # Filter data for specific assessment type
        type_data = assessment_data[assessment_data['assessment_type'] == assessment_type].copy()
        
        growth_rates = {}
        
        for persona_id in type_data['persona_id'].unique():
            persona_data = type_data[type_data['persona_id'] == persona_id].sort_values('simulation_day')
            
            if len(persona_data) < 2:
                continue
            
            # Calculate linear growth rate
            slope, intercept, r_value, p_value, std_err = linregress(
                persona_data['simulation_day'], persona_data['total_score']
            )
            
            # Calculate average daily change
            daily_changes = persona_data['total_score'].diff().dropna()
            avg_daily_change = daily_changes.mean()
            
            # Calculate acceleration (second derivative)
            if len(daily_changes) > 1:
                acceleration = daily_changes.diff().dropna().mean()
            else:
                acceleration = 0.0
            
            growth_rates[persona_id] = {
                "linear_growth_rate": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "avg_daily_change": avg_daily_change,
                "acceleration": acceleration,
                "std_error": std_err,
                "assessment_count": len(persona_data),
                "time_span": persona_data['simulation_day'].max() - persona_data['simulation_day'].min()
            }
        
        return growth_rates
    
    def detect_trajectory_patterns(self, assessment_data: pd.DataFrame,
                                 assessment_type: str = "phq9",
                                 pattern_types: List[str] = None) -> Dict[str, Any]:
        """
        Detect common trajectory patterns in assessment data.
        
        Args:
            assessment_data: DataFrame with assessment results
            assessment_type: Type of assessment to analyze
            pattern_types: Types of patterns to detect
            
        Returns:
            Dictionary with pattern detection results
        """
        if pattern_types is None:
            pattern_types = ["linear_increase", "linear_decrease", "stable", "fluctuating", "accelerating", "decelerating"]
        
        # Filter data for specific assessment type
        type_data = assessment_data[assessment_data['assessment_type'] == assessment_type].copy()
        
        pattern_results = {}
        
        for persona_id in type_data['persona_id'].unique():
            persona_data = type_data[type_data['persona_id'] == persona_id].sort_values('simulation_day')
            
            if len(persona_data) < 3:
                continue
            
            scores = persona_data['total_score'].values
            days = persona_data['simulation_day'].values
            
            patterns = {}
            
            # Linear trend pattern
            slope, _, r_squared, p_value, _ = linregress(days, scores)
            if abs(slope) > 0.01 and p_value < 0.05:
                if slope > 0:
                    patterns["linear_increase"] = {"slope": slope, "r_squared": r_squared, "p_value": p_value}
                else:
                    patterns["linear_decrease"] = {"slope": slope, "r_squared": r_squared, "p_value": p_value}
            elif abs(slope) <= 0.01:
                patterns["stable"] = {"slope": slope, "r_squared": r_squared, "p_value": p_value}
            
            # Fluctuating pattern (high variance)
            variance = np.var(scores)
            mean_score = np.mean(scores)
            coefficient_of_variation = np.sqrt(variance) / mean_score if mean_score > 0 else 0
            
            if coefficient_of_variation > 0.2:  # Threshold for high variability
                patterns["fluctuating"] = {
                    "coefficient_of_variation": coefficient_of_variation,
                    "variance": variance
                }
            
            # Acceleration/deceleration patterns
            if len(scores) >= 4:
                # Calculate second derivative (acceleration)
                daily_changes = np.diff(scores)
                acceleration = np.mean(np.diff(daily_changes))
                
                if abs(acceleration) > 0.01:
                    if acceleration > 0:
                        patterns["accelerating"] = {"acceleration": acceleration}
                    else:
                        patterns["decelerating"] = {"acceleration": acceleration}
            
            pattern_results[persona_id] = patterns
        
        return pattern_results
    
    def analyze_cross_temporal_correlations(self, assessment_data: pd.DataFrame,
                                          mechanistic_data: pd.DataFrame,
                                          time_lags: List[int] = None) -> Dict[str, Any]:
        """
        Analyze correlations between assessment scores and mechanistic data across time lags.
        
        Args:
            assessment_data: DataFrame with assessment results
            mechanistic_data: DataFrame with mechanistic data
            time_lags: List of time lags to analyze (in days)
            
        Returns:
            Dictionary with cross-temporal correlation results
        """
        if time_lags is None:
            time_lags = [0, 1, 3, 7, 14]  # Same day, 1 day, 3 days, 1 week, 2 weeks
        
        correlation_results = {}
        
        # Get unique assessment types
        assessment_types = assessment_data['assessment_type'].unique()
        
        for assessment_type in assessment_types:
            type_assessment_data = assessment_data[assessment_data['assessment_type'] == assessment_type]
            
            for analysis_type in mechanistic_data['analysis_type'].unique():
                type_mechanistic_data = mechanistic_data[mechanistic_data['analysis_type'] == analysis_type]
                
                correlations = {}
                
                for lag in time_lags:
                    lag_correlations = []
                    
                    for persona_id in type_assessment_data['persona_id'].unique():
                        # Get assessment data for this persona
                        persona_assessment = type_assessment_data[
                            type_assessment_data['persona_id'] == persona_id
                        ].sort_values('simulation_day')
                        
                        # Get mechanistic data for this persona
                        persona_mechanistic = type_mechanistic_data[
                            type_mechanistic_data['persona_id'] == persona_id
                        ].sort_values('simulation_day')
                        
                        if len(persona_assessment) < 2 or len(persona_mechanistic) < 2:
                            continue
                        
                        # Align data with time lag
                        aligned_data = []
                        
                        for _, assessment_row in persona_assessment.iterrows():
                            assessment_day = assessment_row['simulation_day']
                            mechanistic_day = assessment_day - lag
                            
                            # Find mechanistic data for the lagged day
                            mechanistic_match = persona_mechanistic[
                                persona_mechanistic['simulation_day'] == mechanistic_day
                            ]
                            
                            if not mechanistic_match.empty:
                                # Get value column based on analysis type
                                if analysis_type == "attention":
                                    value_col = "attention_weight"
                                elif analysis_type == "activation":
                                    value_col = "activation_value"
                                else:
                                    value_col = "drift_magnitude"
                                
                                aligned_data.append({
                                    'assessment_score': assessment_row['total_score'],
                                    'mechanistic_value': mechanistic_match.iloc[0][value_col]
                                })
                        
                        if len(aligned_data) >= 2:
                            assessment_scores = [d['assessment_score'] for d in aligned_data]
                            mechanistic_values = [d['mechanistic_value'] for d in aligned_data]
                            
                            # Calculate correlation
                            correlation = self.stats_analyzer.correlation_analysis(
                                np.array(assessment_scores),
                                np.array(mechanistic_values),
                                method="pearson"
                            )
                            
                            lag_correlations.append({
                                'persona_id': persona_id,
                                'correlation': correlation['correlation'],
                                'p_value': correlation['p_value'],
                                'significant': correlation['significant'],
                                'n': correlation['n']
                            })
                    
                    if lag_correlations:
                        correlations[f"lag_{lag}_days"] = lag_correlations
                
                correlation_results[f"{assessment_type}_{analysis_type}"] = correlations
        
        return correlation_results
    
    def calculate_trajectory_similarity(self, assessment_data: pd.DataFrame,
                                      assessment_type: str = "phq9",
                                      similarity_method: str = "correlation") -> Dict[str, Any]:
        """
        Calculate similarity between assessment trajectories.
        
        Args:
            assessment_data: DataFrame with assessment results
            assessment_type: Type of assessment to analyze
            similarity_method: Method for calculating similarity
            
        Returns:
            Dictionary with trajectory similarity results
        """
        # Filter data for specific assessment type
        type_data = assessment_data[assessment_data['assessment_type'] == assessment_type].copy()
        
        persona_ids = type_data['persona_id'].unique()
        n_personas = len(persona_ids)
        
        if n_personas < 2:
            return {"error": "Need at least 2 personas for similarity analysis"}
        
        # Create similarity matrix
        similarity_matrix = np.zeros((n_personas, n_personas))
        
        for i, persona1 in enumerate(persona_ids):
            for j, persona2 in enumerate(persona_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    continue
                
                # Get trajectories for both personas
                traj1 = type_data[type_data['persona_id'] == persona1].sort_values('simulation_day')
                traj2 = type_data[type_data['persona_id'] == persona2].sort_values('simulation_day')
                
                if len(traj1) < 2 or len(traj2) < 2:
                    similarity_matrix[i, j] = np.nan
                    continue
                
                # Align trajectories to common time points
                common_days = set(traj1['simulation_day']) & set(traj2['simulation_day'])
                
                if len(common_days) < 2:
                    similarity_matrix[i, j] = np.nan
                    continue
                
                # Get aligned scores
                scores1 = traj1[traj1['simulation_day'].isin(common_days)]['total_score'].values
                scores2 = traj2[traj2['simulation_day'].isin(common_days)]['total_score'].values
                
                # Calculate similarity
                if similarity_method == "correlation":
                    correlation = self.stats_analyzer.correlation_analysis(
                        scores1, scores2, method="pearson"
                    )
                    similarity_matrix[i, j] = correlation['correlation']
                elif similarity_method == "euclidean":
                    # Normalize scores for comparison
                    scores1_norm = (scores1 - np.mean(scores1)) / np.std(scores1) if np.std(scores1) > 0 else scores1
                    scores2_norm = (scores2 - np.mean(scores2)) / np.std(scores2) if np.std(scores2) > 0 else scores2
                    similarity_matrix[i, j] = 1 / (1 + np.linalg.norm(scores1_norm - scores2_norm))
                else:
                    raise ValueError(f"Unknown similarity method: {similarity_method}")
        
        return {
            "similarity_matrix": similarity_matrix,
            "persona_ids": persona_ids,
            "similarity_method": similarity_method,
            "assessment_type": assessment_type,
            "mean_similarity": np.nanmean(similarity_matrix),
            "std_similarity": np.nanstd(similarity_matrix)
        } 