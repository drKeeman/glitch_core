"""
Cross-condition analysis tools for AI personality drift research.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, kruskal
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


class CrossConditionAnalyzer:
    """Cross-condition analysis tools for personality drift research."""
    
    def __init__(self):
        """Initialize the cross-condition analyzer."""
        self.logger = logging.getLogger(__name__)
        self.stats_analyzer = StatisticalAnalyzer()
    
    def compare_experimental_conditions(self, assessment_data: pd.DataFrame,
                                     condition_column: str,
                                     assessment_type: str = "phq9",
                                     comparison_metric: str = "final_score") -> Dict[str, Any]:
        """
        Compare assessment results across experimental conditions.
        
        Args:
            assessment_data: DataFrame with assessment results
            condition_column: Column name for experimental conditions
            assessment_type: Type of assessment to analyze
            comparison_metric: Metric to compare ("final_score", "score_change", "growth_rate")
            
        Returns:
            Dictionary with comparison results
        """
        # Filter data for specific assessment type
        type_data = assessment_data[assessment_data['assessment_type'] == assessment_type].copy()
        
        if type_data.empty:
            return {"error": f"No data found for assessment type: {assessment_type}"}
        
        # Calculate comparison metric for each persona
        persona_metrics = {}
        
        for persona_id in type_data['persona_id'].unique():
            persona_data = type_data[type_data['persona_id'] == persona_id].sort_values('simulation_day')
            
            if len(persona_data) < 2:
                continue
            
            condition = persona_data[condition_column].iloc[0]
            
            if comparison_metric == "final_score":
                metric_value = persona_data.iloc[-1]['total_score']
            elif comparison_metric == "score_change":
                metric_value = persona_data.iloc[-1]['total_score'] - persona_data.iloc[0]['total_score']
            elif comparison_metric == "growth_rate":
                # Calculate linear growth rate
                slope, _, _, _, _ = stats.linregress(
                    persona_data['simulation_day'], persona_data['total_score']
                )
                metric_value = slope
            else:
                raise ValueError(f"Unknown comparison metric: {comparison_metric}")
            
            persona_metrics[persona_id] = {
                "condition": condition,
                "metric_value": metric_value,
                "assessment_count": len(persona_data)
            }
        
        # Group by condition
        condition_groups = {}
        for persona_id, metrics in persona_metrics.items():
            condition = metrics["condition"]
            if condition not in condition_groups:
                condition_groups[condition] = []
            condition_groups[condition].append(metrics["metric_value"])
        
        # Perform statistical comparison
        conditions = list(condition_groups.keys())
        if len(conditions) < 2:
            return {"error": "Need at least 2 conditions for comparison"}
        
        # Convert to arrays for statistical testing
        condition_arrays = [np.array(condition_groups[cond]) for cond in conditions]
        
        # Test normality for each condition
        normality_tests = {}
        for i, condition in enumerate(conditions):
            normality_tests[condition] = self.stats_analyzer.test_normality(condition_arrays[i])
        
        # Determine if parametric or non-parametric test is appropriate
        all_normal = all(normality_tests[cond]["shapiro_wilk"]["is_normal"] for cond in conditions)
        
        if all_normal and len(conditions) == 2:
            # Two-sample t-test
            t_stat, t_p = stats.ttest_ind(condition_arrays[0], condition_arrays[1])
            test_name = "Independent t-test"
            test_statistic = t_stat
            p_value = t_p
        elif all_normal and len(conditions) > 2:
            # One-way ANOVA
            f_stat, f_p = f_oneway(*condition_arrays)
            test_name = "One-way ANOVA"
            test_statistic = f_stat
            p_value = f_p
        elif len(conditions) == 2:
            # Mann-Whitney U test
            u_stat, u_p = stats.mannwhitneyu(condition_arrays[0], condition_arrays[1], alternative='two-sided')
            test_name = "Mann-Whitney U test"
            test_statistic = u_stat
            p_value = u_p
        else:
            # Kruskal-Wallis test
            h_stat, h_p = kruskal(*condition_arrays)
            test_name = "Kruskal-Wallis test"
            test_statistic = h_stat
            p_value = h_p
        
        # Calculate effect sizes
        effect_sizes = {}
        if len(conditions) == 2:
            effect_size = self.stats_analyzer.calculate_effect_size(condition_arrays[0], condition_arrays[1])
            effect_sizes["overall"] = effect_size
        else:
            # Calculate pairwise effect sizes
            for i in range(len(conditions)):
                for j in range(i + 1, len(conditions)):
                    pair_name = f"{conditions[i]}_vs_{conditions[j]}"
                    effect_size = self.stats_analyzer.calculate_effect_size(condition_arrays[i], condition_arrays[j])
                    effect_sizes[pair_name] = effect_size
        
        # Calculate descriptive statistics
        descriptive_stats = {}
        for condition in conditions:
            data = condition_arrays[conditions.index(condition)]
            descriptive_stats[condition] = {
                "n": len(data),
                "mean": np.mean(data),
                "std": np.std(data),
                "median": np.median(data),
                "min": np.min(data),
                "max": np.max(data)
            }
        
        return {
            "test_name": test_name,
            "test_statistic": test_statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "conditions": conditions,
            "effect_sizes": effect_sizes,
            "descriptive_stats": descriptive_stats,
            "normality_tests": normality_tests,
            "comparison_metric": comparison_metric,
            "assessment_type": assessment_type
        }
    
    def analyze_condition_by_time_interaction(self, assessment_data: pd.DataFrame,
                                           condition_column: str,
                                           assessment_type: str = "phq9") -> Dict[str, Any]:
        """
        Analyze interaction between experimental conditions and time.
        
        Args:
            assessment_data: DataFrame with assessment results
            condition_column: Column name for experimental conditions
            assessment_type: Type of assessment to analyze
            
        Returns:
            Dictionary with interaction analysis results
        """
        # Filter data for specific assessment type
        type_data = assessment_data[assessment_data['assessment_type'] == assessment_type].copy()
        
        if type_data.empty:
            return {"error": f"No data found for assessment type: {assessment_type}"}
        
        # Prepare data for mixed ANOVA
        conditions = type_data[condition_column].unique()
        
        if len(conditions) < 2:
            return {"error": "Need at least 2 conditions for interaction analysis"}
        
        # Create long-format data for analysis
        analysis_data = []
        
        for persona_id in type_data['persona_id'].unique():
            persona_data = type_data[type_data['persona_id'] == persona_id].sort_values('simulation_day')
            
            if len(persona_data) < 2:
                continue
            
            condition = persona_data[condition_column].iloc[0]
            
            for _, row in persona_data.iterrows():
                analysis_data.append({
                    'persona_id': persona_id,
                    'condition': condition,
                    'simulation_day': row['simulation_day'],
                    'total_score': row['total_score']
                })
        
        analysis_df = pd.DataFrame(analysis_data)
        
        if analysis_df.empty:
            return {"error": "No valid data for interaction analysis"}
        
        # Perform mixed ANOVA
        try:
            # Test for sphericity
            sphericity_test = pg.sphericity(data=analysis_df, dv='total_score', 
                                          within='simulation_day', subject='persona_id')
            sphericity_assumed = sphericity_test.sphericity
        except:
            sphericity_assumed = False
        
        # Perform mixed ANOVA
        if sphericity_assumed:
            mixed_anova = pg.mixed_anova(data=analysis_df, dv='total_score', 
                                       between='condition', within='simulation_day', 
                                       subject='persona_id')
        else:
            # Use non-parametric alternative or report sphericity violation
            mixed_anova = None
        
        # Calculate simple effects for each condition
        simple_effects = {}
        for condition in conditions:
            condition_data = analysis_df[analysis_df['condition'] == condition]
            
            if len(condition_data) >= 2:
                # Repeated measures ANOVA for this condition
                try:
                    rm_anova = pg.rm_anova(data=condition_data, dv='total_score', 
                                         within='simulation_day', subject='persona_id')
                    simple_effects[condition] = {
                        "f_statistic": rm_anova.loc[0, 'F'],
                        "p_value": rm_anova.loc[0, 'p-unc'],
                        "significant": rm_anova.loc[0, 'p-unc'] < 0.05,
                        "effect_size": rm_anova.loc[0, 'np2']
                    }
                except:
                    simple_effects[condition] = {"error": "Insufficient data for analysis"}
        
        return {
            "mixed_anova": mixed_anova.to_dict() if mixed_anova is not None else None,
            "sphericity_assumed": sphericity_assumed,
            "simple_effects": simple_effects,
            "conditions": list(conditions),
            "assessment_type": assessment_type
        }
    
    def compare_personality_traits_by_condition(self, personas: List[Persona],
                                              condition_mapping: Dict[str, str],
                                              traits: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare personality traits across experimental conditions.
        
        Args:
            personas: List of personas
            condition_mapping: Dictionary mapping persona_id to condition
            traits: List of traits to compare (if None, uses all Big Five)
            
        Returns:
            Dictionary with trait comparison results
        """
        if traits is None:
            traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        
        comparison_results = {}
        
        for trait in traits:
            # Group trait values by condition
            condition_groups = {}
            
            for persona in personas:
                if persona.state.persona_id not in condition_mapping:
                    continue
                
                condition = condition_mapping[persona.state.persona_id]
                
                # Calculate current trait value
                baseline_value = persona.baseline.get_trait(PersonalityTrait(trait))
                current_value = baseline_value + persona.state.trait_changes.get(trait, 0.0)
                
                if condition not in condition_groups:
                    condition_groups[condition] = []
                condition_groups[condition].append(current_value)
            
            # Perform statistical comparison
            if len(condition_groups) >= 2:
                conditions = list(condition_groups.keys())
                condition_arrays = [np.array(condition_groups[cond]) for cond in conditions]
                
                # Test normality
                normality_tests = {}
                for i, condition in enumerate(conditions):
                    normality_tests[condition] = self.stats_analyzer.test_normality(condition_arrays[i])
                
                # Determine appropriate test
                all_normal = all(normality_tests[cond]["shapiro_wilk"]["is_normal"] for cond in conditions)
                
                if all_normal and len(conditions) == 2:
                    t_stat, t_p = stats.ttest_ind(condition_arrays[0], condition_arrays[1])
                    test_name = "Independent t-test"
                    test_statistic = t_stat
                    p_value = t_p
                elif all_normal and len(conditions) > 2:
                    f_stat, f_p = f_oneway(*condition_arrays)
                    test_name = "One-way ANOVA"
                    test_statistic = f_stat
                    p_value = f_p
                elif len(conditions) == 2:
                    u_stat, u_p = stats.mannwhitneyu(condition_arrays[0], condition_arrays[1], alternative='two-sided')
                    test_name = "Mann-Whitney U test"
                    test_statistic = u_stat
                    p_value = u_p
                else:
                    h_stat, h_p = kruskal(*condition_arrays)
                    test_name = "Kruskal-Wallis test"
                    test_statistic = h_stat
                    p_value = h_p
                
                # Calculate effect sizes
                effect_sizes = {}
                if len(conditions) == 2:
                    effect_size = self.stats_analyzer.calculate_effect_size(condition_arrays[0], condition_arrays[1])
                    effect_sizes["overall"] = effect_size
                else:
                    for i in range(len(conditions)):
                        for j in range(i + 1, len(conditions)):
                            pair_name = f"{conditions[i]}_vs_{conditions[j]}"
                            effect_size = self.stats_analyzer.calculate_effect_size(condition_arrays[i], condition_arrays[j])
                            effect_sizes[pair_name] = effect_size
                
                # Descriptive statistics
                descriptive_stats = {}
                for condition in conditions:
                    data = condition_arrays[conditions.index(condition)]
                    descriptive_stats[condition] = {
                        "n": len(data),
                        "mean": np.mean(data),
                        "std": np.std(data),
                        "median": np.median(data)
                    }
                
                comparison_results[trait] = {
                    "test_name": test_name,
                    "test_statistic": test_statistic,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "conditions": conditions,
                    "effect_sizes": effect_sizes,
                    "descriptive_stats": descriptive_stats,
                    "normality_tests": normality_tests
                }
            else:
                comparison_results[trait] = {"error": "Need at least 2 conditions for comparison"}
        
        return comparison_results
    
    def analyze_mechanistic_differences_by_condition(self, mechanistic_data: pd.DataFrame,
                                                   condition_mapping: Dict[str, str],
                                                   analysis_type: str = "attention") -> Dict[str, Any]:
        """
        Compare mechanistic data across experimental conditions.
        
        Args:
            mechanistic_data: DataFrame with mechanistic data
            condition_mapping: Dictionary mapping persona_id to condition
            analysis_type: Type of mechanistic analysis
            
        Returns:
            Dictionary with mechanistic comparison results
        """
        # Filter data for specific analysis type
        type_data = mechanistic_data[mechanistic_data['analysis_type'] == analysis_type].copy()
        
        if type_data.empty:
            return {"error": f"No data found for analysis type: {analysis_type}"}
        
        # Add condition information
        type_data['condition'] = type_data['persona_id'].map(condition_mapping)
        type_data = type_data.dropna(subset=['condition'])
        
        if type_data.empty:
            return {"error": "No data with valid condition mapping"}
        
        comparison_results = {}
        
        # Get value column based on analysis type
        if analysis_type == "attention":
            value_col = "attention_weight"
        elif analysis_type == "activation":
            value_col = "activation_value"
        else:
            value_col = "drift_magnitude"
        
        # Compare by layer
        for layer in type_data['layer'].unique():
            layer_data = type_data[type_data['layer'] == layer]
            
            # Group by condition
            condition_groups = {}
            for condition in layer_data['condition'].unique():
                condition_data = layer_data[layer_data['condition'] == condition][value_col]
                if len(condition_data) > 0:
                    condition_groups[condition] = condition_data.values
            
            if len(condition_groups) >= 2:
                conditions = list(condition_groups.keys())
                condition_arrays = [condition_groups[cond] for cond in conditions]
                
                # Perform statistical comparison
                if len(conditions) == 2:
                    # Two-sample comparison
                    comparison = self.stats_analyzer.compare_groups(condition_arrays[0], condition_arrays[1])
                else:
                    # Multiple group comparison
                    all_normal = True
                    for condition in conditions:
                        normality = self.stats_analyzer.test_normality(condition_groups[condition])
                        if not normality["shapiro_wilk"]["is_normal"]:
                            all_normal = False
                            break
                    
                    if all_normal:
                        # One-way ANOVA
                        f_stat, f_p = f_oneway(*condition_arrays)
                        comparison = {
                            "test_name": "One-way ANOVA",
                            "test_statistic": f_stat,
                            "p_value": f_p,
                            "significant": f_p < 0.05
                        }
                    else:
                        # Kruskal-Wallis test
                        h_stat, h_p = kruskal(*condition_arrays)
                        comparison = {
                            "test_name": "Kruskal-Wallis test",
                            "test_statistic": h_stat,
                            "p_value": h_p,
                            "significant": h_p < 0.05
                        }
                
                comparison_results[f"layer_{layer}"] = comparison
        
        return comparison_results
    
    def calculate_condition_effect_sizes(self, assessment_data: pd.DataFrame,
                                       condition_column: str,
                                       assessment_type: str = "phq9") -> Dict[str, Any]:
        """
        Calculate effect sizes for experimental conditions.
        
        Args:
            assessment_data: DataFrame with assessment results
            condition_column: Column name for experimental conditions
            assessment_type: Type of assessment to analyze
            
        Returns:
            Dictionary with effect size results
        """
        # Filter data for specific assessment type
        type_data = assessment_data[assessment_data['assessment_type'] == assessment_type].copy()
        
        if type_data.empty:
            return {"error": f"No data found for assessment type: {assessment_type}"}
        
        # Calculate final scores for each persona
        final_scores = {}
        for persona_id in type_data['persona_id'].unique():
            persona_data = type_data[type_data['persona_id'] == persona_id].sort_values('simulation_day')
            
            if len(persona_data) >= 2:
                condition = persona_data[condition_column].iloc[0]
                final_score = persona_data.iloc[-1]['total_score']
                baseline_score = persona_data.iloc[0]['total_score']
                score_change = final_score - baseline_score
                
                final_scores[persona_id] = {
                    "condition": condition,
                    "final_score": final_score,
                    "baseline_score": baseline_score,
                    "score_change": score_change
                }
        
        # Group by condition
        condition_groups = {}
        for persona_id, scores in final_scores.items():
            condition = scores["condition"]
            if condition not in condition_groups:
                condition_groups[condition] = {"final_scores": [], "score_changes": []}
            condition_groups[condition]["final_scores"].append(scores["final_score"])
            condition_groups[condition]["score_changes"].append(scores["score_change"])
        
        # Calculate effect sizes
        effect_sizes = {}
        conditions = list(condition_groups.keys())
        
        if len(conditions) >= 2:
            for i in range(len(conditions)):
                for j in range(i + 1, len(conditions)):
                    cond1, cond2 = conditions[i], conditions[j]
                    
                    # Effect size for final scores
                    final_effect = self.stats_analyzer.calculate_effect_size(
                        np.array(condition_groups[cond1]["final_scores"]),
                        np.array(condition_groups[cond2]["final_scores"])
                    )
                    
                    # Effect size for score changes
                    change_effect = self.stats_analyzer.calculate_effect_size(
                        np.array(condition_groups[cond1]["score_changes"]),
                        np.array(condition_groups[cond2]["score_changes"])
                    )
                    
                    pair_name = f"{cond1}_vs_{cond2}"
                    effect_sizes[pair_name] = {
                        "final_scores": final_effect,
                        "score_changes": change_effect
                    }
        
        return {
            "effect_sizes": effect_sizes,
            "condition_groups": condition_groups,
            "assessment_type": assessment_type
        } 