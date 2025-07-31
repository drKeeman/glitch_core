"""
Statistical analysis tools for AI personality drift research.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
import pingouin as pg

# Use absolute imports instead of relative imports
try:
    from models import Persona, AssessmentResult, PHQ9Result, GAD7Result, PSS10Result
    from models.persona import PersonalityTrait
except ImportError:
    # Fallback for when running from notebooks
    from src.models import Persona, AssessmentResult, PHQ9Result, GAD7Result, PSS10Result
    from src.models.persona import PersonalityTrait

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Statistical analysis tools for personality drift research."""
    
    def __init__(self):
        """Initialize the statistical analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_effect_size(self, group1: np.ndarray, group2: np.ndarray, 
                            method: str = "cohens_d") -> Dict[str, float]:
        """
        Calculate effect size between two groups.
        
        Args:
            group1: First group data
            group2: Second group data  
            method: Effect size method ("cohens_d", "hedges_g", "cliffs_delta")
            
        Returns:
            Dictionary with effect size and interpretation
        """
        if method == "cohens_d":
            effect_size = pg.effectsize.cohens_d(group1, group2)
            interpretation = self._interpret_cohens_d(effect_size)
        elif method == "hedges_g":
            effect_size = pg.effectsize.hedges_g(group1, group2)
            interpretation = self._interpret_cohens_d(effect_size)
        elif method == "cliffs_delta":
            effect_size = pg.effectsize.cliffs_delta(group1, group2)
            interpretation = self._interpret_cliffs_delta(effect_size)
        else:
            raise ValueError(f"Unknown effect size method: {method}")
            
        return {
            "effect_size": effect_size,
            "method": method,
            "interpretation": interpretation
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's delta effect size."""
        if abs(delta) < 0.147:
            return "negligible"
        elif abs(delta) < 0.33:
            return "small"
        elif abs(delta) < 0.474:
            return "medium"
        else:
            return "large"
    
    def test_normality(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Test data normality using multiple methods.
        
        Args:
            data: Data to test
            
        Returns:
            Dictionary with normality test results
        """
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(data)
        
        # Anderson-Darling test
        anderson_result = stats.anderson(data)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        
        return {
            "shapiro_wilk": {
                "statistic": shapiro_stat,
                "p_value": shapiro_p,
                "is_normal": shapiro_p > 0.05
            },
            "anderson_darling": {
                "statistic": anderson_result.statistic,
                "critical_values": anderson_result.critical_values,
                "significance_level": anderson_result.significance_level,
                "is_normal": anderson_result.statistic < anderson_result.critical_values[2]
            },
            "kolmogorov_smirnov": {
                "statistic": ks_stat,
                "p_value": ks_p,
                "is_normal": ks_p > 0.05
            }
        }
    
    def compare_groups(self, group1: np.ndarray, group2: np.ndarray, 
                      test_type: str = "auto") -> Dict[str, Any]:
        """
        Compare two groups using appropriate statistical tests.
        
        Args:
            group1: First group data
            group2: Second group data
            test_type: Test type ("auto", "parametric", "nonparametric")
            
        Returns:
            Dictionary with comparison results
        """
        # Test normality
        normality1 = self.test_normality(group1)
        normality2 = self.test_normality(group2)
        
        # Determine test type
        if test_type == "auto":
            both_normal = (normality1["shapiro_wilk"]["is_normal"] and 
                          normality2["shapiro_wilk"]["is_normal"])
            test_type = "parametric" if both_normal else "nonparametric"
        
        # Perform appropriate test
        if test_type == "parametric":
            # Independent t-test
            t_stat, t_p = ttest_ind(group1, group2)
            test_name = "Independent t-test"
            test_statistic = t_stat
            p_value = t_p
        else:
            # Mann-Whitney U test
            u_stat, u_p = mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
            test_statistic = u_stat
            p_value = u_p
        
        # Calculate effect size
        effect_size = self.calculate_effect_size(group1, group2)
        
        return {
            "test_name": test_name,
            "test_statistic": test_statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "effect_size": effect_size,
            "group1_stats": {
                "n": len(group1),
                "mean": np.mean(group1),
                "std": np.std(group1),
                "median": np.median(group1)
            },
            "group2_stats": {
                "n": len(group2),
                "mean": np.mean(group2),
                "std": np.std(group2),
                "median": np.median(group2)
            },
            "normality_tests": {
                "group1": normality1,
                "group2": normality2
            }
        }
    
    def correlation_analysis(self, x: np.ndarray, y: np.ndarray, 
                           method: str = "pearson") -> Dict[str, Any]:
        """
        Perform correlation analysis between two variables.
        
        Args:
            x: First variable
            y: Second variable
            method: Correlation method ("pearson", "spearman", "kendall")
            
        Returns:
            Dictionary with correlation results
        """
        if method == "pearson":
            corr, p_value = pearsonr(x, y)
        elif method == "spearman":
            corr, p_value = spearmanr(x, y)
        elif method == "kendall":
            corr, p_value = stats.kendalltau(x, y)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Interpret correlation strength
        if abs(corr) < 0.1:
            strength = "negligible"
        elif abs(corr) < 0.3:
            strength = "small"
        elif abs(corr) < 0.5:
            strength = "medium"
        else:
            strength = "large"
        
        return {
            "correlation": corr,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "strength": strength,
            "method": method,
            "n": len(x)
        }
    
    def repeated_measures_analysis(self, data: pd.DataFrame, 
                                 subject_col: str, time_col: str, value_col: str,
                                 test_type: str = "auto") -> Dict[str, Any]:
        """
        Perform repeated measures analysis for longitudinal data.
        
        Args:
            data: DataFrame with repeated measures
            subject_col: Column name for subject IDs
            time_col: Column name for time points
            value_col: Column name for measured values
            test_type: Test type ("auto", "parametric", "nonparametric")
            
        Returns:
            Dictionary with repeated measures analysis results
        """
        # Test for sphericity (Mauchly's test)
        try:
            sphericity_test = pg.sphericity(data=data, dv=value_col, 
                                          within=time_col, subject=subject_col)
            sphericity_assumed = sphericity_test.sphericity
        except:
            sphericity_assumed = False
        
        # Perform appropriate test
        if test_type == "auto":
            test_type = "parametric" if sphericity_assumed else "nonparametric"
        
        if test_type == "parametric":
            # One-way repeated measures ANOVA
            rm_anova = pg.rm_anova(data=data, dv=value_col, 
                                 within=time_col, subject=subject_col)
            
            return {
                "test_name": "One-way repeated measures ANOVA",
                "f_statistic": rm_anova.loc[0, 'F'],
                "p_value": rm_anova.loc[0, 'p-unc'],
                "significant": rm_anova.loc[0, 'p-unc'] < 0.05,
                "effect_size": rm_anova.loc[0, 'np2'],
                "sphericity_assumed": sphericity_assumed,
                "df_between": rm_anova.loc[0, 'ddof1'],
                "df_within": rm_anova.loc[0, 'ddof2']
            }
        else:
            # Friedman test
            friedman_result = pg.friedman(data=data, dv=value_col, 
                                        within=time_col, subject=subject_col)
            
            return {
                "test_name": "Friedman test",
                "chi_square": friedman_result.loc[0, 'Q'],
                "p_value": friedman_result.loc[0, 'p-unc'],
                "significant": friedman_result.loc[0, 'p-unc'] < 0.05,
                "df": friedman_result.loc[0, 'ddof1']
            }
    
    def calculate_reliability(self, data: pd.DataFrame, 
                            items: List[str], method: str = "cronbach") -> Dict[str, Any]:
        """
        Calculate reliability coefficients for assessment scales.
        
        Args:
            data: DataFrame with assessment items
            items: List of item column names
            method: Reliability method ("cronbach", "split_half")
            
        Returns:
            Dictionary with reliability results
        """
        if method == "cronbach":
            alpha = pg.cronbach_alpha(data[items])
            return {
                "method": "Cronbach's Alpha",
                "coefficient": alpha[0],
                "items": len(items),
                "interpretation": self._interpret_reliability(alpha[0])
            }
        elif method == "split_half":
            # Split items into two halves
            n_items = len(items)
            half1 = items[:n_items//2]
            half2 = items[n_items//2:]
            
            # Calculate split-half reliability
            half1_scores = data[half1].sum(axis=1)
            half2_scores = data[half2].sum(axis=1)
            
            corr, p_value = pearsonr(half1_scores, half2_scores)
            
            # Spearman-Brown correction
            corrected_r = (2 * corr) / (1 + corr)
            
            return {
                "method": "Split-half reliability (Spearman-Brown corrected)",
                "raw_correlation": corr,
                "corrected_coefficient": corrected_r,
                "p_value": p_value,
                "items_per_half": len(half1),
                "interpretation": self._interpret_reliability(corrected_r)
            }
        else:
            raise ValueError(f"Unknown reliability method: {method}")
    
    def _interpret_reliability(self, alpha: float) -> str:
        """Interpret reliability coefficient."""
        if alpha < 0.5:
            return "poor"
        elif alpha < 0.7:
            return "questionable"
        elif alpha < 0.8:
            return "acceptable"
        elif alpha < 0.9:
            return "good"
        else:
            return "excellent"
    
    def detect_change_points(self, time_series: np.ndarray, 
                           method: str = "cusum") -> Dict[str, Any]:
        """
        Detect change points in time series data.
        
        Args:
            time_series: Time series data
            method: Change point detection method ("cusum", "pelt")
            
        Returns:
            Dictionary with change point detection results
        """
        if method == "cusum":
            # CUSUM (Cumulative Sum) control chart
            mean = np.mean(time_series)
            std = np.std(time_series)
            
            # Calculate CUSUM statistics
            cusum_pos = np.zeros_like(time_series)
            cusum_neg = np.zeros_like(time_series)
            
            for i in range(1, len(time_series)):
                cusum_pos[i] = max(0, cusum_pos[i-1] + (time_series[i] - mean) / std)
                cusum_neg[i] = max(0, cusum_neg[i-1] + (mean - time_series[i]) / std)
            
            # Detect change points (threshold = 5)
            threshold = 5
            change_points = []
            
            for i in range(len(time_series)):
                if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                    change_points.append(i)
            
            return {
                "method": "CUSUM control chart",
                "change_points": change_points,
                "threshold": threshold,
                "cusum_positive": cusum_pos,
                "cusum_negative": cusum_neg
            }
        
        elif method == "pelt":
            # PELT (Pruned Exact Linear Time) algorithm
            try:
                from ruptures import Pelt
                algo = Pelt(model="l2").fit(time_series)
                change_points = algo.predict(pen=10)
                
                return {
                    "method": "PELT algorithm",
                    "change_points": change_points[:-1],  # Remove last point (end of series)
                    "penalty": 10
                }
            except ImportError:
                self.logger.warning("ruptures library not available, falling back to CUSUM")
                return self.detect_change_points(time_series, method="cusum")
        
        else:
            raise ValueError(f"Unknown change point detection method: {method}")
    
    def calculate_trend_analysis(self, time_series: np.ndarray, 
                               time_points: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate trend analysis for time series data.
        
        Args:
            time_series: Time series data
            time_points: Time points (if None, uses indices)
            
        Returns:
            Dictionary with trend analysis results
        """
        if time_points is None:
            time_points = np.arange(len(time_series))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, time_series)
        
        # Calculate R-squared
        r_squared = r_value ** 2
        
        # Determine trend direction
        if slope > 0:
            direction = "increasing"
        elif slope < 0:
            direction = "decreasing"
        else:
            direction = "stable"
        
        # Calculate rate of change
        rate_of_change = slope
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "direction": direction,
            "rate_of_change": rate_of_change,
            "std_error": std_err,
            "correlation": r_value
        } 