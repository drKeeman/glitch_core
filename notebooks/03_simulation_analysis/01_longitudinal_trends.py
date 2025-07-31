# %% markdown
# AI Personality Drift Simulation - Longitudinal Trends Analysis
# 
# This notebook analyzes longitudinal trends in simulation data, including:
# - Assessment score trajectories over time
# - Personality trait drift patterns
# - Mechanistic analysis trends
# - Change point detection
# - Growth rate analysis
# - Trajectory similarity analysis
# 
# **Author**: Mike Keeman  
# **Date**: July 2025  
# **Version**: 1.0

# %% markdown
# ## Setup and Imports

# %%
import sys
import os
sys.path.append('../../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Analysis imports
from analysis.statistical_analyzer import StatisticalAnalyzer
from analysis.visualization_toolkit import VisualizationToolkit
from analysis.longitudinal_analyzer import LongitudinalAnalyzer
from analysis.data_export import DataExporter

# Model imports
from models import Persona, AssessmentResult, PHQ9Result, GAD7Result, PSS10Result
from models.persona import PersonalityTrait

# %%
# Setup plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Initialize analysis tools
stats_analyzer = StatisticalAnalyzer()
viz_toolkit = VisualizationToolkit(style="publication")
longitudinal_analyzer = LongitudinalAnalyzer()
data_exporter = DataExporter()

# %%
# Create output directory
output_dir = Path("../../data/results/longitudinal_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# %% markdown
# ## Load Simulation Data

# %%
# Load assessment data (sample data if files not found)
assessment_data_path = Path("../../data/simulation/assessment_results.csv")
if assessment_data_path.exists():
    assessment_data = pd.read_csv(assessment_data_path)
    print(f"Loaded {len(assessment_data)} assessment records")
else:
    # Create sample data for demonstration
    print("Assessment data not found, creating sample data...")
    np.random.seed(42)
    
    # Generate sample assessment data
    personas = ['persona_1', 'persona_2', 'persona_3']
    assessment_types = ['phq9', 'gad7', 'pss10']
    days = range(0, 30, 7)  # Weekly assessments
    
    assessment_data = []
    for persona in personas:
        for assessment_type in assessment_types:
            baseline_score = np.random.uniform(5, 15)
            for day in days:
                # Add some drift over time
                drift = np.random.normal(0, 0.5) * (day / 30)
                score = max(0, baseline_score + drift)
                
                assessment_data.append({
                    'assessment_id': f"{persona}_{assessment_type}_{day}",
                    'persona_id': persona,
                    'assessment_type': assessment_type,
                    'simulation_day': day,
                    'total_score': score,
                    'severity_level': 'mild' if score < 10 else 'moderate',
                    'response_consistency': np.random.uniform(0.7, 0.95),
                    'response_time_avg': np.random.uniform(2.0, 5.0),
                    'created_at': datetime.now().isoformat()
                })
    
    assessment_data = pd.DataFrame(assessment_data)
    print(f"Created sample data with {len(assessment_data)} records")

print("Assessment data summary:")
print(assessment_data.describe())

# %%
# Load mechanistic data (sample data if files not found)
mechanistic_data_path = Path("../../data/simulation/mechanistic_analysis.csv")
if mechanistic_data_path.exists():
    mechanistic_data = pd.read_csv(mechanistic_data_path)
    print(f"Loaded {len(mechanistic_data)} mechanistic records")
else:
    # Create sample mechanistic data
    print("Mechanistic data not found, creating sample data...")
    
    mechanistic_data = []
    layers = range(1, 13)  # 12 transformer layers
    
    for persona in personas:
        for day in days:
            for layer in layers:
                # Generate attention weights and activation values
                attention_weight = np.random.uniform(0.1, 0.9)
                activation_value = np.random.uniform(-1.0, 1.0)
                drift_magnitude = np.random.uniform(0.0, 0.2)
                
                # Create records for both attention and activation analysis types
                mechanistic_data.append({
                    'persona_id': persona,
                    'simulation_day': day,
                    'layer': layer,
                    'analysis_type': 'attention',
                    'attention_weight': attention_weight,
                    'activation_value': activation_value,
                    'drift_magnitude': drift_magnitude,
                    'created_at': datetime.now().isoformat()
                })
                
                mechanistic_data.append({
                    'persona_id': persona,
                    'simulation_day': day,
                    'layer': layer,
                    'analysis_type': 'activation',
                    'attention_weight': attention_weight,
                    'activation_value': activation_value,
                    'drift_magnitude': drift_magnitude,
                    'created_at': datetime.now().isoformat()
                })
    
    mechanistic_data = pd.DataFrame(mechanistic_data)
    print(f"Created sample mechanistic data with {len(mechanistic_data)} records")

print("Mechanistic data summary:")
print(mechanistic_data.describe())

# %% markdown
# ## Assessment Score Trajectories

# %%
# Analyze assessment trajectories for each type
assessment_types = ['phq9', 'gad7', 'pss10']

for assessment_type in assessment_types:
    print(f"\n=== {assessment_type.upper()} Trajectory Analysis ===")
    
    # Analyze trajectories
    trajectory_results = longitudinal_analyzer.analyze_assessment_trajectories(
        assessment_data, assessment_type
    )
    
    # Print summary for each persona
    for persona_id, trajectory in trajectory_results.items():
        if isinstance(trajectory, dict) and 'error' not in trajectory:
            print(f"\nPersona: {persona_id}")
            print(f"  Baseline Score: {trajectory['baseline_score']:.2f}")
            print(f"  Final Score: {trajectory['final_score']:.2f}")
            print(f"  Score Change: {trajectory['score_change']:.2f}")
            print(f"  Change Percentage: {trajectory['score_change_percentage']:.1f}%")
            print(f"  Assessment Count: {trajectory['assessment_count']}")
            
            # Trend analysis
            trend = trajectory['trend_analysis']
            print(f"  Trend Direction: {trend['direction']}")
            print(f"  Trend Significance: {trend['significant']}")
            print(f"  R-squared: {trend['r_squared']:.3f}")

# %%
# Visualize assessment trends
for assessment_type in assessment_types:
    fig = viz_toolkit.plot_assessment_trends(
        assessment_data, assessment_type, plot_type="line"
    )
    plt.savefig(output_dir / f"{assessment_type}_trends_line.png", bbox_inches='tight')
    plt.show()

# %%
# Box plots for assessment distributions
for assessment_type in assessment_types:
    fig = viz_toolkit.plot_assessment_trends(
        assessment_data, assessment_type, plot_type="box"
    )
    plt.savefig(output_dir / f"{assessment_type}_trends_box.png", bbox_inches='tight')
    plt.show()

# %% markdown
# ## Growth Rate Analysis

# %%
# Calculate growth rates for each assessment type
for assessment_type in assessment_types:
    print(f"\n=== {assessment_type.upper()} Growth Rate Analysis ===")
    
    growth_rates = longitudinal_analyzer.calculate_growth_rates(
        assessment_data, assessment_type
    )
    
    for persona_id, growth in growth_rates.items():
        print(f"\nPersona: {persona_id}")
        print(f"  Linear Growth Rate: {growth['linear_growth_rate']:.4f}")
        print(f"  Average Daily Change: {growth['avg_daily_change']:.4f}")
        print(f"  Acceleration: {growth['acceleration']:.4f}")
        print(f"  R-squared: {growth['r_squared']:.3f}")
        print(f"  Significant: {growth['significant']}")
        print(f"  Assessment Count: {growth['assessment_count']}")
        print(f"  Time Span: {growth['time_span']} days")

# %% markdown
# ## Change Point Detection

# %%
# Detect change points in assessment trajectories
for assessment_type in assessment_types:
    print(f"\n=== {assessment_type.upper()} Change Point Detection ===")
    
    type_data = assessment_data[assessment_data['assessment_type'] == assessment_type]
    
    for persona_id in type_data['persona_id'].unique():
        persona_data = type_data[type_data['persona_id'] == persona_id].sort_values('simulation_day')
        
        if len(persona_data) >= 3:
            change_points = stats_analyzer.detect_change_points(
                persona_data['total_score'].values, method="cusum"
            )
            
            print(f"\nPersona: {persona_id}")
            print(f"  Change Points: {change_points['change_points']}")
            print(f"  Method: {change_points['method']}")
            
            if change_points['change_points']:
                print(f"  Number of Change Points: {len(change_points['change_points'])}")
            else:
                print("  No significant change points detected")

# %% markdown
# ## Trajectory Pattern Detection

# %%
# Detect trajectory patterns
for assessment_type in assessment_types:
    print(f"\n=== {assessment_type.upper()} Trajectory Pattern Detection ===")
    
    patterns = longitudinal_analyzer.detect_trajectory_patterns(
        assessment_data, assessment_type
    )
    
    for persona_id, persona_patterns in patterns.items():
        print(f"\nPersona: {persona_id}")
        for pattern_type, pattern_data in persona_patterns.items():
            print(f"  {pattern_type}: {pattern_data}")

# %% markdown
# ## Personality Trait Drift Analysis

# %%
# Load persona data for trait analysis
persona_data_path = Path("../../config/personas")
if persona_data_path.exists():
    import yaml
    
    personas = []
    for file in persona_data_path.glob("*.yaml"):
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create sample state with drift
        from models import PersonaBaseline, PersonaState, Persona
        
        baseline = PersonaBaseline(**config)
        
        # Generate valid clinical scores within the model constraints
        # PHQ-9: 0-27, GAD-7: 0-21, PSS-10: 0-40
        current_phq9 = max(0.0, min(27.0, baseline.baseline_phq9 + np.random.normal(0, 2)))
        current_gad7 = max(0.0, min(21.0, baseline.baseline_gad7 + np.random.normal(0, 2)))
        current_pss10 = max(0.0, min(40.0, baseline.baseline_pss10 + np.random.normal(0, 2)))
        
        state = PersonaState(
            persona_id=baseline.name.lower().replace(" ", "_"),
            simulation_day=30,
            last_assessment_day=28,
            current_phq9=current_phq9,
            current_gad7=current_gad7,
            current_pss10=current_pss10,
            drift_magnitude=np.random.uniform(0.1, 0.3),
            emotional_state="neutral",
            stress_level=np.random.uniform(0.0, 5.0)  # Stress level 0-10, but keep reasonable
        )
        
        # Add some trait changes (keep small to avoid exceeding 0-1 bounds)
        trait_changes = {
            'openness': np.random.normal(0, 0.02),
            'conscientiousness': np.random.normal(0, 0.02),
            'extraversion': np.random.normal(0, 0.02),
            'agreeableness': np.random.normal(0, 0.02),
            'neuroticism': np.random.normal(0, 0.02)
        }
        state.trait_changes = trait_changes
        
        persona = Persona(baseline=baseline, state=state)
        personas.append(persona)
    
    print(f"Loaded {len(personas)} personas with drift data")
else:
    print("Persona config not found, skipping trait drift analysis")

# %%
# Analyze personality trait trajectories
if 'personas' in locals() and personas:
    trait_trajectories = longitudinal_analyzer.analyze_personality_trajectories(personas)
    
    print("=== Personality Trait Trajectory Analysis ===")
    
    for trait, trajectories in trait_trajectories.items():
        print(f"\n{trait.title()} Trajectories:")
        for trajectory in trajectories:
            print(f"  {trajectory['persona_id']}:")
            print(f"    Baseline: {trajectory['baseline_value']:.3f}")
            print(f"    Current: {trajectory['current_value']:.3f}")
            print(f"    Change: {trajectory['change_magnitude']:.3f}")
            print(f"    Change %: {trajectory['change_percentage']:.1f}%")

# %% markdown
# ## Mechanistic Analysis Trends

# %%
# Analyze mechanistic trajectories
analysis_types = ['attention', 'activation']

# Debug mechanistic data structure
print("=== Mechanistic Data Debug ===")
print(f"Mechanistic data shape: {mechanistic_data.shape}")
print(f"Mechanistic data columns: {mechanistic_data.columns.tolist()}")
print(f"Analysis types in data: {mechanistic_data['analysis_type'].unique()}")
print(f"Layers in data: {mechanistic_data['layer'].unique()}")
print(f"Personas in data: {mechanistic_data['persona_id'].unique()}")

for analysis_type in analysis_types:
    print(f"\n=== {analysis_type.title()} Mechanistic Analysis ===")
    
    # Check if this analysis type exists in the data
    type_data = mechanistic_data[mechanistic_data['analysis_type'] == analysis_type]
    print(f"Data for {analysis_type}: {len(type_data)} records")
    
    if len(type_data) == 0:
        print(f"No data found for analysis type: {analysis_type}")
        continue
    
    mechanistic_trajectories = longitudinal_analyzer.analyze_mechanistic_trajectories(
        mechanistic_data, analysis_type
    )
    
    # Add debugging to understand the structure
    print(f"Mechanistic trajectories type: {type(mechanistic_trajectories)}")
    print(f"Mechanistic trajectories keys: {list(mechanistic_trajectories.keys())}")
    
    for persona_id, persona_trajectory in mechanistic_trajectories.items():
        print(f"\nPersona: {persona_id}")
        print(f"Trajectory type: {type(persona_trajectory)}")
        
        # Check if it's an error message
        if isinstance(persona_trajectory, dict) and 'error' in persona_trajectory:
            print(f"Error: {persona_trajectory['error']}")
            continue
            
        # Check if it has the expected structure
        if isinstance(persona_trajectory, dict) and 'layers' in persona_trajectory:
            for layer, layer_trajectory in persona_trajectory['layers'].items():
                print(f"  Layer {layer}:")
                print(f"    Baseline Value: {layer_trajectory['baseline_value']:.3f}")
                print(f"    Final Value: {layer_trajectory['final_value']:.3f}")
                print(f"    Change: {layer_trajectory['value_change']:.3f}")
        else:
            print(f"Unexpected trajectory structure: {persona_trajectory}")

# %%
# Visualize mechanistic trends
for analysis_type in analysis_types:
    fig = viz_toolkit.plot_mechanistic_analysis(
        mechanistic_data, analysis_type, plot_type="line"
    )
    plt.savefig(output_dir / f"{analysis_type}_mechanistic_trends.png", bbox_inches='tight')
    plt.show()

# %%
# Mechanistic heatmaps
for analysis_type in analysis_types:
    fig = viz_toolkit.plot_mechanistic_analysis(
        mechanistic_data, analysis_type, plot_type="heatmap"
    )
    plt.savefig(output_dir / f"{analysis_type}_mechanistic_heatmap.png", bbox_inches='tight')
    plt.show()

# %% markdown
# ## Cross-Temporal Correlations

# %%
# Analyze correlations between assessment scores and mechanistic data
print("=== Cross-Temporal Correlation Analysis ===")

correlations = longitudinal_analyzer.analyze_cross_temporal_correlations(
    assessment_data, mechanistic_data
)

for correlation_key, lag_correlations in correlations.items():
    print(f"\n{correlation_key}:")
    for lag_key, lag_data in lag_correlations.items():
        if lag_data:
            avg_correlation = np.mean([c['correlation'] for c in lag_data])
            significant_count = sum(1 for c in lag_data if c['significant'])
            print(f"  {lag_key}: avg_corr={avg_correlation:.3f}, significant={significant_count}/{len(lag_data)}")

# %% markdown
# ## Trajectory Similarity Analysis

# %%
# Calculate trajectory similarity between personas
for assessment_type in assessment_types:
    print(f"\n=== {assessment_type.upper()} Trajectory Similarity ===")
    
    similarity_results = longitudinal_analyzer.calculate_trajectory_similarity(
        assessment_data, assessment_type
    )
    
    if 'error' not in similarity_results:
        print(f"Mean Similarity: {similarity_results['mean_similarity']:.3f}")
        print(f"Std Similarity: {similarity_results['std_similarity']:.3f}")
        print(f"Similarity Method: {similarity_results['similarity_method']}")
        
        # Show similarity matrix
        similarity_matrix = similarity_results['similarity_matrix']
        persona_ids = similarity_results['persona_ids']
        
        print("\nSimilarity Matrix:")
        print("          ", end="")
        for pid in persona_ids:
            print(f"{pid:>10}", end="")
        print()
        
        for i, pid1 in enumerate(persona_ids):
            print(f"{pid1:>10}", end="")
            for j, pid2 in enumerate(persona_ids):
                sim = similarity_matrix[i, j]
                if np.isnan(sim):
                    print(f"{'N/A':>10}", end="")
                else:
                    print(f"{sim:>10.3f}", end="")
            print()

# %% markdown
# ## Interactive Dashboard

# %%
# Create proper persona data structure for dashboard
print("Creating persona data for dashboard...")

# Create persona data with trait information
persona_dashboard_data = []
traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

for persona_id in mechanistic_data['persona_id'].unique():
    for day in mechanistic_data['simulation_day'].unique():
        # Get baseline trait values (simulate from persona config or use defaults)
        baseline_traits = {
            'openness': 0.6,
            'conscientiousness': 0.7,
            'extraversion': 0.5,
            'agreeableness': 0.8,
            'neuroticism': 0.3
        }
        
        # Add some drift over time
        for trait in traits:
            drift = np.random.normal(0, 0.01) * (day / 30)  # Small drift over time
            current_value = max(0.0, min(1.0, baseline_traits[trait] + drift))
            
            persona_dashboard_data.append({
                'persona_id': persona_id,
                'simulation_day': day,
                'trait': trait,
                'trait_value': current_value
            })

persona_dashboard_df = pd.DataFrame(persona_dashboard_data)
print(f"Created persona dashboard data: {len(persona_dashboard_df)} records")

# %%
# Create interactive dashboard
print("Creating interactive dashboard...")
print(f"Assessment data shape: {assessment_data.shape}")
print(f"Mechanistic data shape: {mechanistic_data.shape}")
print(f"Persona dashboard data shape: {persona_dashboard_df.shape}")
print(f"Persona dashboard columns: {persona_dashboard_df.columns.tolist()}")

try:
    dashboard = viz_toolkit.create_interactive_dashboard(
        assessment_data, mechanistic_data, persona_dashboard_df
    )
    
    # Save dashboard
    dashboard.write_html(str(output_dir / "interactive_dashboard.html"))
    print(f"Interactive dashboard saved to: {output_dir / 'interactive_dashboard.html'}")
    
except Exception as e:
    print(f"Error creating dashboard: {e}")
    import traceback
    traceback.print_exc()

# %% markdown
# ## Summary and Export

# %%
# Helper function to convert numpy types to native Python types
def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, pd.DataFrame):
        # Convert DataFrame to dict with converted values
        return convert_numpy_types(obj.to_dict('records'))
    elif isinstance(obj, pd.Series):
        # Convert Series to list with converted values
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bytes_):
        return str(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif hasattr(obj, 'item'):  # Handle other numpy scalars
        return obj.item()
    elif hasattr(obj, 'dtype'):  # Handle pandas/numpy objects with dtype
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return str(obj)
    elif str(type(obj)).startswith("<class 'numpy."):  # Catch any remaining numpy types
        try:
            return obj.item()
        except:
            return str(obj)
    else:
        return obj

# Generate summary statistics
summary_stats = {
    'total_assessments': int(len(assessment_data)),
    'unique_personas': convert_numpy_types(assessment_data['persona_id'].nunique()),
    'assessment_types': convert_numpy_types(assessment_data['assessment_type'].unique().tolist()),
    'simulation_days': convert_numpy_types(assessment_data['simulation_day'].max()),
    'mechanistic_records': int(len(mechanistic_data)),
    'analysis_timestamp': datetime.now().isoformat()
}

# Convert summary_stats to ensure all values are JSON serializable
summary_stats = convert_numpy_types(summary_stats)

print("=== Longitudinal Analysis Summary ===")
print(json.dumps(summary_stats, indent=2))

# %%
# Export results
# Create proper AssessmentResult objects for export
assessment_results = []
for _, row in assessment_data.iterrows():
    # Convert severity_level string to enum
    from models.assessment import SeverityLevel, AssessmentResult
    
    severity_map = {
        'minimal': SeverityLevel.MINIMAL,
        'mild': SeverityLevel.MILD,
        'moderate': SeverityLevel.MODERATE,
        'severe': SeverityLevel.SEVERE
    }
    
    # Create assessment result with proper data
    assessment_result = AssessmentResult(
        assessment_id=row['assessment_id'],
        persona_id=row['persona_id'],
        assessment_type=row['assessment_type'],
        simulation_day=row['simulation_day'],
        total_score=row['total_score'],
        severity_level=severity_map.get(row['severity_level'], SeverityLevel.MILD),
        response_consistency=row['response_consistency'],
        response_time_avg=row['response_time_avg'],
        raw_responses=[],  # Empty for sample data
        parsed_scores=[],  # Empty for sample data
        created_at=datetime.fromisoformat(row['created_at'])
    )
    assessment_results.append(assessment_result)

data_exporter.export_assessment_data(assessment_results, format="csv")

data_exporter.export_mechanistic_data(mechanistic_data, format="csv")

# Export research summary
print("Debugging summary_stats before export:")
print(f"Summary stats type: {type(summary_stats)}")
print(f"Summary stats keys: {list(summary_stats.keys())}")

# Debug the data types in the DataFrames
print("\nDebugging DataFrame data types:")
print(f"Assessment data dtypes: {assessment_data.dtypes}")
print(f"Mechanistic data dtypes: {mechanistic_data.dtypes}")

# Check for any numpy types in the summary_stats
print("\nDebugging summary_stats contents:")
for key, value in summary_stats.items():
    print(f"  {key}: {type(value)} = {value}")

try:
    # Final conversion to ensure all numpy types are handled
    summary_stats = convert_numpy_types(summary_stats)
    
    research_summary = data_exporter.export_research_summary(
        assessment_data, 
        pd.DataFrame(),  # Empty persona data for now
        mechanistic_data,
        summary_stats,
        format="json"
    )
    print(f"Research summary exported to: {research_summary}")
except Exception as e:
    print(f"Error during research summary export: {e}")
    import traceback
    traceback.print_exc()

print(f"Analysis complete. Results saved to: {output_dir}") 