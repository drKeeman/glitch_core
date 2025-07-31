# %% markdown
# AI Personality Drift Simulation - Persona Baseline Analysis
# 
# This notebook analyzes the baseline characteristics of our AI personas, including:
# - Personality trait distributions
# - Clinical baseline scores
# - Persona demographics and characteristics
# - Baseline reliability and validity checks
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
from analysis.data_export import DataExporter

# Model imports
from models import Persona, PersonaBaseline, PersonaState
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
data_exporter = DataExporter()

# %%
# Create output directory
output_dir = Path("../../data/results/baseline_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# %% markdown
# ## Load Persona Data

# %%
# Load persona configurations
config_dir = Path("../../config/personas")
persona_files = list(config_dir.glob("*.yaml"))

print(f"Found {len(persona_files)} persona configuration files:")
for file in persona_files:
    print(f"  - {file.name}")

# %%
# Load and parse persona data
import yaml

personas = []
for file in persona_files:
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create PersonaBaseline
    baseline = PersonaBaseline(**config)
    
    # Create PersonaState (initial state)
    state = PersonaState(
        persona_id=baseline.name.lower().replace(" ", "_"),
        simulation_day=0,
        last_assessment_day=-1,
        current_phq9=baseline.baseline_phq9,
        current_gad7=baseline.baseline_gad7,
        current_pss10=baseline.baseline_pss10,
        drift_magnitude=0.0,
        emotional_state="neutral",
        stress_level=0.0
    )
    
    # Create Persona
    persona = Persona(baseline=baseline, state=state)
    personas.append(persona)

print(f"Loaded {len(personas)} personas:")
for persona in personas:
    print(f"  - {persona.baseline.name} (ID: {persona.state.persona_id})")

# %% markdown
# ## Personality Trait Analysis

# %%
# Extract personality traits for analysis
trait_data = []
for persona in personas:
    traits = persona.baseline.get_traits_dict()
    traits['persona_id'] = persona.state.persona_id
    traits['name'] = persona.baseline.name
    traits['age'] = persona.baseline.age
    traits['occupation'] = persona.baseline.occupation
    trait_data.append(traits)

trait_df = pd.DataFrame(trait_data)
print("Personality trait summary:")
print(trait_df.describe())

# %%
# Visualize personality traits
fig = viz_toolkit.plot_personality_traits(personas, plot_type="radar")
plt.savefig(output_dir / "personality_traits_radar.png", bbox_inches='tight')
plt.show()

# %%
# Bar plot of personality traits
fig = viz_toolkit.plot_personality_traits(personas, plot_type="bar")
plt.savefig(output_dir / "personality_traits_bar.png", bbox_inches='tight')
plt.show()

# %%
# Heatmap of personality traits
fig = viz_toolkit.plot_personality_traits(personas, plot_type="heatmap")
plt.savefig(output_dir / "personality_traits_heatmap.png", bbox_inches='tight')
plt.show()

# %% markdown
# ## Clinical Baseline Analysis

# %%
# Extract clinical baseline scores
clinical_data = []
for persona in personas:
    clinical_data.append({
        'persona_id': persona.state.persona_id,
        'name': persona.baseline.name,
        'phq9_baseline': persona.baseline.baseline_phq9,
        'gad7_baseline': persona.baseline.baseline_gad7,
        'pss10_baseline': persona.baseline.baseline_pss10,
        'age': persona.baseline.age,
        'occupation': persona.baseline.occupation
    })

clinical_df = pd.DataFrame(clinical_data)
print("Clinical baseline summary:")
print(clinical_df.describe())

# %%
# Visualize clinical baseline scores
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# PHQ-9 baseline
axes[0].bar(range(len(clinical_df)), clinical_df['phq9_baseline'])
axes[0].set_title('PHQ-9 Baseline Scores')
axes[0].set_xlabel('Persona')
axes[0].set_ylabel('PHQ-9 Score')
axes[0].set_xticks(range(len(clinical_df)))
axes[0].set_xticklabels(clinical_df['name'], rotation=45)

# GAD-7 baseline
axes[1].bar(range(len(clinical_df)), clinical_df['gad7_baseline'])
axes[1].set_title('GAD-7 Baseline Scores')
axes[1].set_xlabel('Persona')
axes[1].set_ylabel('GAD-7 Score')
axes[1].set_xticks(range(len(clinical_df)))
axes[1].set_xticklabels(clinical_df['name'], rotation=45)

# PSS-10 baseline
axes[2].bar(range(len(clinical_df)), clinical_df['pss10_baseline'])
axes[2].set_title('PSS-10 Baseline Scores')
axes[2].set_xlabel('Persona')
axes[2].set_ylabel('PSS-10 Score')
axes[2].set_xticks(range(len(clinical_df)))
axes[2].set_xticklabels(clinical_df['name'], rotation=45)

plt.tight_layout()
plt.savefig(output_dir / "clinical_baseline_scores.png", bbox_inches='tight')
plt.show()

# %% markdown
# ## Correlation Analysis

# %%
# Analyze correlations between personality traits and clinical scores
correlation_vars = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism',
                   'phq9_baseline', 'gad7_baseline', 'pss10_baseline']

correlation_data = trait_df.merge(clinical_df[['persona_id', 'phq9_baseline', 'gad7_baseline', 'pss10_baseline']], 
                                 on='persona_id')

# Create correlation matrix
corr_matrix = correlation_data[correlation_vars].corr()

# Visualize correlation matrix
fig = viz_toolkit.plot_correlation_matrix(correlation_data, variables=correlation_vars)
plt.savefig(output_dir / "personality_clinical_correlations.png", bbox_inches='tight')
plt.show()

# %%
# Statistical significance of correlations
print("Correlation analysis between personality traits and clinical scores:")
print("=" * 60)

for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
    for clinical in ['phq9_baseline', 'gad7_baseline', 'pss10_baseline']:
        correlation = stats_analyzer.correlation_analysis(
            correlation_data[trait].values,
            correlation_data[clinical].values,
            method="pearson"
        )
        
        print(f"{trait.title()} vs {clinical.replace('_baseline', '').upper()}:")
        print(f"  Correlation: {correlation['correlation']:.3f}")
        print(f"  P-value: {correlation['p_value']:.3f}")
        print(f"  Significant: {correlation['significant']}")
        print(f"  Strength: {correlation['strength']}")
        print()

# %% markdown
# ## Demographics Analysis

# %%
# Analyze persona demographics
demographics = clinical_df[['name', 'age', 'occupation']].copy()

print("Persona demographics:")
print("=" * 30)
for _, row in demographics.iterrows():
    print(f"Name: {row['name']}")
    print(f"Age: {row['age']}")
    print(f"Occupation: {row['occupation']}")
    print("-" * 20)

# %%
# Age distribution
plt.figure(figsize=(8, 6))
plt.hist(demographics['age'], bins=10, alpha=0.7, edgecolor='black')
plt.title('Age Distribution of AI Personas')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig(output_dir / "age_distribution.png", bbox_inches='tight')
plt.show()

# %%
# Occupation analysis
occupation_counts = demographics['occupation'].value_counts()
print("Occupation distribution:")
print(occupation_counts)

plt.figure(figsize=(10, 6))
occupation_counts.plot(kind='bar')
plt.title('Occupation Distribution of AI Personas')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "occupation_distribution.png", bbox_inches='tight')
plt.show()

# %% markdown
# ## Baseline Reliability Analysis

# %%
# Analyze baseline consistency across personas
print("Baseline reliability analysis:")
print("=" * 40)

# Personality trait reliability (internal consistency across personas)
for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
    trait_values = trait_df[trait].values
    
    # Calculate coefficient of variation
    cv = np.std(trait_values) / np.mean(trait_values)
    
    print(f"{trait.title()}:")
    print(f"  Mean: {np.mean(trait_values):.3f}")
    print(f"  Std: {np.std(trait_values):.3f}")
    print(f"  CV: {cv:.3f}")
    print(f"  Range: {np.min(trait_values):.3f} - {np.max(trait_values):.3f}")
    print()

# %%
# Clinical score reliability
clinical_scores = ['phq9_baseline', 'gad7_baseline', 'pss10_baseline']
clinical_names = ['PHQ-9', 'GAD-7', 'PSS-10']

print("Clinical baseline reliability:")
print("=" * 35)

for score, name in zip(clinical_scores, clinical_names):
    values = clinical_df[score].values
    
    # Calculate coefficient of variation
    cv = np.std(values) / np.mean(values)
    
    print(f"{name}:")
    print(f"  Mean: {np.mean(values):.3f}")
    print(f"  Std: {np.std(values):.3f}")
    print(f"  CV: {cv:.3f}")
    print(f"  Range: {np.min(values):.3f} - {np.max(values):.3f}")
    print()

# %% markdown
# ## Baseline Validation

# %%
# Validate baseline scores against clinical thresholds
print("Baseline clinical validation:")
print("=" * 35)

# PHQ-9 thresholds
phq9_thresholds = {
    'minimal': 5,
    'mild': 10,
    'moderate': 15,
    'severe': 20
}

print("PHQ-9 Baseline Classification:")
for _, row in clinical_df.iterrows():
    score = row['phq9_baseline']
    if score < phq9_thresholds['minimal']:
        severity = 'minimal'
    elif score < phq9_thresholds['mild']:
        severity = 'mild'
    elif score < phq9_thresholds['moderate']:
        severity = 'moderate'
    else:
        severity = 'severe'
    
    print(f"  {row['name']}: {score} ({severity})")

# %%
# GAD-7 thresholds
gad7_thresholds = {
    'minimal': 5,
    'mild': 10,
    'moderate': 15,
    'severe': 20
}

print("GAD-7 Baseline Classification:")
for _, row in clinical_df.iterrows():
    score = row['gad7_baseline']
    if score < gad7_thresholds['minimal']:
        severity = 'minimal'
    elif score < gad7_thresholds['mild']:
        severity = 'mild'
    elif score < gad7_thresholds['moderate']:
        severity = 'moderate'
    else:
        severity = 'severe'
    
    print(f"  {row['name']}: {score} ({severity})")

# %%
# PSS-10 thresholds
pss10_thresholds = {
    'minimal': 13,
    'mild': 16,
    'moderate': 19,
    'severe': 22
}

print("PSS-10 Baseline Classification:")
for _, row in clinical_df.iterrows():
    score = row['pss10_baseline']
    if score < pss10_thresholds['minimal']:
        severity = 'minimal'
    elif score < pss10_thresholds['mild']:
        severity = 'mild'
    elif score < pss10_thresholds['moderate']:
        severity = 'moderate'
    else:
        severity = 'severe'
    
    print(f"  {row['name']}: {score} ({severity})")

# %% markdown
# ## Summary Statistics

# %%
# Generate comprehensive summary
summary_stats = {
    'total_personas': len(personas),
    'personality_traits': {
        'mean': trait_df[['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']].mean().to_dict(),
        'std': trait_df[['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']].std().to_dict(),
        'range': {
            trait: f"{trait_df[trait].min():.3f} - {trait_df[trait].max():.3f}"
            for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        }
    },
    'clinical_baselines': {
        'phq9': {
            'mean': clinical_df['phq9_baseline'].mean(),
            'std': clinical_df['phq9_baseline'].std(),
            'range': f"{clinical_df['phq9_baseline'].min():.3f} - {clinical_df['phq9_baseline'].max():.3f}"
        },
        'gad7': {
            'mean': clinical_df['gad7_baseline'].mean(),
            'std': clinical_df['gad7_baseline'].std(),
            'range': f"{clinical_df['gad7_baseline'].min():.3f} - {clinical_df['gad7_baseline'].max():.3f}"
        },
        'pss10': {
            'mean': clinical_df['pss10_baseline'].mean(),
            'std': clinical_df['pss10_baseline'].std(),
            'range': f"{clinical_df['pss10_baseline'].min():.3f} - {clinical_df['pss10_baseline'].max():.3f}"
        }
    },
    'demographics': {
        'age_range': f"{demographics['age'].min()} - {demographics['age'].max()}",
        'mean_age': demographics['age'].mean(),
        'occupations': demographics['occupation'].unique().tolist()
    }
}

print("Baseline Analysis Summary:")
print("=" * 30)
print(json.dumps(summary_stats, indent=2))

# %%
# Save summary to file
with open(output_dir / "baseline_summary.json", 'w') as f:
    json.dump(summary_stats, f, indent=2)

# Export data
data_exporter.export_persona_data(personas, format="csv")
data_exporter.export_persona_data(personas, format="json")

print(f"Analysis complete. Results saved to: {output_dir}") 