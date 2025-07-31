# %% markdown
# AI Personality Drift Simulation - Descriptive Statistics
# 
# This notebook generates descriptive statistics and publication-quality figures for the research paper, including:
# - Summary statistics for all variables
# - Distribution plots and visualizations
# - Correlation matrices
# - Condition comparisons
# - Baseline analysis
# - Publication-ready figure generation
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
from models import Persona, AssessmentResult, PHQ9Result, GAD7Result, PSS10Result
from models.persona import PersonalityTrait

# %%
# Setup plotting style for publication
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

# Initialize analysis tools
stats_analyzer = StatisticalAnalyzer()
viz_toolkit = VisualizationToolkit(style="publication")
data_exporter = DataExporter()

# %%
# Create output directory
output_dir = Path("../../data/results/publication")
output_dir.mkdir(parents=True, exist_ok=True)

# %% markdown
# ## Load Research Data

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
    personas = ['persona_1', 'persona_2', 'persona_3', 'persona_4', 'persona_5']
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
# Load persona data (sample data if files not found)
persona_data_path = Path("../../data/simulation/persona_data.csv")
if persona_data_path.exists():
    persona_data = pd.read_csv(persona_data_path)
    print(f"Loaded {len(persona_data)} persona records")
else:
    # Create sample persona data
    print("Persona data not found, creating sample data...")
    
    persona_data = []
    for i, persona in enumerate(personas):
        persona_data.append({
            'persona_id': persona,
            'name': f'Persona {i+1}',
            'age': np.random.randint(25, 65),
            'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist', 'Manager']),
            'openness': np.random.uniform(0.3, 0.8),
            'conscientiousness': np.random.uniform(0.4, 0.9),
            'extraversion': np.random.uniform(0.2, 0.7),
            'agreeableness': np.random.uniform(0.5, 0.9),
            'neuroticism': np.random.uniform(0.1, 0.6),
            'baseline_phq9': np.random.uniform(3, 12),
            'baseline_gad7': np.random.uniform(2, 10),
            'baseline_pss10': np.random.uniform(8, 18),
            'drift_magnitude': np.random.uniform(0.05, 0.25),
            'emotional_state': np.random.choice(['neutral', 'anxious', 'depressed', 'content']),
            'stress_level': np.random.uniform(0.1, 0.6)
        })
    
    persona_data = pd.DataFrame(persona_data)
    print(f"Created sample persona data with {len(persona_data)} records")

print("Persona data summary:")
print(persona_data.describe())

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
    
    mechanistic_data = pd.DataFrame(mechanistic_data)
    print(f"Created sample mechanistic data with {len(mechanistic_data)} records")

print("Mechanistic data summary:")
print(mechanistic_data.describe())

# %% markdown
# ## Comprehensive Descriptive Statistics

# %%
# Assessment data descriptive statistics
print("=== Assessment Data Descriptive Statistics ===")
print(assessment_data.describe())

# %%
# Persona data descriptive statistics
print("=== Persona Data Descriptive Statistics ===")
print(persona_data.describe())

# %%
# Mechanistic data descriptive statistics
print("=== Mechanistic Data Descriptive Statistics ===")
print(mechanistic_data.describe())

# %%
# Assessment scores by type
print("=== Assessment Scores by Type ===")
for assessment_type in assessment_data['assessment_type'].unique():
    type_data = assessment_data[assessment_data['assessment_type'] == assessment_type]['total_score']
    print(f"\n{assessment_type.upper()}:")
    print(f"  Mean: {type_data.mean():.2f}")
    print(f"  Std: {type_data.std():.2f}")
    print(f"  Min: {type_data.min():.2f}")
    print(f"  Max: {type_data.max():.2f}")
    print(f"  Median: {type_data.median():.2f}")

# %%
# Personality traits descriptive statistics
trait_columns = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
print("=== Personality Traits Descriptive Statistics ===")
for trait in trait_columns:
    trait_data = persona_data[trait]
    print(f"\n{trait.title()}:")
    print(f"  Mean: {trait_data.mean():.3f}")
    print(f"  Std: {trait_data.std():.3f}")
    print(f"  Min: {trait_data.min():.3f}")
    print(f"  Max: {trait_data.max():.3f}")
    print(f"  Median: {trait_data.median():.3f}")

# %% markdown
# ## Distribution Analysis

# %%
# Distribution of assessment scores
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, assessment_type in enumerate(['phq9', 'gad7', 'pss10']):
    type_data = assessment_data[assessment_data['assessment_type'] == assessment_type]['total_score']
    
    axes[i].hist(type_data, bins=15, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{assessment_type.upper()} Score Distribution')
    axes[i].set_xlabel('Score')
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "assessment_score_distributions.png", bbox_inches='tight')
plt.show()

# %%
# Distribution of personality traits
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, trait in enumerate(trait_columns):
    row = i // 3
    col = i % 3
    
    trait_data = persona_data[trait]
    axes[row, col].hist(trait_data, bins=10, alpha=0.7, edgecolor='black')
    axes[row, col].set_title(f'{trait.title()} Distribution')
    axes[row, col].set_xlabel('Trait Value')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True, alpha=0.3)

# Remove the last subplot if not needed
if len(trait_columns) < 6:
    axes[1, 2].remove()

plt.tight_layout()
plt.savefig(output_dir / "personality_trait_distributions.png", bbox_inches='tight')
plt.show()

# %%
# Distribution of mechanistic data
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Attention weights
attention_data = mechanistic_data['attention_weight']
axes[0].hist(attention_data, bins=20, alpha=0.7, edgecolor='black')
axes[0].set_title('Attention Weight Distribution')
axes[0].set_xlabel('Attention Weight')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

# Activation values
activation_data = mechanistic_data['activation_value']
axes[1].hist(activation_data, bins=20, alpha=0.7, edgecolor='black')
axes[1].set_title('Activation Value Distribution')
axes[1].set_xlabel('Activation Value')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

# Drift magnitude
drift_data = mechanistic_data['drift_magnitude']
axes[2].hist(drift_data, bins=15, alpha=0.7, edgecolor='black')
axes[2].set_title('Drift Magnitude Distribution')
axes[2].set_xlabel('Drift Magnitude')
axes[2].set_ylabel('Frequency')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "mechanistic_data_distributions.png", bbox_inches='tight')
plt.show()

# %% markdown
# ## Correlation Analysis

# %%
# Correlation matrix for assessment scores
assessment_corr = assessment_data[['total_score', 'response_consistency', 'response_time_avg']].corr()
print("=== Assessment Data Correlation Matrix ===")
print(assessment_corr)

# %%
# Correlation matrix for personality traits
personality_corr = persona_data[trait_columns].corr()
print("=== Personality Traits Correlation Matrix ===")
print(personality_corr)

# %%
# Visualize correlation matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Assessment correlations
sns.heatmap(assessment_corr, annot=True, cmap='RdBu_r', center=0, 
           square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[0])
axes[0].set_title('Assessment Data Correlations')

# Personality correlations
sns.heatmap(personality_corr, annot=True, cmap='RdBu_r', center=0, 
           square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[1])
axes[1].set_title('Personality Traits Correlations')

plt.tight_layout()
plt.savefig(output_dir / "correlation_matrices.png", bbox_inches='tight')
plt.show()

# %%
# Correlation between personality traits and clinical scores
clinical_traits = persona_data[['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism',
                               'baseline_phq9', 'baseline_gad7', 'baseline_pss10']]
clinical_corr = clinical_traits.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(clinical_corr, annot=True, cmap='RdBu_r', center=0, 
           square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title('Personality Traits vs Clinical Scores Correlations')

plt.tight_layout()
plt.savefig(output_dir / "personality_clinical_correlations.png", bbox_inches='tight')
plt.show()

# %% markdown
# ## Condition Comparisons

# %%
# Compare assessment scores across personas
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, assessment_type in enumerate(['phq9', 'gad7', 'pss10']):
    type_data = assessment_data[assessment_data['assessment_type'] == assessment_type]
    
    # Box plot
    type_data.boxplot(column='total_score', by='persona_id', ax=axes[i])
    axes[i].set_title(f'{assessment_type.upper()} Scores by Persona')
    axes[i].set_xlabel('Persona')
    axes[i].set_ylabel('Score')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(output_dir / "assessment_scores_by_persona.png", bbox_inches='tight')
plt.show()

# %%
# Compare personality traits across personas
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, trait in enumerate(trait_columns):
    row = i // 3
    col = i % 3
    
    # Box plot
    persona_data.boxplot(column=trait, by='persona_id', ax=axes[row, col])
    axes[row, col].set_title(f'{trait.title()} by Persona')
    axes[row, col].set_xlabel('Persona')
    axes[row, col].set_ylabel('Trait Value')
    axes[row, col].tick_params(axis='x', rotation=45)

# Remove the last subplot if not needed
if len(trait_columns) < 6:
    axes[1, 2].remove()

plt.tight_layout()
plt.savefig(output_dir / "personality_traits_by_persona.png", bbox_inches='tight')
plt.show()

# %%
# Statistical comparison of assessment scores across personas
print("=== Statistical Comparison of Assessment Scores ===")

for assessment_type in ['phq9', 'gad7', 'pss10']:
    print(f"\n{assessment_type.upper()} Scores:")
    type_data = assessment_data[assessment_data['assessment_type'] == assessment_type]
    
    # One-way ANOVA
    from scipy.stats import f_oneway
    
    persona_groups = [group['total_score'].values for name, group in type_data.groupby('persona_id')]
    f_stat, p_value = f_oneway(*persona_groups)
    
    print(f"  One-way ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
    print(f"  Significant difference: {p_value < 0.05}")

# %% markdown
# ## Baseline Analysis

# %%
# Baseline clinical scores analysis
print("=== Baseline Clinical Scores Analysis ===")

baseline_scores = ['baseline_phq9', 'baseline_gad7', 'baseline_pss10']
score_names = ['PHQ-9', 'GAD-7', 'PSS-10']

for score, name in zip(baseline_scores, score_names):
    score_data = persona_data[score]
    print(f"\n{name} Baseline Scores:")
    print(f"  Mean: {score_data.mean():.2f}")
    print(f"  Std: {score_data.std():.2f}")
    print(f"  Min: {score_data.min():.2f}")
    print(f"  Max: {score_data.max():.2f}")
    print(f"  Median: {score_data.median():.2f}")

# %%
# Visualize baseline scores
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (score, name) in enumerate(zip(baseline_scores, score_names)):
    score_data = persona_data[score]
    
    axes[i].bar(range(len(score_data)), score_data, alpha=0.7)
    axes[i].set_title(f'{name} Baseline Scores')
    axes[i].set_xlabel('Persona')
    axes[i].set_ylabel('Score')
    axes[i].set_xticks(range(len(score_data)))
    axes[i].set_xticklabels(persona_data['persona_id'], rotation=45)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "baseline_clinical_scores.png", bbox_inches='tight')
plt.show()

# %%
# Baseline personality traits
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, trait in enumerate(trait_columns):
    row = i // 3
    col = i % 3
    
    trait_data = persona_data[trait]
    
    axes[row, col].bar(range(len(trait_data)), trait_data, alpha=0.7)
    axes[row, col].set_title(f'{trait.title()} Baseline Values')
    axes[row, col].set_xlabel('Persona')
    axes[row, col].set_ylabel('Trait Value')
    axes[row, col].set_xticks(range(len(trait_data)))
    axes[row, col].set_xticklabels(persona_data['persona_id'], rotation=45)
    axes[row, col].grid(True, alpha=0.3)

# Remove the last subplot if not needed
if len(trait_columns) < 6:
    axes[1, 2].remove()

plt.tight_layout()
plt.savefig(output_dir / "baseline_personality_traits.png", bbox_inches='tight')
plt.show()

# %% markdown
# ## Statistical Summary Tables

# %%
# Create comprehensive summary table
print("=== Comprehensive Statistical Summary ===")

# Assessment data summary
assessment_summary = assessment_data.groupby('assessment_type')['total_score'].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).round(2)

print("\nAssessment Scores Summary:")
print(assessment_summary)

# %%
# Personality traits summary
personality_summary = persona_data[trait_columns].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).round(3)

print("\nPersonality Traits Summary:")
print(personality_summary)

# %%
# Clinical baseline summary
clinical_summary = persona_data[baseline_scores].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).round(2)

print("\nClinical Baseline Summary:")
print(clinical_summary)

# %%
# Mechanistic data summary
mechanistic_summary = mechanistic_data[['attention_weight', 'activation_value', 'drift_magnitude']].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).round(3)

print("\nMechanistic Data Summary:")
print(mechanistic_summary)

# %% markdown
# ## Publication-Quality Figure Generation

# %%
# Create comprehensive publication figure
fig = plt.figure(figsize=(20, 16))

# Create subplots
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# 1. Assessment score distributions (top left)
ax1 = fig.add_subplot(gs[0, 0])
for assessment_type in ['phq9', 'gad7', 'pss10']:
    type_data = assessment_data[assessment_data['assessment_type'] == assessment_type]['total_score']
    ax1.hist(type_data, alpha=0.6, label=assessment_type.upper(), bins=15)
ax1.set_title('Assessment Score Distributions')
ax1.set_xlabel('Score')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Personality traits radar (top right)
ax2 = fig.add_subplot(gs[0, 1], projection='polar')
angles = np.linspace(0, 2 * np.pi, len(trait_columns), endpoint=False).tolist()
angles += angles[:1]

for i, persona in enumerate(persona_data['persona_id']):
    values = [persona_data[persona_data['persona_id'] == persona][trait].iloc[0] for trait in trait_columns]
    values += values[:1]
    ax2.plot(angles, values, 'o-', linewidth=2, label=persona, alpha=0.7)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels([trait.title() for trait in trait_columns])
ax2.set_ylim(0, 1)
ax2.set_title('Personality Traits Comparison')
ax2.legend(bbox_to_anchor=(1.3, 1.0))

# 3. Assessment trends over time (middle left)
ax3 = fig.add_subplot(gs[1, 0])
for assessment_type in ['phq9', 'gad7', 'pss10']:
    type_data = assessment_data[assessment_data['assessment_type'] == assessment_type]
    ax3.plot(type_data.groupby('simulation_day')['total_score'].mean(), 
             marker='o', label=assessment_type.upper(), linewidth=2)
ax3.set_title('Assessment Trends Over Time')
ax3.set_xlabel('Simulation Day')
ax3.set_ylabel('Mean Score')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Correlation matrix (middle right)
ax4 = fig.add_subplot(gs[1, 1])
correlation_data = persona_data[trait_columns + baseline_scores]
corr_matrix = correlation_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
           square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax4)
ax4.set_title('Personality-Clinical Correlations')

# 5. Mechanistic data heatmap (bottom left)
ax5 = fig.add_subplot(gs[2, 0])
attention_pivot = mechanistic_data.pivot_table(
    values='attention_weight', 
    index='layer', 
    columns='simulation_day', 
    aggfunc='mean'
)
sns.heatmap(attention_pivot, cmap='viridis', ax=ax5)
ax5.set_title('Attention Weights by Layer and Time')

# 6. Box plots for assessment scores (bottom right)
ax6 = fig.add_subplot(gs[2, 1])
assessment_data.boxplot(column='total_score', by='assessment_type', ax=ax6)
ax6.set_title('Assessment Scores by Type')
ax6.set_xlabel('Assessment Type')
ax6.set_ylabel('Score')

# 7. Drift magnitude over time (bottom)
ax7 = fig.add_subplot(gs[3, :2])
for persona in mechanistic_data['persona_id'].unique():
    persona_data = mechanistic_data[mechanistic_data['persona_id'] == persona]
    ax7.plot(persona_data.groupby('simulation_day')['drift_magnitude'].mean(), 
             marker='o', label=persona, linewidth=2)
ax7.set_title('Drift Magnitude Over Time')
ax7.set_xlabel('Simulation Day')
ax7.set_ylabel('Mean Drift Magnitude')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Summary statistics table (bottom right)
ax8 = fig.add_subplot(gs[3, 2:])
ax8.axis('tight')
ax8.axis('off')

# Create summary table
summary_data = [
    ['Total Assessments', len(assessment_data)],
    ['Unique Personas', assessment_data['persona_id'].nunique()],
    ['Assessment Types', len(assessment_data['assessment_type'].unique())],
    ['Simulation Days', assessment_data['simulation_day'].max()],
    ['Mechanistic Records', len(mechanistic_data)],
    ['Personality Traits', len(trait_columns)],
    ['Neural Layers', mechanistic_data['layer'].nunique()]
]

table = ax8.table(cellText=summary_data, colLabels=['Metric', 'Value'], 
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
ax8.set_title('Dataset Summary')

plt.suptitle('AI Personality Drift Simulation: Comprehensive Analysis', fontsize=16, y=0.98)
plt.savefig(output_dir / "comprehensive_publication_figure.png", bbox_inches='tight', dpi=300)
plt.show()

# %% markdown
# ## Export Results

# %%
# Export data for publication
publication_files = data_exporter.export_for_publication(
    assessment_data, persona_data, mechanistic_data
)

print("=== Publication Data Export ===")
for data_type, file_path in publication_files.items():
    print(f"{data_type}: {file_path}")

# %%
# Export summary statistics
summary_stats = {
    'dataset_summary': {
        'total_assessments': len(assessment_data),
        'unique_personas': assessment_data['persona_id'].nunique(),
        'assessment_types': assessment_data['assessment_type'].unique().tolist(),
        'simulation_days': assessment_data['simulation_day'].max(),
        'mechanistic_records': len(mechanistic_data),
        'personality_traits': trait_columns,
        'neural_layers': mechanistic_data['layer'].nunique()
    },
    'assessment_summary': assessment_summary.to_dict(),
    'personality_summary': personality_summary.to_dict(),
    'clinical_summary': clinical_summary.to_dict(),
    'mechanistic_summary': mechanistic_summary.to_dict(),
    'analysis_timestamp': datetime.now().isoformat()
}

# Save summary statistics
with open(output_dir / "descriptive_statistics_summary.json", 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"\nAnalysis complete. Results saved to: {output_dir}")
print("Publication-ready figures and data exported successfully!") 