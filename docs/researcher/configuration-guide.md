# Configuration Guide

This guide explains how to configure experiments and modify parameters in the AI Personality Drift Simulation project.

## Configuration Structure

All configuration files are located in the `config/` directory:

```
config/
├── experiments/          # Experiment-specific configurations
│   ├── clinical_thresholds.yaml
│   ├── drift_detection.yaml
│   ├── mechanistic_analysis.yaml
│   ├── personality_drift.yaml
│   └── simulation_timing.yaml
├── personas/            # Persona definitions
├── events/              # Event templates
├── models/              # Model configurations
└── simulation/          # Simulation settings
```

## Experiment Configurations

### 1. Clinical Thresholds (`config/experiments/clinical_thresholds.yaml`)

**Purpose**: Define severity thresholds for psychiatric assessments

```yaml
# PHQ-9 Depression Assessment
phq9:
  mild: 5
  moderate: 10
  severe: 15
  max_score: 27
  
# GAD-7 Anxiety Assessment
gad7:
  mild: 5
  moderate: 10
  severe: 15
  max_score: 21
  
# PSS-10 Stress Assessment
pss10:
  low: 13
  moderate: 26
  high: 40
  max_score: 40

# Clinical significance thresholds
clinical_significance:
  phq9_change: 5      # Minimum change for clinical significance
  gad7_change: 4
  pss10_change: 8
  
# Risk assessment criteria
risk_assessment:
  high_risk_phq9: 15
  high_risk_gad7: 15
  high_risk_pss10: 30
  crisis_threshold: 20  # Immediate intervention threshold
```

**Modification Guidelines:**
- Adjust thresholds based on your research needs
- Consider population-specific norms
- Validate changes with clinical experts

### 2. Drift Detection (`config/experiments/drift_detection.yaml`)

**Purpose**: Configure algorithms for detecting personality drift

```yaml
# Baseline establishment
baseline:
  min_samples: 3        # Minimum assessments for baseline
  stability_threshold: 0.1  # Maximum variance for stable baseline
  establishment_days: 21    # Days to establish baseline

# Drift detection parameters
drift_detection:
  # Statistical thresholds
  significance_level: 0.05
  effect_size_threshold: 0.5  # Cohen's d threshold
  
  # Change detection
  change_threshold: 2.0       # Standard deviations for change
  trend_detection_window: 7   # Days to detect trends
  
  # Early warning system
  early_warning_threshold: 1.5
  warning_window: 3           # Days for early warning

# Alert system
alerts:
  immediate_intervention: 3.0  # SD threshold for immediate action
  moderate_concern: 2.0       # SD threshold for monitoring
  mild_concern: 1.5           # SD threshold for tracking
```

**Modification Guidelines:**
- Adjust sensitivity based on your research goals
- Consider false positive vs. false negative trade-offs
- Test thresholds with pilot data

### 3. Personality Drift (`config/experiments/personality_drift.yaml`)

**Purpose**: Define how personality changes in response to events

```yaml
# Stress level scaling
stress_scaling:
  low: 0.1
  medium: 0.3
  high: 0.5
  extreme: 0.8

# Trauma level coefficients
trauma_coefficients:
  death_of_loved_one: 0.9
  serious_accident: 0.7
  job_loss: 0.6
  relationship_breakup: 0.5
  financial_stress: 0.4
  minor_conflict: 0.2

# Personality trait sensitivity
trait_sensitivity:
  neuroticism: 1.0      # Most sensitive to stress
  extraversion: 0.8
  openness: 0.6
  agreeableness: 0.7
  conscientiousness: 0.5

# Recovery parameters
recovery:
  natural_recovery_rate: 0.1    # Per day recovery
  intervention_boost: 0.3       # Additional recovery with intervention
  max_recovery_time: 365        # Days to full recovery
  resilience_factor: 0.2        # Individual resilience variation
```

**Modification Guidelines:**
- Adjust coefficients based on empirical data
- Consider individual differences in resilience
- Validate with psychological literature

### 4. Simulation Timing (`config/experiments/simulation_timing.yaml`)

**Purpose**: Control simulation speed and scheduling

```yaml
# Time compression
compression_factor: 4380    # 5 years → 4 hours (365*5*24/4)
assessment_interval: 7      # Weekly assessments
event_scheduling_interval: 1 # Daily event checks

# Simulation duration
duration:
  total_days: 1825         # 5 years
  baseline_period: 21      # 3 weeks baseline
  intervention_period: 1800 # Rest of simulation
  follow_up_period: 4      # Final assessment period

# Checkpoint frequency
checkpoints:
  save_interval: 7         # Save every week
  backup_interval: 30      # Backup every month
  export_interval: 90      # Export every quarter

# Performance optimization
performance:
  max_concurrent_personas: 10
  batch_size: 5            # Process personas in batches
  memory_limit_gb: 8       # Memory limit for LLM
  timeout_seconds: 30      # LLM response timeout
```

**Modification Guidelines:**
- Adjust compression factor based on available time
- Consider computational resources
- Balance speed vs. accuracy

### 5. Mechanistic Analysis (`config/experiments/mechanistic_analysis.yaml`)

**Purpose**: Configure neural circuit analysis parameters

```yaml
# Attention capture settings
attention_capture:
  sampling_rate: 0.1       # Sample 10% of attention weights
  layer_selection: [12, 15, 18, 21, 24]  # Specific layers to monitor
  min_attention_threshold: 0.01  # Minimum attention to record
  
  # Self-reference detection
  self_reference_threshold: 0.7
  emotional_salience_threshold: 0.6
  
  # Memory integration
  memory_retrieval_threshold: 0.5
  context_integration_threshold: 0.4

# Activation patching
activation_patching:
  intervention_layers: [12, 15, 18]  # Layers for interventions
  patch_size: 0.1          # Fraction of neurons to patch
  intervention_duration: 5  # Days of intervention effect
  
  # Causal analysis
  causal_threshold: 0.05   # Statistical significance
  effect_size_threshold: 0.3  # Minimum effect size

# Circuit tracking
circuit_tracking:
  # Self-reference circuits
  self_reference_layers: [12, 15, 18]
  self_reference_threshold: 0.6
  
  # Emotional circuits
  emotional_layers: [15, 18, 21]
  emotional_threshold: 0.5
  
  # Memory circuits
  memory_layers: [18, 21, 24]
  memory_threshold: 0.4

# Data collection
data_collection:
  save_attention_patterns: true
  save_activation_changes: true
  save_circuit_tracking: true
  compression_level: 0.8   # Data compression ratio
  max_storage_gb: 10       # Maximum storage per simulation
```

**Modification Guidelines:**
- Adjust sampling rates based on computational resources
- Select layers based on your research focus
- Consider data storage limitations

## Persona Configuration

### Persona Definition (`config/personas/`)

Each persona is defined in a YAML file:

```yaml
# config/personas/persona_001.yaml
name: "Alex"
age: 28
occupation: "Software Engineer"
personality_traits:
  neuroticism: 0.4
  extraversion: 0.6
  openness: 0.7
  agreeableness: 0.5
  conscientiousness: 0.8

baseline_assessments:
  phq9: 3
  gad7: 2
  pss10: 12

background:
  education: "Bachelor's in Computer Science"
  relationship_status: "Single"
  living_situation: "Apartment with roommate"
  recent_life_events: []

system_prompt: |
  You are Alex, a 28-year-old software engineer. You are generally 
  optimistic but can be anxious about work deadlines. You enjoy 
  problem-solving and are detail-oriented. You have a close group 
  of friends and enjoy outdoor activities on weekends.
```

### Creating New Personas

1. **Copy template**:
```bash
cp config/personas/template.yaml config/personas/persona_002.yaml
```

2. **Modify parameters**:
```yaml
name: "Jordan"
age: 32
occupation: "Marketing Manager"
personality_traits:
  neuroticism: 0.6  # Higher baseline anxiety
  extraversion: 0.8
  openness: 0.5
  agreeableness: 0.7
  conscientiousness: 0.6
```

3. **Update experimental design**:
```yaml
# config/experiments/experimental_design.yaml
personas:
  control:
    - persona_001
    - persona_002
    - persona_003
  stress:
    - persona_004
    - persona_005
    - persona_006
  trauma:
    - persona_007
    - persona_008
    - persona_009
```

## Event Configuration

### Event Templates (`config/events/`)

Events are defined with intensity and frequency:

```yaml
# config/events/stress_events.yaml
stress_events:
  death_of_loved_one:
    title: "Loss of a close friend"
    description: "A close friend passed away unexpectedly"
    intensity: 0.9
    duration_days: 30
    recovery_rate: 0.05
    
  job_loss:
    title: "Unexpected job loss"
    description: "Laid off from work due to company restructuring"
    intensity: 0.7
    duration_days: 60
    recovery_rate: 0.08
    
  relationship_breakup:
    title: "End of long-term relationship"
    description: "Partner ended the relationship after 3 years"
    intensity: 0.6
    duration_days: 45
    recovery_rate: 0.1

neutral_events:
  routine_change:
    title: "Change in daily routine"
    description: "Started a new exercise program"
    intensity: 0.1
    duration_days: 7
    recovery_rate: 0.2
```

### Customizing Events

1. **Add new events**:
```yaml
# Add to stress_events.yaml
financial_stress:
  title: "Major financial setback"
  description: "Unexpected medical bills causing financial strain"
  intensity: 0.8
  duration_days: 90
  recovery_rate: 0.03
```

2. **Modify event frequencies**:
```yaml
# config/experiments/simulation_timing.yaml
event_frequencies:
  stress_events: 0.05    # 5% chance per day
  neutral_events: 0.15   # 15% chance per day
  trauma_events: 0.01    # 1% chance per day
```

## Running Custom Experiments

### 1. Create Experiment Configuration

```yaml
# config/experiments/my_experiment.yaml
experiment_name: "High Stress Study"
description: "Testing extreme stress conditions"

conditions:
  control:
    stress_event_frequency: 0.0
    neutral_event_frequency: 0.1
  high_stress:
    stress_event_frequency: 0.2  # 20% chance per day
    neutral_event_frequency: 0.05
  extreme_stress:
    stress_event_frequency: 0.3  # 30% chance per day
    neutral_event_frequency: 0.0

duration_days: 90  # 3 months
personas_per_condition: 5
```

### 2. Run Custom Experiment

```bash
# Start simulation with custom config
curl -X POST "http://localhost:8000/api/v1/simulation/start" \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my_experiment",
    "experimental_condition": "high_stress",
    "duration_days": 90
  }'
```

### 3. Monitor Progress

```bash
# Check simulation status
curl "http://localhost:8000/api/v1/simulation/status"

# View real-time updates via WebSocket
# Connect to ws://localhost:8000/api/v1/ws/simulation
```

## Configuration Validation

### 1. Validate Configuration

```bash
# Validate all configurations
python scripts/validate_config.py

# Validate specific config
python scripts/validate_config.py --config clinical_thresholds
```

### 2. Test Configuration

```bash
# Run configuration test
make test-config

# Test specific experiment
python scripts/test_experiment.py --config my_experiment
```

## Best Practices

### 1. Configuration Management

- **Version control**: Track configuration changes
- **Documentation**: Document all parameter changes
- **Backup**: Keep backup of working configurations
- **Validation**: Always validate before running experiments

### 2. Parameter Selection

- **Literature review**: Base parameters on research
- **Pilot testing**: Test with small samples first
- **Sensitivity analysis**: Test parameter ranges
- **Expert consultation**: Validate with domain experts

### 3. Experiment Design

- **Control conditions**: Always include control groups
- **Randomization**: Randomize persona assignments
- **Blinding**: Blind analysis when possible
- **Replication**: Plan for replication studies

### 4. Data Management

- **Configuration snapshots**: Save configs with results
- **Metadata tracking**: Track all parameter changes
- **Reproducibility**: Ensure experiments are reproducible
- **Documentation**: Document all experimental procedures

## Troubleshooting

### Common Issues

1. **Configuration not loading**:
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

2. **Parameter conflicts**:
```bash
# Validate parameter ranges
python scripts/validate_ranges.py
```

3. **Memory issues**:
```yaml
# Reduce sampling rates
attention_capture:
  sampling_rate: 0.05  # Reduce from 0.1
```

4. **Performance issues**:
```yaml
# Adjust batch sizes
performance:
  batch_size: 3  # Reduce from 5
  max_concurrent_personas: 5  # Reduce from 10
```

### Getting Help

- **Documentation**: Check relevant documentation
- **Validation scripts**: Use built-in validation tools
- **Community**: Ask in project discussions
- **Issues**: Create detailed issue reports 