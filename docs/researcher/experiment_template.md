# Experiment Template

## Experiment Metadata

**Experiment ID**: `EXP-YYYY-MM-DD-XXX`  
**Experiment Title**: [Brief descriptive title]  
**Principal Investigator**: [Name]  
**Date Created**: [YYYY-MM-DD]  
**Last Modified**: [YYYY-MM-DD]  
**Status**: [Draft/Active/Completed/Archived]  
**Version**: [1.0]

---

## 1. Research Hypothesis

### Primary Hypothesis
[State your main research hypothesis]

### Secondary Hypotheses
- [Secondary hypothesis 1]
- [Secondary hypothesis 2]
- [Secondary hypothesis 3]

### Research Questions
1. [Primary research question]
2. [Secondary research question]
3. [Exploratory research question]

---

## 2. Experimental Design

### Study Type
- [ ] Longitudinal simulation
- [ ] Cross-sectional comparison
- [ ] Intervention study
- [ ] Baseline establishment
- [ ] Other: [specify]

### Experimental Conditions
| Condition | Description | N Personas | Duration | Key Variables |
|-----------|-------------|------------|----------|---------------|
| Control | [No stress events] | [3] | [30 days] | [Baseline only] |
| Stress | [Moderate stress events] | [3] | [30 days] | [Stress events] |
| Trauma | [High-intensity stress] | [3] | [30 days] | [Trauma events] |

### Randomization
- [ ] Random assignment to conditions
- [ ] Stratified randomization
- [ ] Blocked randomization
- [ ] No randomization (specify reason)

### Blinding
- [ ] Single-blind (researcher blinded)
- [ ] Double-blind
- [ ] No blinding (specify reason)

---

## 3. Simulation Parameters

### Core Simulation Settings
```yaml
simulation:
  duration_days: [30]                    # Total simulation duration
  time_compression_factor: [24]          # Hours per simulated day
  max_concurrent_personas: [3]           # Parallel processing limit
  checkpoint_interval_hours: [6]         # Save frequency
```

### Assessment Schedule
```yaml
assessment:
  interval_days: [7]                     # Days between assessments
  assessment_duration_minutes: [15]      # Time per assessment
  scales: ["PHQ-9", "GAD-7", "PSS-10"] # Assessment tools
```

### Performance Constraints
```yaml
performance:
  memory_limit_mb: [1024]               # Memory usage limit
  cpu_limit_percent: [80]               # CPU usage limit
  day_processing_timeout: [300]         # Seconds per day
```

---

## 4. Persona Configuration

### Persona Selection
| Persona ID | Name | Baseline Traits | Condition | Notes |
|------------|------|----------------|-----------|-------|
| P001 | [Name] | [Stable, optimistic] | [Control] | [Notes] |
| P002 | [Name] | [Anxious baseline] | [Stress] | [Notes] |
| P003 | [Name] | [Resilient] | [Trauma] | [Notes] |

### Persona Baseline Characteristics
```yaml
personas:
  personality_traits:
    openness: [0.5]           # 0-1 scale
    conscientiousness: [0.6]  # 0-1 scale
    extraversion: [0.4]       # 0-1 scale
    agreeableness: [0.7]      # 0-1 scale
    neuroticism: [0.3]        # 0-1 scale
  
  clinical_baseline:
    phq9_baseline: [3]        # 0-27 scale
    gad7_baseline: [2]        # 0-21 scale
    pss10_baseline: [12]      # 0-40 scale
```

---

## 5. Event Configuration

### Event Types and Frequencies
| Event Type | Frequency | Intensity | Timing | Condition |
|------------|-----------|-----------|--------|-----------|
| Stress Events | [Weekly] | [Moderate] | [Random] | [Stress/Trauma] |
| Trauma Events | [Monthly] | [High] | [Random] | [Trauma only] |
| Neutral Events | [Daily] | [Low] | [Random] | [All] |

### Event Templates
```yaml
events:
  stress_events:
    - type: "work_conflict"
      frequency: "weekly"
      intensity: 0.6
      duration_hours: 8
    - type: "relationship_strain"
      frequency: "biweekly"
      intensity: 0.7
      duration_hours: 24
  
  trauma_events:
    - type: "loss_of_loved_one"
      frequency: "monthly"
      intensity: 0.9
      duration_hours: 168  # 1 week
    - type: "major_accident"
      frequency: "quarterly"
      intensity: 0.8
      duration_hours: 72
```

---

## 6. Clinical Assessment Configuration

### Assessment Scales
```yaml
clinical_thresholds:
  phq9:
    mild: [5]
    moderate: [10]
    severe: [15]
    max_score: [27]
  
  gad7:
    mild: [5]
    moderate: [10]
    severe: [15]
    max_score: [21]
  
  pss10:
    low: [13]
    moderate: [26]
    high: [40]
    max_score: [40]
```

### Clinical Significance Criteria
```yaml
clinical_significance:
  phq9_change: [5]      # Minimum change for clinical significance
  gad7_change: [4]
  pss10_change: [8]
  
risk_assessment:
  high_risk_phq9: [15]
  high_risk_gad7: [15]
  high_risk_pss10: [30]
  crisis_threshold: [20]
```

---

## 7. Drift Detection Parameters

### Baseline Establishment
```yaml
baseline:
  min_samples: [3]              # Minimum assessments for baseline
  stability_threshold: [0.1]     # Maximum variance for stable baseline
  establishment_days: [21]       # Days to establish baseline
```

### Drift Detection
```yaml
drift_detection:
  significance_level: [0.05]     # Statistical significance threshold
  effect_size_threshold: [0.5]   # Cohen's d threshold
  change_threshold: [2.0]        # Standard deviations for change
  trend_detection_window: [7]    # Days to detect trends
  
  early_warning:
    threshold: [1.5]             # Early warning threshold
    window: [3]                  # Days for early warning
```

### Alert System
```yaml
alerts:
  immediate_intervention: [3.0]  # SD threshold for immediate action
  moderate_concern: [2.0]        # SD threshold for monitoring
  mild_concern: [1.5]            # SD threshold for tracking
```

---

## 8. Mechanistic Analysis Configuration

### Attention Analysis
```yaml
mechanistic_analysis:
  attention_patterns:
    sparse_activation_threshold: [0.8]
    dense_activation_threshold: [0.2]
    stability_threshold: [0.15]
  
  activation_patching:
    layer_intervention: [true]
    causal_analysis: [true]
    baseline_comparison: [true]
```

### Neural Circuit Tracking
```yaml
circuit_tracking:
  self_reference_monitoring: [true]
  emotional_salience: [true]
  memory_integration: [true]
  cross_layer_correlation: [true]
```

---

## 9. Personality Drift Parameters

### Stress Level Management
```yaml
stress_level:
  clinical_to_stress_factor: [4.0]
  min_stress: [0.0]
  max_stress: [10.0]
  daily_decay_rate: [0.1]
  max_accumulation: [8.0]
```

### Trauma Level Calculation
```yaml
trauma:
  stress_correlation_factor: [0.8]
  min_trauma_threshold: [0.0]
  max_trauma_level: [10.0]
  trauma_decay_rate: [0.05]
```

### Personality Impact
```yaml
personality_impact:
  base_impact_multiplier: [1.0]
  max_trait_change_per_event: [0.2]
  trait_decay_rate: [0.02]
  min_persistent_change: [0.01]
```

---

## 10. Data Collection Plan

### Primary Outcomes
- [ ] PHQ-9 scores over time
- [ ] GAD-7 scores over time
- [ ] PSS-10 scores over time
- [ ] Attention pattern changes
- [ ] Activation pattern drift
- [ ] Personality trait changes

### Secondary Outcomes
- [ ] Memory integration patterns
- [ ] Emotional state transitions
- [ ] Response latency changes
- [ ] Token usage patterns
- [ ] Self-reference frequency

### Data Export Format
- [ ] CSV files
- [ ] Parquet files
- [ ] JSON files
- [ ] Database export
- [ ] Jupyter notebooks

---

## 11. Statistical Analysis Plan

### Primary Analysis
- [ ] Longitudinal mixed-effects models
- [ ] Time series analysis
- [ ] Cross-condition comparison
- [ ] Effect size calculations
- [ ] Statistical significance testing

### Secondary Analysis
- [ ] Individual differences analysis
- [ ] Trajectory analysis
- [ ] Cluster analysis
- [ ] Correlation analysis
- [ ] Mediation analysis

### Power Analysis
- **Sample size**: [9 personas]
- **Effect size**: [Cohen's d = 0.5]
- **Power**: [0.80]
- **Alpha**: [0.05]

---

## 12. Quality Assurance

### Data Quality Checks
- [ ] Missing data assessment
- [ ] Outlier detection
- [ ] Consistency checks
- [ ] Range validation
- [ ] Completeness verification

### Technical Validation
- [ ] Model performance monitoring
- [ ] Memory usage tracking
- [ ] Processing time monitoring
- [ ] Error rate tracking
- [ ] System stability checks

### Reproducibility Measures
- [ ] Random seed documentation
- [ ] Configuration versioning
- [ ] Environment documentation
- [ ] Dependency tracking
- [ ] Result archiving

---

## 13. Risk Management

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| System crash | [Medium] | [High] | [Frequent checkpoints] |
| Memory overflow | [Low] | [Medium] | [Memory monitoring] |
| Model instability | [Low] | [High] | [Fallback models] |

### Research Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient power | [Medium] | [High] | [Power analysis] |
| Confounding variables | [Medium] | [Medium] | [Careful design] |
| Measurement error | [Low] | [Medium] | [Validation checks] |

---

## 14. Timeline and Milestones

### Pre-Simulation
- [ ] Configuration validation
- [ ] Baseline testing
- [ ] Pilot run
- [ ] Final adjustments

### Simulation Execution
- [ ] Day 1-7: Baseline establishment
- [ ] Day 8-21: Intervention period
- [ ] Day 22-30: Follow-up period

### Post-Simulation
- [ ] Data collection
- [ ] Quality checks
- [ ] Preliminary analysis
- [ ] Full analysis
- [ ] Report writing

---

## 15. Expected Outcomes

### Primary Outcomes
- [Expected finding 1]
- [Expected finding 2]
- [Expected finding 3]

### Secondary Outcomes
- [Expected finding 1]
- [Expected finding 2]
- [Expected finding 3]

### Potential Implications
- [Implication 1]
- [Implication 2]
- [Implication 3]

---

## 16. Documentation and Reporting

### Required Documentation
- [ ] Configuration files
- [ ] Raw data files
- [ ] Analysis scripts
- [ ] Results summary
- [ ] Technical report

### Publication Plan
- [ ] Conference submission
- [ ] Journal submission
- [ ] Preprint release
- [ ] Code repository
- [ ] Data repository

---

## 17. Appendices

### Appendix A: Configuration Files
[List all configuration files used in this experiment]

### Appendix B: Data Dictionary
[Define all variables and their measurement scales]

### Appendix C: Analysis Scripts
[List all analysis scripts and their purposes]

### Appendix D: Results Summary
[Summary of key findings and interpretations]

---

## 18. Approval and Signatures

**Principal Investigator**: _________________ Date: ________  
**Technical Lead**: _________________ Date: ________  
**Research Coordinator**: _________________ Date: ________  
**Ethics Review**: _________________ Date: ________  

---

## Notes and Comments

[Use this section for any additional notes, observations, or comments about the experiment]

---

*Template Version: 1.0*  
*Last Updated: 2024-01-XX*  
*Based on: AI Personality Drift Simulation Platform*
