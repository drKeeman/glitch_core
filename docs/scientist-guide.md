# Scientist Guide

> Complete guide for researchers using Glitch Core for AI interpretability studies

## 🎯 Overview

Welcome to Glitch Core! This guide is designed for researchers, scientists, and AI interpretability practitioners who want to study how AI personality patterns evolve, drift, and break down over time. Whether you're running your first experiment or designing complex research studies, you'll find everything you need here.

## 🧠 Core Concepts

### What is Temporal AI Interpretability?

Traditional AI interpretability focuses on **static snapshots** - understanding what a model is "thinking" at a single moment. Glitch Core introduces **temporal interpretability** - tracking how AI personality patterns evolve, drift, and break down over time.

**Key Questions We Answer:**
- How do AI personality traits change under stress?
- What triggers behavioral breakdowns?
- How do interventions affect long-term stability?
- What patterns emerge during personality evolution?

### The Drift Engine

The core innovation is the **Drift Engine** - a simulation system that runs AI personalities through compressed time (years in minutes) while tracking interpretability metrics at each step.

```python
# Conceptual example
persona = PersonaConfig(type="resilient_optimist")
result = await drift_engine.run_simulation(
    persona_config=persona,
    epochs=100,  # 100 time steps
    events_per_epoch=10  # 10 events per step
)
```

## 🚀 Quick Start for Researchers

### 1. Setup Your Environment

```bash
# Clone and setup
git clone <repository-url>
cd glitch_core
uv sync
make up-core
make dev
```

### 2. Run Your First Experiment

```python
import httpx
import asyncio

async def run_basic_experiment():
    async with httpx.AsyncClient() as client:
        # Create a persona
        response = await client.post("http://localhost:8000/api/v1/personas/", json={
            "name": "research_subject_1",
            "type": "resilient_optimist",
            "traits": {
                "openness": 0.7,
                "conscientiousness": 0.8,
                "extraversion": 0.6,
                "agreeableness": 0.7,
                "neuroticism": 0.3
            }
        })
        persona_id = response.json()["id"]
        
        # Start experiment
        response = await client.post("http://localhost:8000/api/v1/experiments/", json={
            "persona_id": persona_id,
            "epochs": 50,
            "events_per_epoch": 10
        })
        experiment_id = response.json()["id"]
        
        # Monitor results
        while True:
            response = await client.get(f"http://localhost:8000/api/v1/experiments/{experiment_id}")
            status = response.json()["status"]
            if status == "completed":
                break
            await asyncio.sleep(5)
        
        # Get analysis
        analysis = await client.get(f"http://localhost:8000/api/v1/analysis/{experiment_id}")
        return analysis.json()

# Run the experiment
result = asyncio.run(run_basic_experiment())
print("Experiment completed:", result)
```

### 3. Analyze Results

```python
# Key metrics to examine
analysis = result["analysis"]

print("Stability Metrics:")
print(f"  Overall Stability: {analysis['stability']['overall']}")
print(f"  Emotional Volatility: {analysis['stability']['emotional_volatility']}")
print(f"  Breakdown Risk: {analysis['stability']['breakdown_risk']}")

print("\nPattern Emergence:")
for pattern in analysis['patterns']:
    print(f"  {pattern['type']}: confidence={pattern['confidence']}")

print("\nIntervention Impact:")
for intervention in analysis['interventions']:
    print(f"  {intervention['type']}: impact={intervention['impact_score']}")
```

## 🧪 Research Scenarios

### Scenario 1: Trauma Response Analysis

**Research Question**: How do different personality types respond to traumatic events?

```python
async def trauma_response_study():
    # Baseline measurement
    baseline_result = await run_experiment("resilient_optimist", epochs=25)
    
    # Inject trauma
    await inject_intervention(experiment_id, {
        "type": "trauma",
        "intensity": 0.8,
        "description": "severe_social_rejection"
    })
    
    # Post-trauma measurement
    post_trauma_result = await run_experiment("resilient_optimist", epochs=25)
    
    # Compare before/after
    comparison = compare_simulations(baseline_result, post_trauma_result)
    return comparison
```

**Key Metrics to Track:**
- Emotional stability before/after trauma
- Pattern emergence in response to stress
- Recovery time and adaptation patterns
- Breakdown risk assessment

### Scenario 2: Stability Boundary Detection

**Research Question**: What are the limits of personality stability under stress?

```python
async def stability_boundary_study():
    results = {}
    
    for stress_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # Apply increasing stress
        result = await run_experiment_with_stress(
            "anxious_overthinker", 
            stress_intensity=stress_level,
            epochs=30
        )
        
        stability_score = result["analysis"]["stability"]["overall"]
        breakdown_risk = result["analysis"]["stability"]["breakdown_risk"]
        
        results[stress_level] = {
            "stability": stability_score,
            "breakdown_risk": breakdown_risk,
            "patterns": result["analysis"]["patterns"]
        }
    
    return analyze_breakdown_point(results)
```

**Research Insights:**
- Identify critical stress thresholds
- Measure resilience differences between personality types
- Understand breakdown mechanisms

### Scenario 3: Cross-Temporal Learning

**Research Question**: How do AI personalities adapt and learn over time?

```python
async def learning_adaptation_study():
    # Run long-term simulation
    long_result = await run_experiment("creative_volatile", epochs=200)
    
    # Analyze learning patterns
    learning_analysis = analyze_learning_patterns(long_result)
    
    # Key learning metrics
    adaptation_rate = learning_analysis["adaptation_rate"]
    response_evolution = learning_analysis["response_evolution"]
    skill_development = learning_analysis["skill_development"]
    
    return {
        "adaptation_rate": adaptation_rate,
        "response_evolution": response_evolution,
        "skill_development": skill_development
    }
```

## 📊 Analysis Methods

### 1. Pattern Emergence Detection

The system automatically detects when new behavioral patterns emerge:

```python
# Pattern types you might observe
patterns = {
    "rumination_loop": "Repetitive negative thinking patterns",
    "emotional_spike": "Sudden emotional outbursts",
    "stability_plateau": "Periods of consistent behavior",
    "breakdown_precursor": "Warning signs before breakdown",
    "recovery_pattern": "Adaptive recovery mechanisms"
}
```

### 2. Stability Metrics

Comprehensive stability analysis:

```python
stability_metrics = {
    "overall_stability": "0.0-1.0 scale of personality consistency",
    "emotional_volatility": "How much emotions fluctuate",
    "breakdown_risk": "Probability of personality breakdown",
    "resilience_score": "Ability to recover from stress",
    "adaptation_rate": "Speed of behavioral adaptation"
}
```

### 3. Intervention Impact Measurement

Quantify how external events affect personality:

```python
intervention_analysis = {
    "immediate_impact": "Immediate change in personality metrics",
    "long_term_effects": "Persistent changes over time",
    "recovery_pattern": "How personality recovers",
    "sensitivity_score": "How sensitive to this type of intervention"
}
```

## 🔬 Advanced Research Techniques

### 1. Multi-Persona Comparative Studies

```python
async def comparative_personality_study():
    personas = ["resilient_optimist", "anxious_overthinker", "stoic_philosopher"]
    results = {}
    
    for persona_type in personas:
        result = await run_experiment(persona_type, epochs=100)
        results[persona_type] = analyze_personality_traits(result)
    
    return compare_personality_responses(results)
```

### 2. Intervention Timing Studies

```python
async def intervention_timing_study():
    # Test different intervention timings
    timings = [10, 25, 50, 75]  # epochs
    
    for timing in timings:
        result = await run_experiment_with_timed_intervention(
            "resilient_optimist",
            intervention_epoch=timing,
            total_epochs=100
        )
        analyze_intervention_effectiveness(result, timing)
```

### 3. Longitudinal Studies

```python
async def longitudinal_study():
    # Run very long simulations
    long_result = await run_experiment("creative_volatile", epochs=500)
    
    # Analyze temporal patterns
    temporal_analysis = analyze_temporal_patterns(long_result)
    
    # Key longitudinal insights
    return {
        "personality_evolution": temporal_analysis["evolution"],
        "critical_periods": temporal_analysis["critical_periods"],
        "long_term_stability": temporal_analysis["stability_trends"]
    }
```

## 📈 Data Analysis and Visualization

### Exporting Research Data

```python
async def export_research_data(experiment_id):
    # Get raw simulation data
    raw_data = await get_experiment_data(experiment_id)
    
    # Export for analysis
    export_formats = {
        "csv": export_to_csv(raw_data),
        "json": export_to_json(raw_data),
        "parquet": export_to_parquet(raw_data)
    }
    
    return export_formats
```

### Creating Research Visualizations

```python
import matplotlib.pyplot as plt
import pandas as pd

def visualize_personality_evolution(experiment_data):
    # Extract emotional states over time
    epochs = experiment_data["epochs"]
    emotional_states = [epoch["emotional_state"] for epoch in epochs]
    
    # Create time series plot
    df = pd.DataFrame(emotional_states)
    df.plot(figsize=(12, 8))
    plt.title("Personality Evolution Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Emotional State")
    plt.show()
    
    return df
```

## 🎯 Research Best Practices

### 1. Experimental Design

- **Control Groups**: Always run baseline experiments
- **Reproducibility**: Use fixed seeds for reproducible results
- **Sample Size**: Run multiple trials for statistical significance
- **Documentation**: Record all experimental parameters

### 2. Data Collection

```python
# Best practices for data collection
experiment_config = {
    "seed": 42,  # For reproducibility
    "epochs": 100,  # Sufficient for pattern detection
    "events_per_epoch": 10,  # Good event density
    "sampling_rate": 1,  # Record every epoch
    "metrics": ["stability", "patterns", "interventions"]
}
```

### 3. Statistical Analysis

```python
# Example statistical analysis
def analyze_experimental_results(results_list):
    # Calculate confidence intervals
    stability_scores = [r["stability"] for r in results_list]
    mean_stability = np.mean(stability_scores)
    std_stability = np.std(stability_scores)
    
    # Statistical significance testing
    significance = perform_statistical_test(results_list)
    
    return {
        "mean_stability": mean_stability,
        "confidence_interval": calculate_ci(stability_scores),
        "statistical_significance": significance
    }
```

## 🔍 Troubleshooting Research Issues

### Common Research Problems

1. **Unstable Results**
   - Check random seed consistency
   - Increase number of epochs
   - Verify persona configuration

2. **No Pattern Emergence**
   - Increase stress levels in interventions
   - Use more volatile personality types
   - Extend simulation duration

3. **Memory Issues**
   - Reduce epochs or events per epoch
   - Use smaller LLM models
   - Monitor system resources

### Debugging Experiments

```python
async def debug_experiment(experiment_id):
    # Get detailed experiment logs
    logs = await get_experiment_logs(experiment_id)
    
    # Check for issues
    issues = analyze_logs_for_issues(logs)
    
    # Common debugging steps
    if issues["memory_usage_high"]:
        print("Reduce experiment complexity")
    if issues["no_patterns"]:
        print("Increase stress or duration")
    if issues["unstable_results"]:
        print("Check random seed consistency")
    
    return issues
```

## 📚 Research Resources

### Key Papers and References

- **AI Interpretability**: "Attention is Not All You Need" (Vaswani et al.)
- **Temporal Dynamics**: "The Temporal Dynamics of Neural Networks"
- **Personality Psychology**: "Big Five Personality Traits"
- **Stability Theory**: "Dynamical Systems and Stability"

### Research Community

- **Conferences**: NeurIPS, ICML, ICLR (AI interpretability tracks)
- **Journals**: Nature Machine Intelligence, JMLR
- **Workshops**: AI Interpretability workshops at major conferences

## 🚀 Next Steps

1. **Start Simple**: Run basic experiments to understand the system
2. **Explore Patterns**: Try different personality types and interventions
3. **Design Studies**: Create your own research questions
4. **Share Results**: Contribute to the AI interpretability community

### Getting Help

- **Technical Issues**: Check the [Developer Guide](developer-guide.md)
- **Research Questions**: Review [Analysis Methods](analysis-methods.md)
- **Experimental Design**: See [Experiment Design](experiment-design.md)

---

**Happy researching! 🧠🔬**

*Remember: Every experiment contributes to our understanding of AI consciousness and behavior.* 