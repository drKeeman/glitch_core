# User Guides

## Overview

This document provides comprehensive guides for different user personas of Glitch Core, from researchers conducting psychological studies to developers building AI systems.

## For Researchers

### Getting Started

1. **Installation**
   ```bash
   git clone https://github.com/your-org/glitch-core.git
   cd glitch-core
   pip install -e .
   ```

2. **Quick Start**
   ```python
   from glitch_core import get_drift_engine
   
   # Get the drift engine
   engine = get_drift_engine()
   
   # Run a basic simulation
   result = await engine.run_simulation(
       persona_config={"type": "resilient_optimist"},
       drift_profile={"type": "gradual_evolution"},
       epochs=100
   )
   ```

### Research Workflows

#### 1. Personality Evolution Study

```python
from glitch_core import get_drift_engine
from glitch_core.core.types import PersonaConfig, DriftProfile

# Define personality configuration
persona_config = PersonaConfig({
    "type": "anxious_overthinker",
    "traits": {
        "openness": 0.6,
        "conscientiousness": 0.7,
        "extraversion": 0.3,
        "agreeableness": 0.5,
        "neuroticism": 0.8
    },
    "emotional_baselines": {
        "joy": 0.3,
        "sadness": 0.4,
        "anger": 0.2,
        "anxiety": 0.6
    }
})

# Define drift profile
drift_profile = DriftProfile({
    "type": "stress_induced_drift",
    "evolution_rules": [
        {"trait": "neuroticism", "direction": "increase", "rate": 0.01},
        {"trait": "conscientiousness", "direction": "decrease", "rate": 0.005}
    ],
    "stability_thresholds": {
        "breakdown_risk": 0.7,
        "recovery_threshold": 0.3
    }
})

# Run simulation
engine = get_drift_engine()
result = await engine.run_simulation(
    persona_config=persona_config,
    drift_profile=drift_profile,
    epochs=200,
    events_per_epoch=15
)

# Analyze results
print(f"Patterns detected: {len(result.pattern_emergence)}")
print(f"Stability warnings: {len(result.stability_warnings)}")
print(f"Interventions applied: {len(result.interventions)}")
```

#### 2. Intervention Effectiveness Study

```python
from glitch_core import get_drift_engine
from glitch_core.core.analysis import InterventionAnalyzer

# Run baseline simulation
baseline_result = await engine.run_simulation(
    persona_config={"type": "anxious_overthinker"},
    drift_profile={"type": "stress_induced_drift"},
    epochs=100
)

# Apply intervention
intervention = {
    "type": "cognitive_behavioral_therapy",
    "intensity": 0.8,
    "target_traits": ["neuroticism", "anxiety"],
    "epoch": 50
}

await engine.inject_intervention("experiment_1", intervention)

# Run intervention simulation
intervention_result = await engine.run_simulation(
    persona_config={"type": "anxious_overthinker"},
    drift_profile={"type": "stress_induced_drift"},
    epochs=100
)

# Analyze intervention effectiveness
analyzer = InterventionAnalyzer()
effectiveness = analyzer.compare_simulations(
    baseline_result, intervention_result
)

print(f"Stability improvement: {effectiveness.stability_improvement:.3f}")
print(f"Volatility reduction: {effectiveness.volatility_reduction:.3f}")
```

#### 3. Memory and Learning Study

```python
from glitch_core import get_memory_manager

memory_manager = get_memory_manager()

# Store memories with emotional context
await memory_manager.save_memory(
    content="Received positive feedback at work",
    emotional_weight=0.8,
    persona_bias={"extraversion": 0.7, "neuroticism": 0.3},
    memory_type="event",
    context={"source": "work", "significance": "high"}
)

# Retrieve contextual memories
memories = await memory_manager.retrieve_contextual(
    query="work performance",
    emotional_state={"joy": 0.6, "anxiety": 0.2},
    limit=5
)

for memory in memories:
    print(f"Memory: {memory.content}")
    print(f"Emotional weight: {memory.emotional_weight}")
    print(f"Relevance score: {memory.relevance_score}")
```

### Data Analysis

#### 1. Statistical Analysis

```python
import pandas as pd
import numpy as np
from glitch_core.core.analysis import StatisticalAnalyzer

# Convert simulation results to DataFrame
df = pd.DataFrame([
    {
        "epoch": i,
        "openness": epoch.traits["openness"],
        "conscientiousness": epoch.traits["conscientiousness"],
        "extraversion": epoch.traits["extraversion"],
        "agreeableness": epoch.traits["agreeableness"],
        "neuroticism": epoch.traits["neuroticism"],
        "stability": epoch.stability_metrics.overall_stability,
        "emotional_volatility": epoch.stability_metrics.emotional_volatility
    }
    for i, epoch in enumerate(result.epochs)
])

# Calculate correlations
correlations = df.corr()
print("Trait correlations:")
print(correlations["stability"])

# Trend analysis
analyzer = StatisticalAnalyzer()
trends = analyzer.analyze_temporal_trends(result)

for trait, trend in trends.trait_trends.items():
    print(f"{trait}: slope={trend['slope']:.4f}, p={trend['p_value']:.4f}")
```

#### 2. Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot personality evolution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "stability"]

for i, trait in enumerate(traits):
    row, col = i // 3, i % 3
    axes[row, col].plot(df["epoch"], df[trait])
    axes[row, col].set_title(f"{trait.title()} Evolution")
    axes[row, col].set_xlabel("Epoch")
    axes[row, col].set_ylabel("Value")

plt.tight_layout()
plt.show()

# Plot emotional states over time
emotional_df = pd.DataFrame([
    {
        "epoch": i,
        "joy": epoch.emotional_state["joy"],
        "sadness": epoch.emotional_state["sadness"],
        "anger": epoch.emotional_state["anger"],
        "anxiety": epoch.emotional_state["anxiety"]
    }
    for i, epoch in enumerate(result.epochs)
])

plt.figure(figsize=(12, 6))
for emotion in ["joy", "sadness", "anger", "anxiety"]:
    plt.plot(emotional_df["epoch"], emotional_df[emotion], label=emotion)

plt.title("Emotional State Evolution")
plt.xlabel("Epoch")
plt.ylabel("Emotional Intensity")
plt.legend()
plt.show()
```

### Export and Reporting

```python
from glitch_core.core.export import SimulationExporter

exporter = SimulationExporter()

# Export to JSON
json_data = exporter.to_json(result, include_metadata=True)
with open("simulation_results.json", "w") as f:
    json.dump(json_data, f, indent=2)

# Export to CSV
csv_data = exporter.to_csv(result)
csv_data.to_csv("simulation_results.csv", index=False)

# Generate research report
report = exporter.generate_report(
    result,
    title="Personality Evolution Study",
    author="Dr. Researcher",
    methodology="Longitudinal simulation with intervention"
)
with open("research_report.md", "w") as f:
    f.write(report)
```

## For Developers

### API Integration

#### 1. REST API Usage

```python
import requests

# Start simulation
response = requests.post("http://localhost:8000/api/v1/experiments", json={
    "persona_config": {
        "type": "resilient_optimist",
        "traits": {"openness": 0.7, "conscientiousness": 0.8}
    },
    "drift_profile": {
        "type": "gradual_evolution"
    },
    "epochs": 100,
    "events_per_epoch": 10
})

experiment_id = response.json()["experiment_id"]

# Get simulation status
status_response = requests.get(f"http://localhost:8000/api/v1/experiments/{experiment_id}")
status = status_response.json()

# Get results
results_response = requests.get(f"http://localhost:8000/api/v1/experiments/{experiment_id}/results")
results = results_response.json()
```

#### 2. WebSocket Integration

```python
import asyncio
import websockets
import json

async def monitor_simulation(experiment_id: str):
    uri = f"ws://localhost:8000/ws/experiments/{experiment_id}"
    
    async with websockets.connect(uri) as websocket:
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "epoch_completed":
                    print(f"Epoch {data['epoch']} completed")
                    print(f"Stability: {data['stability_metrics']['overall_stability']:.3f}")
                
                elif data["type"] == "pattern_emerged":
                    print(f"Pattern emerged: {data['pattern_type']}")
                    print(f"Confidence: {data['confidence']:.3f}")
                
                elif data["type"] == "stability_warning":
                    print(f"Stability warning: {data['risk_level']}")
                    print(f"Stability score: {data['stability_score']:.3f}")
                    
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed")
                break

# Monitor simulation
asyncio.run(monitor_simulation("experiment_123"))
```

### Custom Extensions

#### 1. Custom Personality Types

```python
from glitch_core.core.personality import PersonalityConfig

class CreativeGenius(PersonalityConfig):
    """Custom personality type for creative genius"""
    
    def __init__(self):
        super().__init__()
        self.traits = {
            "openness": 0.9,        # Very high openness
            "conscientiousness": 0.4, # Lower conscientiousness
            "extraversion": 0.6,     # Moderate extraversion
            "agreeableness": 0.5,    # Moderate agreeableness
            "neuroticism": 0.7       # Higher neuroticism
        }
        
        self.cognitive_biases = {
            "creative_thinking": 0.8,
            "risk_taking": 0.7,
            "non_conformity": 0.8
        }
        
        self.emotional_baselines = {
            "joy": 0.6,
            "sadness": 0.3,
            "anger": 0.4,
            "anxiety": 0.5,
            "excitement": 0.7
        }
    
    def apply_creative_bias(self, event: Event) -> Event:
        """Apply creative thinking bias to events"""
        if event.type == "creative_opportunity":
            event.impact["openness"] *= 1.5
            event.impact["excitement"] *= 1.3
        return event
```

#### 2. Custom Drift Profiles

```python
from glitch_core.core.drift_engine import DriftProfile

class TraumaInducedDrift(DriftProfile):
    """Drift profile for trauma-induced personality changes"""
    
    def __init__(self, trauma_intensity: float = 0.8):
        super().__init__()
        self.trauma_intensity = trauma_intensity
        self.evolution_rules = [
            {
                "trigger": "trauma_event",
                "trait_changes": {
                    "neuroticism": 0.2,
                    "openness": -0.1,
                    "extraversion": -0.15
                },
                "emotional_changes": {
                    "anxiety": 0.3,
                    "sadness": 0.2,
                    "anger": 0.1
                }
            }
        ]
    
    def apply_trauma_effects(self, emotional_state: Dict[str, float]) -> Dict[str, float]:
        """Apply trauma-specific effects"""
        if self.trauma_intensity > 0.7:
            emotional_state["anxiety"] *= 1.5
            emotional_state["sadness"] *= 1.3
            emotional_state["joy"] *= 0.7
        
        return emotional_state
```

#### 3. Custom Analysis Algorithms

```python
from glitch_core.core.analysis import BaseAnalyzer

class ClinicalAnalyzer(BaseAnalyzer):
    """Clinical psychology analysis algorithms"""
    
    def detect_mental_health_patterns(self, simulation_result) -> List[ClinicalPattern]:
        """Detect patterns indicative of mental health conditions"""
        patterns = []
        
        for epoch in simulation_result.epochs:
            # Check for depression indicators
            if (epoch.emotional_state["sadness"] > 0.7 and 
                epoch.emotional_state["joy"] < 0.3):
                patterns.append(ClinicalPattern(
                    type="depression_risk",
                    confidence=epoch.emotional_state["sadness"],
                    epoch=epoch.epoch,
                    severity="moderate"
                ))
            
            # Check for anxiety indicators
            if epoch.emotional_state["anxiety"] > 0.8:
                patterns.append(ClinicalPattern(
                    type="anxiety_risk",
                    confidence=epoch.emotional_state["anxiety"],
                    epoch=epoch.epoch,
                    severity="high"
                ))
        
        return patterns
    
    def calculate_clinical_metrics(self, simulation_result) -> ClinicalMetrics:
        """Calculate clinical psychology metrics"""
        return ClinicalMetrics(
            depression_risk=self.calculate_depression_risk(simulation_result),
            anxiety_risk=self.calculate_anxiety_risk(simulation_result),
            stability_score=self.calculate_stability_score(simulation_result),
            recovery_potential=self.calculate_recovery_potential(simulation_result)
        )
```

### Testing and Validation

#### 1. Unit Tests

```python
import pytest
from glitch_core import get_drift_engine

@pytest.mark.asyncio
async def test_personality_evolution():
    """Test basic personality evolution"""
    engine = get_drift_engine()
    
    result = await engine.run_simulation(
        persona_config={"type": "resilient_optimist"},
        drift_profile={"type": "gradual_evolution"},
        epochs=10
    )
    
    assert len(result.epochs) == 10
    assert result.epochs[0].traits["openness"] > 0
    assert result.epochs[-1].traits["openness"] > 0

@pytest.mark.asyncio
async def test_intervention_effectiveness():
    """Test intervention application"""
    engine = get_drift_engine()
    
    # Run baseline
    baseline = await engine.run_simulation(
        persona_config={"type": "anxious_overthinker"},
        drift_profile={"type": "stress_induced_drift"},
        epochs=50
    )
    
    # Apply intervention
    intervention = {
        "type": "therapy",
        "intensity": 0.8,
        "target_traits": ["neuroticism"]
    }
    
    await engine.inject_intervention("test_exp", intervention)
    
    # Run with intervention
    intervention_result = await engine.run_simulation(
        persona_config={"type": "anxious_overthinker"},
        drift_profile={"type": "stress_induced_drift"},
        epochs=50
    )
    
    # Verify intervention had effect
    baseline_neuroticism = baseline.epochs[-1].traits["neuroticism"]
    intervention_neuroticism = intervention_result.epochs[-1].traits["neuroticism"]
    
    assert intervention_neuroticism < baseline_neuroticism
```

#### 2. Integration Tests

```python
@pytest.mark.asyncio
async def test_full_simulation_workflow():
    """Test complete simulation workflow"""
    from glitch_core import get_drift_engine, get_memory_manager
    
    engine = get_drift_engine()
    memory_manager = get_memory_manager()
    
    # Run simulation
    result = await engine.run_simulation(
        persona_config={"type": "resilient_optimist"},
        drift_profile={"type": "gradual_evolution"},
        epochs=20
    )
    
    # Verify memory storage
    memories = await memory_manager.retrieve_contextual(
        query="positive experience",
        emotional_state={"joy": 0.6},
        limit=5
    )
    
    assert len(memories) > 0
    
    # Verify pattern detection
    assert len(result.pattern_emergence) >= 0
    
    # Verify stability metrics
    assert result.epochs[-1].stability_metrics.overall_stability > 0
```

### Performance Optimization

#### 1. Caching Strategies

```python
from functools import lru_cache
from glitch_core.core.caching import SimulationCache

class OptimizedDriftEngine:
    """Optimized drift engine with caching"""
    
    def __init__(self):
        self.cache = SimulationCache()
    
    @lru_cache(maxsize=1000)
    def calculate_emotional_impact(self, event_type: str, personality_traits: tuple) -> float:
        """Cache emotional impact calculations"""
        # Convert dict to tuple for caching
        traits_dict = dict(zip(self.trait_names, personality_traits))
        return self._calculate_impact(event_type, traits_dict)
    
    async def run_simulation_with_caching(self, config: dict) -> SimulationResult:
        """Run simulation with intelligent caching"""
        cache_key = self._generate_cache_key(config)
        
        # Check cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Run simulation
        result = await self.run_simulation(config)
        
        # Cache result
        await self.cache.set(cache_key, result)
        
        return result
```

#### 2. Parallel Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelDriftEngine:
    """Drift engine with parallel processing"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def run_parallel_simulations(self, configs: List[dict]) -> List[SimulationResult]:
        """Run multiple simulations in parallel"""
        
        async def run_single_simulation(config):
            return await self.run_simulation(config)
        
        # Run simulations in parallel
        tasks = [run_single_simulation(config) for config in configs]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def analyze_parallel_results(self, results: List[SimulationResult]) -> AnalysisResult:
        """Analyze multiple simulation results in parallel"""
        
        async def analyze_single_result(result):
            return await self.analyze_simulation(result)
        
        # Analyze results in parallel
        tasks = [analyze_single_result(result) for result in results]
        analyses = await asyncio.gather(*tasks)
        
        return self.combine_analyses(analyses)
```

## Troubleshooting

### Common Issues

1. **Memory Connection Errors**
   ```python
   # Check Qdrant connection
   from qdrant_client import QdrantClient
   client = QdrantClient("http://localhost:6333")
   collections = client.get_collections()
   print(f"Available collections: {collections}")
   ```

2. **LLM Connection Issues**
   ```python
   # Test Ollama connection
   import httpx
   response = httpx.get("http://localhost:11434/api/tags")
   print(f"Ollama status: {response.status_code}")
   ```

3. **Performance Issues**
   ```python
   # Monitor memory usage
   import psutil
   process = psutil.Process()
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

### Debug Mode

```python
import logging
from glitch_core.config.logging import setup_logging

# Enable debug logging
setup_logging("DEBUG")

# Run simulation with debug output
result = await engine.run_simulation(
    persona_config={"type": "resilient_optimist"},
    drift_profile={"type": "gradual_evolution"},
    epochs=10,
    debug=True
)
```

## Best Practices

1. **Configuration Management**: Use environment variables for sensitive settings
2. **Error Handling**: Always handle exceptions gracefully
3. **Logging**: Use structured logging for debugging
4. **Testing**: Write comprehensive tests for custom extensions
5. **Documentation**: Document custom personality types and drift profiles
6. **Performance**: Monitor resource usage and optimize bottlenecks
7. **Security**: Validate all inputs and sanitize data
8. **Backup**: Regularly backup simulation data and configurations 