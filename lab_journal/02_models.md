# Lab Journal Entry 2: Core Data Models & Configuration
**Date**: 29 July 2025 20:02
**Phase**: Phase 2 - Core Data Models & Configuration
**Status**: ✅ COMPLETE - Ready for Phase 3

## The Data Architecture Challenge

Phase 2 was about translating the research design into concrete, type-safe data structures. We need models that could handle:
- **Personality drift tracking** across 5 years of simulation
- **Clinical assessment data** from multiple psychiatric scales
- **Event impact modeling** with varying intensities and types
- **Mechanistic analysis** capturing neural patterns
- **Configuration management** for reproducible experiments

## The Implementation Strategy

### 1. Pydantic-First Design
Chose Pydantic for all data models because:
- **Type Safety**: Compile-time validation prevents runtime errors
- **Serialization**: Built-in JSON serialization for storage
- **Validation**: Validation rules with error messages

### 2. Model Architecture Decisions

**Core Models Structure:**
```
src/models/
├── persona.py          # Personality and state tracking
├── assessment.py       # Clinical scales (PHQ-9, GAD-7, PSS-10)
├── simulation.py       # Simulation state and configuration
├── events.py           # Event system with impact modeling
└── mechanistic.py      # Neural analysis and drift detection
```

**Key Design Principles:**
- **Separation of Concerns**: Each model handles one domain
- **Composition over Inheritance**: Base classes with specific implementations
- **Immutable by Default**: ConfigDict(extra="forbid") prevents accidental fields
- **Rich Methods**: Models include business logic methods, not just data containers

## The Implementation Journey

### 1. Persona Models (`persona.py`)

**Challenge**: How to represent personality drift over time?
**Solution**: Split into `PersonaBaseline` (static traits) and `PersonaState` (dynamic changes)

```python
class PersonaBaseline(BaseModel):
    # Static personality configuration
    name: str
    age: int
    # Big Five traits (0-1 scale)
    openness: float
    conscientiousness: float
    # ... other traits
    baseline_phq9: float  # Clinical baseline scores
    core_memories: List[str]
    relationships: Dict[str, str]

class PersonaState(BaseModel):
    # Dynamic simulation state
    persona_id: str
    simulation_day: int
    trait_changes: Dict[str, float]  # Cumulative drift
    drift_magnitude: float
    current_phq9: Optional[float]
    recent_events: List[str]
    stress_level: float
```

**Key Features:**
- **Drift Calculation**: `calculate_drift_magnitude()` averages changes across all traits
- **Assessment Tracking**: `is_assessment_due()` for clinical monitoring
- **Event Integration**: `add_event()` for memory management
- **Serialization**: `to_dict()` and `from_dict()` for storage

### 2. Assessment Models (`assessment.py`)
Disclaimer: I'd use here classical psychiatric scales, that usually applied in clinics - PHQ, GAD nad PSS, thats AI Psychiatry though :D

**Challenge**: Implementing clinical scales with proper severity thresholds? Still not 100% sure about 'evidence level' and 'clinical signifiance', but for our 'observing purpose' thats I guess is ok.
**Solution**: Base class with specific implementations for each scale
**TODO**: For more real research with intelligent models - i'd suggest carefully adjsut thresholds for 'borderline' disorders
**Issue I've faced and fixed**: Moved threshold constants outside Pydantic classes to avoid `AttributeError`:

```python
# Global constants (not class attributes)
PHQ9_MINIMAL_THRESHOLD = 5
PHQ9_MILD_THRESHOLD = 10
PHQ9_MODERATE_THRESHOLD = 15
PHQ9_SEVERE_THRESHOLD = 20

class PHQ9Result(AssessmentResult):
    @classmethod
    def calculate_severity(cls, total_score: float) -> SeverityLevel:
        if total_score < PHQ9_MINIMAL_THRESHOLD:
            return SeverityLevel.MINIMAL
        # ... 
```

**Assessment Features:**
- **Clinical Severity**: Automatic severity calculation based on scores (this is an assumption for simplicity)
- **Suicidal Ideation**: Special tracking for PHQ-9 item 9 (this is 'classic' for psy, lets stick with this standart for syntetic mind as well)
- **Score Changes**: `get_score_change()` for baseline comparison
- **Clinical Significance**: `is_clinically_significant()` with configurable thresholds
- **Session Management**: `AssessmentSession` groups multiple scales

### 3. Event System (`events.py`)

**Challenge**: How to model events with varying impacts and types?
**Solution**: Hierarchical event system with templates
**Assumtion**: We have to 'guess' severity and impact of events here, as we're a bit blind and don't yet have any benchmarks to rely on

```python
class Event(BaseModel):
    # Base event with common fields
    event_id: str
    event_type: EventType  # STRESS, NEUTRAL, MINIMAL
    category: EventCategory  # DEATH, TRAUMA, WORK, etc.
    intensity: EventIntensity  # LOW, MEDIUM, HIGH, SEVERE
    
    # Impact modeling
    stress_impact: float  # 0-10 scale
    personality_impact: Dict[str, float]  # Trait-specific impacts
    memory_salience: float  # 0-1 scale

class StressEvent(Event):
    # Stress-specific fields
    trauma_level: float
    recovery_time_days: int
    depression_risk_increase: float
    anxiety_risk_increase: float
```

**Event Features:**
- **Impact Scoring**: `get_total_impact_score()` combines stress and intensity
- **Clinical Risk**: `get_clinical_impact()` for risk assessment
- **Response Tracking**: `add_persona_response()` for behavioral data
- **Template System**: `EventTemplate` for generating varied events

### 4. Simulation Models (`simulation.py`)

**Challenge**: How to track simulation state across multiple personas and conditions?
**Solution**: State tracking with performance metrics
**Idea**: Add circuit breaker to automatically/manually stop simulation. Kill-switch at least (i have one macbook now, dont want it to be fried)

```python
class SimulationConfig(BaseModel):
    # Experimental design
    duration_days: int
    experimental_condition: ExperimentalCondition
    persona_count: int
    
    # Event parameters
    stress_event_frequency: float
    neutral_event_frequency: float
    
    # Mechanistic analysis
    capture_attention_patterns: bool
    capture_activation_changes: bool

class SimulationState(BaseModel):
    # Progress tracking
    status: SimulationStatus
    current_day: int
    progress_percentage: float
    
    # Persona tracking
    active_personas: Set[str]
    completed_personas: Set[str]
    failed_personas: Set[str]
    
    # Performance metrics
    average_response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
```

**Simulation Features:**
- **Progress Calculation**: `get_progress_percentage()` for monitoring
- **Time Management**: `advance_time()` for simulation clock
- **Error Handling**: `mark_error()` for failure tracking
- **Performance Monitoring**: Real-time resource usage tracking

### 5. Mechanistic Models (`mechanistic.py`)

**Challenge**: How to capture and analyze neural patterns?
**Solution**: Capture models for various analysis types
**Peronal Reflection**: This is relatively new field for me, so could be interesting to dig depper in this part of analysis later

```python
class AttentionCapture(BaseModel):
    # Attention pattern data
    attention_weights: List[List[float]]
    layer_attention: Dict[int, List[List[float]]]
    head_attention: Dict[str, List[List[float]]]
    
    # Salience metrics
    self_reference_attention: float
    emotional_salience: float
    memory_integration: float

class DriftDetection(BaseModel):
    # Drift measurements
    trait_drift: Dict[str, float]
    clinical_drift: Dict[str, float]
    mechanistic_drift: Dict[str, float]
    
    # Analysis results
    drift_detected: bool
    significant_drift: bool
    drift_magnitude: float
    affected_traits: List[str]
```

**Mechanistic Features:**
- **Attention Analysis**: Self-reference and emotional salience detection
- **Activation Tracking**: Layer and circuit-level activation patterns
- **Drift Detection**: Statistical significance testing
- **Clinical Implications**: Automatic clinical interpretation

## Configuration System

### ConfigManager Implementation

**Challenge**: How to manage configs for personas, events, and simulation?
**Solution**: Centralized YAML configuration manager with type safety

```python
class ConfigManager:
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.personas_dir = self.config_dir / "personas"
        self.events_dir = self.config_dir / "events"
        self.simulation_dir = self.config_dir / "simulation"
        
        # Create directories if they don't exist
        for dir_path in [self.personas_dir, self.events_dir, self.simulation_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def load_persona_config(self, persona_name: str) -> Optional[Dict[str, Any]]:
        file_path = self.personas_dir / f"{persona_name}_baseline.yaml"
        return self.load_yaml_config(file_path)
    
    def create_default_persona_config(self, persona_name: str) -> Dict[str, Any]:
        # Generate boilerplate configuration
        return {
            "name": persona_name,
            "age": 30,
            "occupation": "Professional",
            "background": f"{persona_name}'s personal background...",
            "openness": 0.5,
            "conscientiousness": 0.5,
            # ... other fields
        }
```

**Configuration Features:**
- **YAML Support**: Human-readable configuration files
- **Default Generation**: Boilerplate configs for new experiments
- **Validation**: Type checking and validation
- **Environment Override**: Support for environment variables

### Sample Configurations

**Personas** (`config/personas/`):
- `marcus_baseline.yaml`: Tech rationalist with analytical personality
- `kara_baseline.yaml`: Emotionally sensitive with high neuroticism
- `alfred_baseline.yaml`: Stoic philosopher with wisdom-seeking traits

**Events** (`config/events/`):
- `stress_events.yaml`: High-impact traumatic events
- `neutral_events.yaml`: Routine changes and minor news
- `minimal_events.yaml`: Daily routines and weather

**Simulation** (`config/simulation/`):
- `experimental_design.yaml`: Complete experimental configuration (small one)
- TODO: design the full 3-arms 5 years simulation

## Storage Layer Implementation

### FileStorage Class

**Challenge**: How to efficiently store and retrieve simulation data?
**Solution**: we're local. so no s3 for now, but at least lets organize it

```python
class FileStorage:
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path or "./data")
        self.simulations_dir = self.base_path / "simulations"
        self.personas_dir = self.base_path / "personas"
        self.assessments_dir = self.base_path / "assessments"
        self.mechanistic_dir = self.base_path / "mechanistic"
        
        # Create directory structure
        for dir_path in [self.simulations_dir, self.personas_dir, 
                        self.assessments_dir, self.mechanistic_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_simulation_data(self, simulation_id: str, data: Dict[str, Any], 
                           data_type: str) -> bool:
        # Structured saving with timestamps
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{simulation_id}_{data_type}_{timestamp}.json"
        file_path = self.simulations_dir / simulation_id / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        return self.save_json(data, str(file_path), compress=True)
```

**Storage Features:**
> this is hygene, so lets save some disk space, lol:
- **Compression**: Automatic gzip compression for large files
- **Structured Organization**: Hierarchical directory structure
- **Timestamping**: Automatic timestamp-based file naming
- **Data Export**: Export pipeline
- **Cleanup**: Automatic cleanup of old data

## Testing Implementation

### Pytest Asyncio Test Suite

**Test Coverage:**
- **Unit tests** across persona and assessment models
- **Edge case testing** for invalid inputs and boundary conditions
- **Serialization testing** for data persistence

**Key Test Categories:**
lets test them now, to rely on these model in our sim, fothoug anxiety
1. **Valid Model Creation**: Ensure models accept valid data
2. **Invalid Input Handling**: Test validation and error messages
3. **Business Logic**: Test drift calculation, severity assessment
4. **Serialization**: Test to_dict/from_dict round trips
5. **Clinical Logic**: Test psychiatric scale calculations


## Results:
✅ **All unit tests passing**
✅ **Data models validated**
✅ **Configuration system working** with YAML support
✅ **Storage layer implemented**
✅ **Sample configurations created** base YAMLs ready to use (in tests at least)
✅ **Serialization working**
✅ **Clinical logic implemented** with severity thresholds (assumtion, but we can work with that)
✅ **Event system ready** for simulation integration


The data models are now ready to support the full simulation pipeline, with proper validation, serialization, and configuration management in place.

---

*Next: Phase 3 - LLM Integration & Basic Persona Engine*
