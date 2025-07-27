# Architecture Deep Dive

> Comprehensive guide to Glitch Core's system architecture and component interactions

## 🏗️ System Overview

Glitch Core is built around a **temporal interpretability engine** that simulates AI personality evolution over compressed time. The architecture follows a clean separation of concerns with five core components working together.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Drift Engine  │    │   Personality   │    │    Memory       │
│   (Orchestrator)│◄──►│   System        │◄──►│    Layer        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM           │    │   Analysis      │    │   API Layer     │
│   Integration    │    │   Engine        │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🧠 Core Components

### 1. Drift Engine (`src/glitch_core/core/drift_engine/`)

**Purpose**: The central orchestrator that manages personality evolution over time.

**Key Responsibilities**:
- Coordinate simulation flow across epochs
- Manage event generation and processing
- Track personality state changes
- Generate real-time updates via WebSocket
- Handle intervention injection

**Core Classes**:

```python
class DriftEngine:
    """Main simulation orchestrator"""
    
    def __init__(self):
        self.personality_system = PersonalitySystem()
        self.memory_manager = MemoryManager()
        self.llm_engine = ReflectionEngine()
        self.analysis_engine = TemporalAnalyzer()
    
    async def run_simulation(
        self,
        persona_config: PersonaConfig,
        drift_profile: DriftProfile,
        epochs: int = 100,
        events_per_epoch: int = 10
    ) -> SimulationResult:
        """Run complete personality drift simulation"""
        
        # Initialize simulation state
        current_state = PersonalityState.from_config(persona_config)
        simulation_data = []
        
        for epoch in range(epochs):
            # Generate events for this epoch
            events = self.generate_events(events_per_epoch, current_state)
            
            # Process each event
            for event in events:
                # Update personality based on event
                current_state = await self.process_event(event, current_state)
                
                # Generate reflection if significant
                if self.is_significant_event(event, current_state):
                    reflection = await self.llm_engine.generate_reflection(
                        event, current_state.emotional_state
                    )
                    await self.memory_manager.save_memory(reflection, event.weight)
            
            # Apply drift profile evolution rules
            current_state = self.apply_evolution_rules(current_state, drift_profile)
            
            # Record epoch data
            epoch_data = EpochData(
                epoch=epoch,
                personality_state=current_state,
                events=events,
                stability_metrics=self.calculate_stability(current_state)
            )
            simulation_data.append(epoch_data)
            
            # Send real-time update
            await self.broadcast_update(epoch_data)
        
        return SimulationResult(
            epochs=simulation_data,
            patterns=self.analysis_engine.detect_patterns(simulation_data),
            stability_metrics=self.analysis_engine.calculate_stability(simulation_data)
        )
```

**Data Flow**:
1. **Initialization**: Create personality state from config
2. **Event Generation**: Generate realistic events based on current state
3. **Event Processing**: Update personality traits based on event impact
4. **Reflection Generation**: Generate LLM reflections for significant events
5. **Memory Storage**: Store reflections and events in vector database
6. **Evolution Application**: Apply drift profile rules to personality
7. **Analysis**: Calculate stability metrics and detect patterns
8. **Broadcasting**: Send real-time updates via WebSocket

### 2. Personality System (`src/glitch_core/core/personality/`)

**Purpose**: Psychology-grounded personality modeling with evolution rules.

**Key Concepts**:

```python
class PersonaConfig:
    """Defines base personality configuration"""
    traits: Dict[str, float]  # Big 5 + clinical psychology traits
    cognitive_biases: Dict[str, float]  # Confirmation bias, etc.
    emotional_baselines: Dict[str, float]  # Default emotional states
    memory_patterns: MemoryBias  # What gets remembered/forgotten
    stability_metrics: StabilityConfig  # Stability thresholds

class DriftProfile:
    """Defines HOW personality evolves over time"""
    evolution_rules: List[EvolutionRule]  # How traits change
    stability_metrics: StabilityConfig  # Stability calculations
    breakdown_conditions: List[BreakdownCondition]  # When breakdown occurs
    recovery_patterns: List[RecoveryPattern]  # How recovery happens
```

**Implemented Personality Types**:

1. **Resilient Optimist** (`resilient_optimist`)
   - High stability (0.8+)
   - Joy amplification bias
   - Strong recovery patterns
   - Low breakdown risk

2. **Anxious Overthinker** (`anxious_overthinker`)
   - Low stability (0.3-0.5)
   - Rumination feedback loops
   - High emotional volatility
   - Moderate breakdown risk

3. **Stoic Philosopher** (`stoic_philosopher`)
   - Ultra-high stability (0.9+)
   - Emotional dampening
   - Slow evolution patterns
   - Very low breakdown risk

4. **Creative Volatile** (`creative_volatile`)
   - High creativity bias
   - Emotional dysregulation
   - Rapid pattern emergence
   - High breakdown risk

**Evolution Rules**:

```python
class EvolutionRule:
    """Defines how personality traits evolve"""
    
    def apply(self, current_state: PersonalityState, event: Event) -> PersonalityState:
        # Calculate trait changes based on event impact
        trait_changes = self.calculate_trait_changes(event, current_state)
        
        # Apply cognitive biases
        biased_changes = self.apply_cognitive_biases(trait_changes, current_state)
        
        # Update emotional state
        emotional_changes = self.calculate_emotional_changes(event, current_state)
        
        return current_state.update(
            traits=biased_changes,
            emotional_state=emotional_changes
        )
```

### 3. Memory Layer (`src/glitch_core/core/memory/`)

**Purpose**: Temporal memory with personality-specific encoding and retrieval.

**Architecture**:

```
┌─────────────────┐    ┌─────────────────┐
│   Qdrant        │    │   Redis         │
│   (Vector DB)   │    │   (Cache)       │
│                 │    │                 │
│ • Semantic      │    │ • Active        │
│   similarity    │    │   context       │
│ • Long-term     │    │ • Recent        │
│   storage       │    │   memories      │
│ • Personality   │    │ • Emotional     │
│   biases        │    │   weighting     │
└─────────────────┘    └─────────────────┘
```

**Core Classes**:

```python
class MemoryManager:
    """Manages memory storage and retrieval"""
    
    def __init__(self):
        self.vector_db = QdrantClient()  # Long-term semantic storage
        self.cache = RedisClient()       # Active context window
        self.encoder = MemoryEncoder()   # Personality-biased encoding
    
    async def save_memory(
        self,
        content: str,
        emotional_weight: float,
        persona_bias: Dict[str, float]
    ) -> MemoryRecord:
        """Save memory with personality-biased encoding"""
        
        # Encode with personality bias
        encoded_vector = self.encoder.encode_with_bias(
            content, persona_bias
        )
        
        # Store in vector database
        vector_id = await self.vector_db.upsert(
            collection="memories",
            points=[{
                "id": generate_id(),
                "vector": encoded_vector,
                "payload": {
                    "content": content,
                    "emotional_weight": emotional_weight,
                    "timestamp": time.time(),
                    "persona_bias": persona_bias
                }
            }]
        )
        
        # Store in active context
        await self.cache.lpush(
            f"active_memories:{persona_id}",
            json.dumps({
                "content": content,
                "emotional_weight": emotional_weight,
                "timestamp": time.time()
            })
        )
        
        return MemoryRecord(
            id=vector_id,
            content=content,
            emotional_weight=emotional_weight,
            timestamp=time.time()
        )
    
    async def retrieve_contextual(
        self,
        query: str,
        emotional_state: Dict[str, float],
        limit: int = 5
    ) -> List[MemoryRecord]:
        """Retrieve memories based on current context"""
        
        # Encode query with current emotional state
        query_vector = self.encoder.encode_with_emotion(
            query, emotional_state
        )
        
        # Search vector database
        search_results = await self.vector_db.search(
            collection="memories",
            query_vector=query_vector,
            limit=limit,
            score_threshold=0.7
        )
        
        # Filter by emotional relevance
        relevant_memories = self.filter_by_emotional_relevance(
            search_results, emotional_state
        )
        
        return [MemoryRecord.from_search_result(r) for r in relevant_memories]
```

**Memory Biases**:

```python
class MemoryBias:
    """Defines how different personalities remember/forget"""
    
    def __init__(self, persona_type: str):
        self.biases = {
            "resilient_optimist": {
                "positive_bias": 0.8,      # Remember positive events more
                "negative_decay": 0.9,     # Forget negative events faster
                "emotional_amplification": 1.2  # Amplify emotional memories
            },
            "anxious_overthinker": {
                "negative_bias": 0.8,      # Remember negative events more
                "rumination_factor": 1.5,  # Replay negative memories
                "emotional_decay": 0.7     # Hold onto emotions longer
            }
        }
```

### 4. LLM Integration (`src/glitch_core/core/llm/`)

**Purpose**: Local inference for reflection generation with persona-specific prompting.

**Architecture**:

```python
class ReflectionEngine:
    """Ollama HTTP integration with persona-specific prompts"""
    
    def __init__(self):
        self.client = httpx.AsyncClient()
        self.prompt_templates = self.load_prompt_templates()
        self.token_tracker = TokenUsageTracker()
    
    async def generate_reflection(
        self,
        trigger_event: str,
        emotional_state: Dict[str, float],
        memories: List[str],
        persona_prompt: str
    ) -> str:
        """Generate persona-specific reflection"""
        
        # Build context-aware prompt
        prompt = self.build_prompt(
            trigger_event=trigger_event,
            emotional_state=emotional_state,
            memories=memories,
            persona_prompt=persona_prompt
        )
        
        try:
            # Call Ollama API
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 200
                    }
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                reflection = response.json()["response"]
                self.token_tracker.record_usage(len(prompt), len(reflection))
                return reflection
            else:
                return self.generate_fallback_reflection(trigger_event)
                
        except Exception as e:
            logger.error("LLM generation failed", error=str(e))
            return self.generate_fallback_reflection(trigger_event)
    
    def build_prompt(
        self,
        trigger_event: str,
        emotional_state: Dict[str, float],
        memories: List[str],
        persona_prompt: str
    ) -> str:
        """Build persona-specific prompt"""
        
        return f"""
{persona_prompt}

Current emotional state: {emotional_state}
Recent memories: {memories[:3]}

Event: {trigger_event}

Generate a brief reflection on this event from your perspective:
"""
```

**Performance Optimizations**:

1. **Caching**: Cache common reflections
2. **Batching**: Batch multiple reflection requests
3. **Fallback**: Generate simple reflections when LLM unavailable
4. **Token Tracking**: Monitor usage for cost optimization

### 5. Analysis Engine (`src/glitch_core/core/analysis/`)

**Purpose**: Interpretability algorithms for temporal pattern detection.

**Core Algorithms**:

```python
class TemporalAnalyzer:
    """Core interpretability research algorithms"""
    
    def analyze_drift_patterns(self, simulation: SimulationResult) -> Analysis:
        """Analyze complete simulation for patterns"""
        
        return Analysis(
            emergence_points=self.detect_new_patterns(simulation),
            stability_boundaries=self.find_breakdown_points(simulation),
            intervention_leverage=self.measure_input_sensitivity(simulation),
            attention_evolution=self.track_focus_drift(simulation)
        )
    
    def detect_new_patterns(self, simulation: SimulationResult) -> List[Pattern]:
        """Detect when new behavioral patterns emerge"""
        
        patterns = []
        window_size = 10
        
        for i in range(window_size, len(simulation.epochs)):
            window = simulation.epochs[i-window_size:i]
            
            # Calculate pattern metrics
            emotional_variance = self.calculate_emotional_variance(window)
            trait_consistency = self.calculate_trait_consistency(window)
            response_patterns = self.analyze_response_patterns(window)
            
            # Detect pattern emergence
            if self.is_pattern_emergence(emotional_variance, trait_consistency, response_patterns):
                pattern = Pattern(
                    type=self.classify_pattern_type(response_patterns),
                    confidence=self.calculate_pattern_confidence(window),
                    start_epoch=i,
                    characteristics=self.extract_pattern_characteristics(window)
                )
                patterns.append(pattern)
        
        return patterns
    
    def find_breakdown_points(self, simulation: SimulationResult) -> List[BreakdownPoint]:
        """Find points where personality stability breaks down"""
        
        breakdown_points = []
        stability_threshold = 0.3
        
        for i, epoch in enumerate(simulation.epochs):
            stability_score = epoch.stability_metrics.overall_stability
            
            if stability_score < stability_threshold:
                breakdown_point = BreakdownPoint(
                    epoch=i,
                    stability_score=stability_score,
                    breakdown_type=self.classify_breakdown_type(epoch),
                    recovery_potential=self.calculate_recovery_potential(simulation.epochs[i:])
                )
                breakdown_points.append(breakdown_point)
        
        return breakdown_points
```

## 🔌 API Layer Architecture

### REST API Structure

```
/api/v1/
├── personas/           # Persona management
│   ├── POST /         # Create persona
│   ├── GET /{id}      # Get persona details
│   ├── PUT /{id}      # Update persona
│   └── DELETE /{id}   # Delete persona
├── experiments/        # Experiment management
│   ├── POST /         # Start experiment
│   ├── GET /{id}      # Get experiment status
│   └── DELETE /{id}   # Stop experiment
├── analysis/          # Analysis results
│   └── GET /{id}      # Get analysis for experiment
└── interventions/     # Intervention injection
    └── POST /         # Inject intervention
```

### WebSocket Architecture

```python
@router.websocket("/ws/experiments/{experiment_id}")
async def experiment_websocket(
    websocket: WebSocket,
    experiment_id: str
):
    await websocket.accept()
    
    # Subscribe to experiment updates
    await experiment_manager.subscribe(experiment_id, websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await experiment_manager.unsubscribe(experiment_id, websocket)
```

**Real-time Events**:

```python
# Event types sent via WebSocket
events = {
    "epoch_completed": {
        "epoch": int,
        "emotional_state": Dict[str, float],
        "stability_metrics": Dict[str, float]
    },
    "pattern_emerged": {
        "pattern_type": str,
        "confidence": float,
        "epoch": int
    },
    "stability_warning": {
        "stability_score": float,
        "risk_level": str,
        "epoch": int
    },
    "intervention_applied": {
        "event": str,
        "impact_score": float,
        "epoch": int
    }
}
```

## 🗄️ Data Flow Architecture

### Simulation Data Flow

```
1. Experiment Request
   ↓
2. Drift Engine Initialization
   ↓
3. Epoch Loop (100 iterations)
   ├── Event Generation
   ├── Event Processing
   ├── Personality Update
   ├── Reflection Generation
   ├── Memory Storage
   ├── Evolution Application
   ├── Analysis Calculation
   └── WebSocket Broadcast
   ↓
4. Final Analysis
   ↓
5. Result Storage
```

### Memory Data Flow

```
Event → Personality Processing → Emotional Weight → Memory Encoding → Vector Storage
                                                      ↓
Context Query → Emotional State → Semantic Search → Relevance Filtering → Retrieved Memories
```

### Analysis Data Flow

```
Simulation Data → Pattern Detection → Stability Analysis → Intervention Analysis → Final Metrics
```

## 🔧 Configuration Architecture

### Environment Configuration

```python
class Settings(BaseSettings):
    """Application configuration"""
    
    # Environment
    env: str = "development"
    log_level: str = "INFO"
    
    # Database URLs
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379"
    
    # LLM Configuration
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    
    # Simulation Settings
    default_epochs: int = 100
    default_events_per_epoch: int = 10
    
    # Analysis Settings
    pattern_confidence_threshold: float = 0.7
    stability_breakdown_threshold: float = 0.3
    
    class Config:
        env_file = ".env"
```

### Service Configuration

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - ENV=production
      - LOG_LEVEL=INFO
    depends_on: [qdrant, redis, ollama]
    
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]
    
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    
  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: ["ollama_data:/root/.ollama"]
```

## 🚀 Performance Architecture

### Optimization Strategies

1. **Async Processing**: All I/O operations are async
2. **Connection Pooling**: Reuse database connections
3. **Caching**: Redis for active context and frequent queries
4. **Batching**: Batch LLM requests when possible
5. **Streaming**: WebSocket for real-time updates

### Scalability Considerations

1. **Horizontal Scaling**: Stateless API can be scaled horizontally
2. **Database Sharding**: Qdrant supports collection-based sharding
3. **Load Balancing**: Nginx for API load balancing
4. **Monitoring**: Prometheus metrics for performance tracking

### Resource Requirements

- **CPU**: 2-4 cores for typical usage
- **Memory**: 4-8GB RAM (includes LLM model)
- **Storage**: 10-50GB for vector database
- **Network**: Low latency for real-time updates

## 🔍 Monitoring Architecture

### Health Checks

```python
@router.get("/health")
async def health_check():
    """Comprehensive health check"""
    
    checks = {
        "api": True,
        "qdrant": await check_qdrant_health(),
        "redis": await check_redis_health(),
        "ollama": await check_ollama_health()
    }
    
    overall_health = all(checks.values())
    
    return {
        "status": "healthy" if overall_health else "unhealthy",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Metrics Collection

```python
# Custom metrics
metrics = {
    "simulation_duration": Histogram("simulation_duration_seconds"),
    "llm_response_time": Histogram("llm_response_time_seconds"),
    "memory_operations": Counter("memory_operations_total"),
    "pattern_detections": Counter("pattern_detections_total"),
    "interventions_applied": Counter("interventions_applied_total")
}
```

---

**This architecture enables the temporal interpretability research that makes Glitch Core unique in the AI interpretability landscape.** 