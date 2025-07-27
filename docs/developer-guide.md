# Developer Guide

> Complete guide for developers working on Glitch Core

## 🎯 Overview

This guide covers everything you need to know to develop, test, and contribute to Glitch Core. Whether you're setting up your environment for the first time or diving into the codebase, you'll find what you need here.

## 🚀 Quick Setup

### Prerequisites

- **Python 3.12+** - Required for modern type hints and performance
- **Docker & Docker Compose** - For running dependencies
- **uv** - Fast Python package manager (install via `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Git** - Version control

### One-Command Setup

```bash
# Clone and setup
git clone <repository-url>
cd glitch_core
uv sync
make up-core
make dev
```

### Verify Installation

```bash
# Check all services are running
make health

# Run tests
make test

# Check code quality
make lint
```

## 🏗️ Project Structure

```
glitch_core/
├── src/glitch_core/           # Main application code
│   ├── api/                   # FastAPI endpoints
│   │   ├── v1/               # API version 1
│   │   │   ├── personas.py   # Persona management
│   │   │   ├── experiments.py # Experiment endpoints
│   │   │   ├── analysis.py   # Analysis results
│   │   │   └── interventions.py # Intervention injection
│   │   └── websocket.py      # WebSocket handlers
│   ├── core/                  # Core engine components
│   │   ├── drift_engine/     # Main simulation orchestrator
│   │   ├── personality/      # Psychology-grounded models
│   │   ├── memory/          # Vector + temporal storage
│   │   ├── llm/             # Ollama integration
│   │   └── analysis/        # Interpretability algorithms
│   ├── config/              # Configuration management
│   └── utils/               # Shared utilities
├── tests/                   # Test suite
├── docs/                   # Documentation
├── docker-compose.yml      # Service orchestration
├── Dockerfile              # Application container
├── pyproject.toml         # Project configuration
└── Makefile               # Development commands
```

## 🧠 Core Components Deep Dive

### 1. Drift Engine (`src/glitch_core/core/drift_engine/`)

The heart of the system that orchestrates personality evolution.

```python
# Key classes
class DriftEngine:
    """Orchestrates personality evolution over compressed time"""
    
    async def run_simulation(
        self,
        persona_config: PersonaConfig,
        drift_profile: DriftProfile,
        epochs: int = 100,
        events_per_epoch: int = 10
    ) -> SimulationResult:
        """Run a complete personality drift simulation"""
        pass

class SimulationResult:
    """Contains all simulation data and metrics"""
    epochs: List[EpochData]
    patterns: List[Pattern]
    stability_metrics: StabilityMetrics
    interventions: List[Intervention]
```

**Key Features:**
- Compressed time simulation (years in minutes)
- Reproducible runs with seeded randomness
- Real-time WebSocket updates
- Intervention injection mid-simulation

### 2. Personality System (`src/glitch_core/core/personality/`)

Psychology-grounded personality modeling with evolution rules.

```python
class PersonaConfig:
    """Defines base personality traits + evolution biases"""
    traits: Dict[str, float]  # Big 5 + clinical psychology
    cognitive_biases: Dict[str, float]
    emotional_baselines: Dict[str, float]
    memory_patterns: MemoryBias

class DriftProfile:
    """Defines HOW personality evolves over time"""
    evolution_rules: List[EvolutionRule]
    stability_metrics: StabilityConfig
    breakdown_conditions: List[BreakdownCondition]
```

**Implemented Profiles:**
- `resilient_optimist`: High stability, joy amplification
- `anxious_overthinker`: Rumination feedback loops
- `stoic_philosopher`: Ultra-high stability, emotional dampening
- `creative_volatile`: High creativity, emotional dysregulation

### 3. Memory Layer (`src/glitch_core/core/memory/`)

Temporal memory with personality-specific encoding.

```python
class MemoryManager:
    """Qdrant for semantic similarity + Redis for active context"""
    
    async def save_memory(
        self,
        content: str,
        emotional_weight: float,
        persona_bias: Dict[str, float]
    ) -> MemoryRecord
    
    async def retrieve_contextual(
        self,
        query: str,
        emotional_state: Dict[str, float],
        limit: int = 5
    ) -> List[MemoryRecord]
```

**Key Features:**
- Persona-specific memory biases
- Emotional weighting influences retrieval
- Temporal decay simulation
- Memory consolidation patterns

### 4. LLM Integration (`src/glitch_core/core/llm/`)

Local inference for reflection generation.

```python
class ReflectionEngine:
    """Ollama HTTP integration with persona-specific prompts"""
    
    async def generate_reflection(
        self,
        trigger_event: str,
        emotional_state: Dict[str, float],
        memories: List[str],
        persona_prompt: str
    ) -> str
```

**Performance Requirements:**
- Sub-3 second reflection generation
- Graceful fallback if Ollama unavailable
- Token usage tracking

### 5. Analysis Engine (`src/glitch_core/core/analysis/`)

Interpretability algorithms for temporal patterns.

```python
class TemporalAnalyzer:
    """Core interpretability research algorithms"""
    
    def analyze_drift_patterns(self, simulation: SimulationResult) -> Analysis:
        return {
            "emergence_points": self.detect_new_patterns(),
            "stability_boundaries": self.find_breakdown_points(),
            "intervention_leverage": self.measure_input_sensitivity(),
            "attention_evolution": self.track_focus_drift()
        }
```

## 🔌 API Development

### Adding New Endpoints

1. **Create endpoint in appropriate router:**

```python
# src/glitch_core/api/v1/personas.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class CreatePersonaRequest(BaseModel):
    name: str
    traits: Dict[str, float]
    drift_profile: str

@router.post("/")
async def create_persona(request: CreatePersonaRequest):
    """Create a new persona configuration"""
    try:
        persona = await persona_service.create(request)
        return {"id": persona.id, "status": "created"}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

2. **Add to main router:**

```python
# src/glitch_core/api/v1/router.py
from .personas import router as personas_router

router.include_router(personas_router, prefix="/personas", tags=["personas"])
```

3. **Write tests:**

```python
# tests/api/test_personas.py
import pytest
from httpx import AsyncClient

async def test_create_persona(client: AsyncClient):
    response = await client.post("/api/v1/personas/", json={
        "name": "test_persona",
        "traits": {"openness": 0.5},
        "drift_profile": "resilient_optimist"
    })
    assert response.status_code == 200
    assert "id" in response.json()
```

### WebSocket Development

```python
# src/glitch_core/api/websocket.py
@router.websocket("/ws/experiments/{experiment_id}")
async def experiment_websocket(
    websocket: WebSocket,
    experiment_id: str
):
    await websocket.accept()
    try:
        while True:
            # Send real-time updates
            data = await get_experiment_update(experiment_id)
            await websocket.send_json(data)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/core/test_drift_engine.py

# Run with coverage
pytest --cov=src/glitch_core

# Run integration tests
pytest tests/integration/
```

### Writing Tests

```python
# tests/core/test_drift_engine.py
import pytest
from glitch_core.core.drift_engine import DriftEngine
from glitch_core.core.personality import PersonaConfig

@pytest.mark.asyncio
async def test_drift_engine_simulation():
    """Test basic simulation functionality"""
    engine = DriftEngine()
    persona = PersonaConfig(type="resilient_optimist")
    
    result = await engine.run_simulation(
        persona_config=persona,
        epochs=10,
        events_per_epoch=5
    )
    
    assert len(result.epochs) == 10
    assert result.stability_metrics.overall_stability > 0
```

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── core/              # Core engine tests
│   ├── api/               # API endpoint tests
│   └── utils/             # Utility function tests
├── integration/            # Integration tests
│   ├── test_full_simulation.py
│   └── test_api_workflows.py
├── fixtures/              # Test data and fixtures
└── conftest.py           # Pytest configuration
```

## 🔧 Development Workflow

### Daily Development

```bash
# Start development environment
make dev

# In another terminal, run tests
make test

# Check code quality
make lint
make format

# Commit changes
git add .
git commit -m "feat: add new personality trait analysis"
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# All quality checks
make quality
```

### Database Operations

```bash
# Access Qdrant shell
make qdrant-shell

# Access Redis CLI
make redis-cli

# Reset databases
make reset-db
```

## 🐛 Debugging

### Common Issues

1. **Ollama not responding:**
   ```bash
   # Check Ollama status
   curl http://localhost:11434/api/tags
   
   # Pull model if needed
   make pull-model
   ```

2. **Qdrant connection issues:**
   ```bash
   # Check Qdrant health
   curl http://localhost:6333/health
   
   # Restart services
   make restart
   ```

3. **Memory issues:**
   ```bash
   # Check memory usage
   docker stats
   
   # Clean up
   make clean
   ```

### Logging

```python
# Enable debug logging
import structlog
logger = structlog.get_logger()

logger.info("Starting simulation", experiment_id=exp_id)
logger.debug("Processing epoch", epoch=epoch_num, events=len(events))
logger.error("Simulation failed", error=str(e))
```

### Performance Profiling

```python
# Add timing to critical sections
import time

start = time.time()
result = await engine.run_simulation(...)
duration = time.time() - start
logger.info("Simulation completed", duration=duration)
```

## 🚀 Deployment

### Local Production

```bash
# Build production image
make build

# Start production stack
make up

# Check health
make health
```

### Environment Configuration

```bash
# Copy example config
cp .env.example .env

# Edit configuration
vim .env
```

Key environment variables:
- `ENV`: Environment (development/production)
- `LOG_LEVEL`: Logging verbosity
- `QDRANT_URL`: Vector database URL
- `REDIS_URL`: Cache database URL
- `OLLAMA_URL`: LLM service URL

## 📚 Additional Resources

- **[API Reference](api-reference.md)** - Complete API documentation
- **[Architecture Deep Dive](architecture.md)** - System design details
- **[Testing Guide](testing-guide.md)** - Comprehensive testing strategies
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Make your changes** following the coding standards
4. **Add tests** for new functionality
5. **Run quality checks:** `make quality`
6. **Submit a pull request** with clear description

### Code Standards

- **Type hints** required for all functions
- **Docstrings** for all public methods
- **Tests** for all new functionality
- **Black formatting** for consistent style
- **MyPy** for type checking

---

**Happy coding! 🚀** 