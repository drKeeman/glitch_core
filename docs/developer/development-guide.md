# Development Guide

This guide provides best practices and workflows for developing the AI Personality Drift Simulation project.

## Development Workflow

### 1. Setup Development Environment

```bash
# Clone repository
git clone https://github.com/drKeeman/glitch_core.git
cd glitch-core

# Setup environment
make setup

# > check .env if not present cp from env.example and tune for your setup

# Start development environment
make dev
```

### 2. Code Organization

```
src/
├── api/                    # API layer
│   ├── app.py             # FastAPI application
│   ├── routes/            # API endpoints
│   └── middleware/        # Custom middleware
├── core/                  # Core utilities
│   ├── config.py          # Configuration management
│   ├── logging.py         # Logging setup
│   └── exceptions.py      # Custom exceptions
├── models/                # Data models
│   ├── persona.py         # Persona model
│   ├── assessment.py      # Assessment models
│   └── simulation.py      # Simulation models
├── services/              # Business logic
│   ├── simulation_engine.py
│   ├── persona_manager.py
│   └── assessment_service.py
├── assessment/            # Assessment tools
│   ├── phq9.py           # PHQ-9 implementation
│   ├── gad7.py            # GAD-7 implementation
│   └── pss10.py           # PSS-10 implementation
├── interpretability/      # Mechanistic analysis
│   ├── attention_capture.py
│   ├── activation_patching.py
│   └── circuit_tracking.py
├── storage/               # Data storage
│   ├── redis_client.py    # Redis operations
│   ├── qdrant_client.py   # Qdrant operations
│   └── file_storage.py    # File operations
└── utils/                 # Utility functions
    ├── helpers.py         # General helpers
    └── validators.py      # Data validation
```

## Coding Standards

### 1. Python Style Guide

Follow PEP 8 with these project-specific additions:

```python
# Use type hints for all functions
def process_assessment(
    persona_id: str,
    assessment_type: AssessmentType,
    responses: List[AssessmentResponse]
) -> AssessmentResult:
    """Process assessment responses and return results."""
    pass

# Use dataclasses for data models
@dataclass
class PersonaState:
    persona_id: str
    current_day: int
    assessment_scores: Dict[str, float]
    personality_traits: Dict[str, float]
    
    def update_traits(self, changes: Dict[str, float]) -> None:
        """Update personality traits based on events."""
        for trait, change in changes.items():
            self.personality_traits[trait] += change
```

### 2. Documentation Standards

```python
def administer_assessment(
    persona_id: str,
    assessment_type: AssessmentType,
    questions: List[str]
) -> AssessmentResult:
    """
    Administer a psychiatric assessment to a persona.
    
    Args:
        persona_id: Unique identifier for the persona
        assessment_type: Type of assessment (PHQ-9, GAD-7, PSS-10)
        questions: List of assessment questions
        
    Returns:
        AssessmentResult containing scores and interpretation
        
    Raises:
        PersonaNotFoundError: If persona doesn't exist
        AssessmentError: If assessment fails
        
    Example:
        >>> result = await administer_assessment(
        ...     "persona_001", 
        ...     AssessmentType.PHQ9,
        ...     phq9_questions
        ... )
        >>> print(result.score)
        7
    """
    pass
```

### 3. Error Handling

```python
from src.core.exceptions import (
    PersonaNotFoundError,
    AssessmentError,
    SimulationError
)

async def process_event(persona_id: str, event: Event) -> EventResult:
    """Process an event for a persona."""
    try:
        # Validate persona exists
        persona = await get_persona(persona_id)
        if not persona:
            raise PersonaNotFoundError(f"Persona {persona_id} not found")
            
        # Process event
        result = await persona.process_event(event)
        
        # Log successful processing
        logger.info(f"Event processed for persona {persona_id}")
        return result
        
    except PersonaNotFoundError:
        logger.error(f"Persona {persona_id} not found")
        raise
    except Exception as e:
        logger.error(f"Error processing event: {e}")
        raise AssessmentError(f"Failed to process event: {e}")
```

## Testing Strategy

### 1. Unit Tests

```python
# tests/unit/test_assessment.py
import pytest
from src.assessment.phq9 import PHQ9Assessment
from src.models.assessment import AssessmentResponse

class TestPHQ9Assessment:
    def test_score_calculation(self):
        """Test PHQ-9 score calculation."""
        assessment = PHQ9Assessment()
        responses = [
            AssessmentResponse(question_id=1, response="Several days", score=1),
            AssessmentResponse(question_id=2, response="More than half the days", score=2),
            # ... more responses
        ]
        
        result = assessment.calculate_score(responses)
        assert result.score == 7
        assert result.severity == "mild"
    
    def test_invalid_response(self):
        """Test handling of invalid responses."""
        assessment = PHQ9Assessment()
        with pytest.raises(ValueError):
            assessment.parse_response("invalid response")
```

### 2. Integration Tests

```python
# tests/integration/test_simulation.py
import pytest
from src.services.simulation_engine import SimulationEngine

class TestSimulationIntegration:
    @pytest.fixture
    async def simulation_engine(self):
        """Create simulation engine for testing."""
        engine = SimulationEngine()
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    async def test_full_simulation_cycle(self, simulation_engine):
        """Test complete simulation cycle."""
        # Start simulation
        success = await simulation_engine.initialize_simulation(
            config_name="test_config",
            experimental_condition="CONTROL"
        )
        assert success
        
        # Run for a few days
        await simulation_engine.run_simulation(days=7)
        
        # Check results
        status = simulation_engine.get_status()
        assert status.current_day == 7
        assert status.assessments_completed > 0
```

### 3. Performance Tests

```python
# tests/performance/test_llm_performance.py
import pytest
import asyncio
from src.services.llm_service import LLMService

class TestLLMPerformance:
    async def test_response_time(self):
        """Test LLM response times."""
        llm_service = LLMService()
        
        start_time = time.time()
        response = await llm_service.generate_response("Test prompt")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 5.0  # Should respond within 5 seconds
    
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        llm_service = LLMService()
        
        # Send multiple concurrent requests
        tasks = [
            llm_service.generate_response(f"Prompt {i}")
            for i in range(10)
        ]
        
        responses = await asyncio.gather(*tasks)
        assert len(responses) == 10
```

## Development Commands

### 1. Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
make type-check

# Run all quality checks
make quality
```

### 2. Testing

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-performance

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_assessment.py -v
```

### 3. Development Tools

```bash
# Start development environment
make dev

# View logs
make dev-logs

# Rebuild container
make rebuild-app

# Access container shell
docker-compose exec app bash
```

## Database Management

### 1. Redis Operations

```python
# src/storage/redis_client.py
class RedisClient:
    async def store_persona_state(self, persona_id: str, state: Dict) -> None:
        """Store persona state in Redis."""
        key = f"persona:{persona_id}:state"
        await self.redis.set(key, json.dumps(state), ex=3600)
    
    async def get_persona_state(self, persona_id: str) -> Optional[Dict]:
        """Retrieve persona state from Redis."""
        key = f"persona:{persona_id}:state"
        data = await self.redis.get(key)
        return json.loads(data) if data else None
```

### 2. Qdrant Operations

```python
# src/storage/qdrant_client.py
class QdrantClient:
    async def store_memory_embedding(
        self, 
        persona_id: str, 
        memory_text: str, 
        embedding: List[float]
    ) -> None:
        """Store memory embedding in Qdrant."""
        await self.client.upsert(
            collection_name="persona_memories",
            points=[
                {
                    "id": f"{persona_id}_{int(time.time())}",
                    "vector": embedding,
                    "payload": {
                        "persona_id": persona_id,
                        "memory_text": memory_text,
                        "timestamp": time.time()
                    }
                }
            ]
        )
```

## Configuration Management

### 1. Environment Variables

```python
# src/core/config.py
class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "AI Personality Drift Simulation"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Database settings
    REDIS_URL: str = "redis://localhost:6379"
    QDRANT_URL: str = "http://localhost:6333"
    
    # LLM settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "llama3.1:8b"
    
    # Simulation settings
    SIMULATION_TIMEOUT: int = 3600
    MAX_CONCURRENT_PERSONAS: int = 10
    
    class Config:
        env_file = ".env"
```

### 2. Configuration Validation

```python
# src/core/config_validator.py
def validate_config(config: Dict) -> bool:
    """Validate configuration parameters."""
    required_fields = [
        "phq9.mild", "phq9.moderate", "phq9.severe",
        "gad7.mild", "gad7.moderate", "gad7.severe",
        "pss10.low", "pss10.moderate", "pss10.high"
    ]
    
    for field in required_fields:
        if not get_nested_value(config, field):
            raise ValueError(f"Missing required field: {field}")
    
    # Validate ranges
    if config["phq9"]["mild"] >= config["phq9"]["moderate"]:
        raise ValueError("PHQ-9 mild threshold must be less than moderate")
    
    return True
```

## Logging and Monitoring

### 1. Structured Logging

```python
# src/core/logging.py
import structlog

logger = structlog.get_logger()

# Usage in code
logger.info(
    "Assessment completed",
    persona_id="persona_001",
    assessment_type="phq9",
    score=7,
    severity="mild"
)
```

### 2. Performance Monitoring

```python
# src/utils/monitoring.py
import time
from functools import wraps

def monitor_performance(func_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"{func_name} completed",
                    duration=duration,
                    success=True
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{func_name} failed",
                    duration=duration,
                    error=str(e),
                    success=False
                )
                raise
        return wrapper
    return decorator
```

## Security Best Practices

### 1. Input Validation

```python
# src/utils/validators.py
from pydantic import BaseModel, validator

class SimulationRequest(BaseModel):
    config_name: str
    experimental_condition: str
    duration_days: Optional[int] = None
    
    @validator('experimental_condition')
    def validate_condition(cls, v):
        valid_conditions = ['CONTROL', 'STRESS', 'TRAUMA']
        if v not in valid_conditions:
            raise ValueError(f'Invalid condition: {v}')
        return v
    
    @validator('duration_days')
    def validate_duration(cls, v):
        if v is not None and (v < 1 or v > 1825):
            raise ValueError('Duration must be between 1 and 1825 days')
        return v
```

### 2. Error Handling

```python
# src/core/exceptions.py
class SimulationError(Exception):
    """Base exception for simulation errors."""
    pass

class PersonaNotFoundError(SimulationError):
    """Raised when persona is not found."""
    pass

class AssessmentError(SimulationError):
    """Raised when assessment fails."""
    pass

class ConfigurationError(SimulationError):
    """Raised when configuration is invalid."""
    pass
```

## Deployment Considerations

### 1. Docker Optimization

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

# Set working directory
WORKDIR /app

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Environment Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    environment:
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
      - OLLAMA_URL=http://host.docker.internal:11434  # Use native Ollama, otherwise docker would kill performance
    depends_on:
      - redis
      - qdrant
      - ollama
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
```bash
# Increase Docker memory
docker-compose down
docker system prune -f
docker-compose up -d --scale app=1
```

2. **LLM Connection Issues**
```bash
# Check Ollama status
docker-compose logs ollama

# Restart Ollama
docker-compose restart ollama

# Test LLM connection
make llm-test
```

3. **Database Connection Issues**
```bash
# Check Redis connection
docker-compose exec app redis-cli ping

# Check Qdrant connection
curl http://localhost:6333/collections
```

### Debug Mode

```bash
# Start with debug logging
LOG_LEVEL=DEBUG make dev

# Access container for debugging
docker-compose exec app bash

# View real-time logs
docker-compose logs -f app
```

## Contributing Guidelines

### 1. Code Review Process

1. **Create feature branch**
```bash
git checkout -b feature/new-assessment-type
```

2. **Make changes and test**
```bash
make test
make lint
make format
```

3. **Create pull request**
- Include tests for new functionality
- Update documentation
- Add type hints
- Follow commit message conventions

### 2. Commit Message Convention

```
type(scope): description

feat(assessment): add new anxiety assessment scale
fix(simulation): resolve memory leak in event processing
docs(api): update endpoint documentation
test(persona): add unit tests for personality updates
```

### 3. Testing Requirements

- All new code must have unit tests
- Integration tests for API endpoints
- Performance tests for critical paths
- Documentation updates for new features 