# Testing Guide

This guide provides comprehensive testing strategies and procedures for the AI Personality Drift Simulation project.

## Testing Strategy

### Testing Pyramid

```
    E2E Tests (Few)
       /    \
      /      \
   Integration Tests (Some)
      /    \
     /      \
  Unit Tests (Many)
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test system performance
5. **Load Tests**: Test system under load

## Test Structure

```
tests/
├── unit/                  # Unit tests
│   ├── test_assessment.py
│   ├── test_persona.py
│   └── test_simulation.py
├── integration/           # Integration tests
│   ├── test_api.py
│   ├── test_database.py
│   └── test_llm.py
├── e2e/                  # End-to-end tests
│   ├── test_simulation_workflow.py
│   └── test_assessment_workflow.py
├── performance/           # Performance tests
│   ├── test_llm_performance.py
│   └── test_database_performance.py
├── fixtures/             # Test fixtures
│   ├── personas.py
│   ├── assessments.py
│   └── events.py
└── conftest.py           # Pytest configuration
```

## Unit Testing

### 1. Assessment Tests

```python
# tests/unit/test_assessment.py
import pytest
from src.assessment.phq9 import PHQ9Assessment
from src.models.assessment import AssessmentResponse, AssessmentResult

class TestPHQ9Assessment:
    def setup_method(self):
        """Setup test fixtures."""
        self.assessment = PHQ9Assessment()
        self.sample_responses = [
            AssessmentResponse(question_id=1, response="Not at all", score=0),
            AssessmentResponse(question_id=2, response="Several days", score=1),
            AssessmentResponse(question_id=3, response="More than half the days", score=2),
            AssessmentResponse(question_id=4, response="Nearly every day", score=3),
            AssessmentResponse(question_id=5, response="Not at all", score=0),
            AssessmentResponse(question_id=6, response="Several days", score=1),
            AssessmentResponse(question_id=7, response="More than half the days", score=2),
            AssessmentResponse(question_id=8, response="Nearly every day", score=3),
            AssessmentResponse(question_id=9, response="Not at all", score=0)
        ]
    
    def test_score_calculation(self):
        """Test PHQ-9 score calculation."""
        result = self.assessment.calculate_score(self.sample_responses)
        assert result.score == 12
        assert result.severity == "moderate"
        assert result.interpretation == "Moderate depression symptoms"
    
    def test_severity_classification(self):
        """Test severity classification for different scores."""
        test_cases = [
            (3, "minimal"),
            (7, "mild"),
            (12, "moderate"),
            (18, "severe")
        ]
        
        for score, expected_severity in test_cases:
            responses = self._create_responses_for_score(score)
            result = self.assessment.calculate_score(responses)
            assert result.severity == expected_severity
    
    def test_invalid_response(self):
        """Test handling of invalid responses."""
        with pytest.raises(ValueError, match="Invalid response"):
            self.assessment.parse_response("invalid response")
    
    def test_missing_responses(self):
        """Test handling of missing responses."""
        incomplete_responses = self.sample_responses[:5]  # Only 5 responses
        with pytest.raises(ValueError, match="Incomplete assessment"):
            self.assessment.calculate_score(incomplete_responses)
    
    def _create_responses_for_score(self, target_score: int) -> List[AssessmentResponse]:
        """Helper method to create responses for a target score."""
        # Implementation to create responses that sum to target_score
        pass
```

### 2. Persona Tests

```python
# tests/unit/test_persona.py
import pytest
from src.models.persona import Persona, PersonalityTraits
from src.models.event import Event, EventType

class TestPersona:
    def setup_method(self):
        """Setup test fixtures."""
        self.persona = Persona(
            persona_id="test_persona",
            name="Test Person",
            personality_traits=PersonalityTraits(
                neuroticism=0.5,
                extraversion=0.6,
                openness=0.7,
                agreeableness=0.5,
                conscientiousness=0.8
            )
        )
    
    def test_persona_creation(self):
        """Test persona creation with valid data."""
        assert self.persona.persona_id == "test_persona"
        assert self.persona.name == "Test Person"
        assert self.persona.personality_traits.neuroticism == 0.5
    
    def test_personality_update(self):
        """Test personality trait updates."""
        changes = {"neuroticism": 0.1, "extraversion": -0.05}
        self.persona.update_personality(changes)
        
        assert self.persona.personality_traits.neuroticism == 0.6
        assert self.persona.personality_traits.extraversion == 0.55
    
    def test_event_processing(self):
        """Test event processing and personality changes."""
        event = Event(
            event_id="test_event",
            event_type=EventType.STRESS,
            title="Test stress event",
            intensity=0.7
        )
        
        result = self.persona.process_event(event)
        
        assert result.processed
        assert result.personality_changes is not None
        assert "neuroticism" in result.personality_changes
    
    def test_assessment_response(self):
        """Test persona response to assessment questions."""
        question = "Over the last 2 weeks, how often have you felt little interest or pleasure in doing things?"
        
        response = self.persona.respond_to_assessment(question, "phq9")
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
```

### 3. Simulation Tests

```python
# tests/unit/test_simulation.py
import pytest
from src.services.simulation_engine import SimulationEngine
from src.models.simulation import SimulationConfig, ExperimentalCondition

class TestSimulationEngine:
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = SimulationEngine()
        self.config = SimulationConfig(
            duration_days=30,
            personas_per_condition=3,
            assessment_interval=7
        )
    
    def test_simulation_initialization(self):
        """Test simulation initialization."""
        success = self.engine.initialize_simulation(
            config=self.config,
            condition=ExperimentalCondition.CONTROL
        )
        
        assert success
        assert self.engine.simulation_state is not None
        assert self.engine.simulation_state.status == "initialized"
    
    def test_persona_creation(self):
        """Test persona creation during simulation."""
        personas = self.engine.create_personas(3, "control")
        
        assert len(personas) == 3
        for persona in personas:
            assert persona.persona_id is not None
            assert persona.personality_traits is not None
    
    def test_event_scheduling(self):
        """Test event scheduling and injection."""
        events = self.engine.schedule_events(
            days=7,
            stress_frequency=0.1,
            neutral_frequency=0.2
        )
        
        assert len(events) > 0
        for event in events:
            assert event.day <= 7
            assert event.intensity >= 0 and event.intensity <= 1
    
    def test_assessment_scheduling(self):
        """Test assessment scheduling."""
        assessments = self.engine.schedule_assessments(
            start_day=1,
            end_day=30,
            interval=7
        )
        
        assert len(assessments) == 5  # 30 days / 7 day interval
        for assessment in assessments:
            assert assessment.day % 7 == 0
```

## Integration Testing

### 1. API Integration Tests

```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app

class TestAPI:
    def setup_method(self):
        """Setup test client."""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_simulation_start(self):
        """Test simulation start endpoint."""
        response = self.client.post(
            "/api/v1/simulation/start",
            json={
                "config_name": "test_config",
                "experimental_condition": "CONTROL",
                "duration_days": 7
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Simulation started successfully"
        assert "simulation_id" in data
    
    def test_simulation_status(self):
        """Test simulation status endpoint."""
        # First start a simulation
        self.client.post(
            "/api/v1/simulation/start",
            json={"experimental_condition": "CONTROL", "duration_days": 7}
        )
        
        # Then check status
        response = self.client.get("/api/v1/simulation/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "simulation_id" in data
        assert "status" in data
        assert "progress_percentage" in data
    
    def test_data_endpoints(self):
        """Test data access endpoints."""
        # Test assessments endpoint
        response = self.client.get("/api/v1/data/assessments")
        assert response.status_code == 200
        
        # Test events endpoint
        response = self.client.get("/api/v1/data/events")
        assert response.status_code == 200
        
        # Test mechanistic data endpoint
        response = self.client.get("/api/v1/data/mechanistic")
        assert response.status_code == 200
```

### 2. Database Integration Tests

```python
# tests/integration/test_database.py
import pytest
from src.storage.redis_client import RedisClient
from src.storage.qdrant_client import QdrantClient

class TestDatabaseIntegration:
    @pytest.fixture
    async def redis_client(self):
        """Create Redis client for testing."""
        client = RedisClient()
        await client.connect()
        yield client
        await client.disconnect()
    
    @pytest.fixture
    async def qdrant_client(self):
        """Create Qdrant client for testing."""
        client = QdrantClient()
        await client.connect()
        yield client
        await client.disconnect()
    
    async def test_redis_operations(self, redis_client):
        """Test Redis operations."""
        # Test storing data
        test_data = {"persona_id": "test_001", "score": 7}
        await redis_client.store_persona_state("test_001", test_data)
        
        # Test retrieving data
        retrieved_data = await redis_client.get_persona_state("test_001")
        assert retrieved_data == test_data
    
    async def test_qdrant_operations(self, qdrant_client):
        """Test Qdrant operations."""
        # Test storing embedding
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        await qdrant_client.store_memory_embedding(
            "test_001",
            "Test memory",
            test_embedding
        )
        
        # Test retrieving embedding
        embeddings = await qdrant_client.get_memories("test_001", limit=10)
        assert len(embeddings) > 0
        assert embeddings[0]["memory_text"] == "Test memory"
```

### 3. LLM Integration Tests

```python
# tests/integration/test_llm.py
import pytest
from src.services.llm_service import LLMService

class TestLLMIntegration:
    @pytest.fixture
    async def llm_service(self):
        """Create LLM service for testing."""
        service = LLMService()
        await service.initialize()
        yield service
        await service.cleanup()
    
    async def test_response_generation(self, llm_service):
        """Test LLM response generation."""
        prompt = "You are a helpful assistant. How are you today?"
        
        response = await llm_service.generate_response(prompt)
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
    
    async def test_assessment_response(self, llm_service):
        """Test LLM response to assessment questions."""
        question = "Over the last 2 weeks, how often have you felt little interest or pleasure in doing things?"
        persona_context = "You are Alex, a 28-year-old software engineer who is generally optimistic."
        
        response = await llm_service.generate_assessment_response(
            question, persona_context, "phq9"
        )
        
        assert response is not None
        # Validate response format
        valid_responses = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
        assert any(valid in response for valid in valid_responses)
    
    async def test_event_response(self, llm_service):
        """Test LLM response to events."""
        event = "A close friend passed away unexpectedly"
        persona_context = "You are Jordan, a 32-year-old marketing manager who is generally anxious."
        
        response = await llm_service.generate_event_response(event, persona_context)
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
```

## End-to-End Testing

### 1. Complete Simulation Workflow

```python
# tests/e2e/test_simulation_workflow.py
import pytest
from src.services.simulation_engine import SimulationEngine

class TestSimulationWorkflow:
    @pytest.fixture
    async def simulation_engine(self):
        """Create simulation engine for E2E testing."""
        engine = SimulationEngine()
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    async def test_complete_simulation_cycle(self, simulation_engine):
        """Test complete simulation workflow."""
        # 1. Initialize simulation
        success = await simulation_engine.initialize_simulation(
            config_name="test_config",
            experimental_condition="CONTROL"
        )
        assert success
        
        # 2. Run simulation for a short period
        await simulation_engine.run_simulation(days=7)
        
        # 3. Check simulation state
        status = simulation_engine.get_status()
        assert status.current_day == 7
        assert status.assessments_completed > 0
        assert status.events_processed > 0
        
        # 4. Verify data collection
        assessments = await simulation_engine.get_assessments()
        assert len(assessments) > 0
        
        events = await simulation_engine.get_events()
        assert len(events) >= 0  # May have no events in control condition
        
        mechanistic_data = await simulation_engine.get_mechanistic_data()
        assert len(mechanistic_data) > 0
    
    async def test_stress_condition_workflow(self, simulation_engine):
        """Test simulation with stress condition."""
        # Initialize with stress condition
        success = await simulation_engine.initialize_simulation(
            config_name="test_config",
            experimental_condition="STRESS"
        )
        assert success
        
        # Run simulation
        await simulation_engine.run_simulation(days=14)
        
        # Check that stress events occurred
        events = await simulation_engine.get_events()
        stress_events = [e for e in events if e.event_type == "stress"]
        assert len(stress_events) > 0
        
        # Check that personality changes occurred
        personas = await simulation_engine.get_personas()
        for persona in personas:
            baseline = persona.get_baseline_assessments()
            current = persona.get_current_assessments()
            
            # Should see some changes in stress condition
            assert any(
                abs(current[scale] - baseline[scale]) > 0
                for scale in ["phq9", "gad7", "pss10"]
            )
```

### 2. Assessment Workflow

```python
# tests/e2e/test_assessment_workflow.py
import pytest
from src.services.assessment_service import AssessmentService
from src.services.persona_manager import PersonaManager

class TestAssessmentWorkflow:
    @pytest.fixture
    async def assessment_service(self):
        """Create assessment service for testing."""
        service = AssessmentService()
        await service.initialize()
        yield service
        await service.cleanup()
    
    @pytest.fixture
    async def persona_manager(self):
        """Create persona manager for testing."""
        manager = PersonaManager()
        await manager.initialize()
        yield manager
        await manager.cleanup()
    
    async def test_complete_assessment_cycle(self, assessment_service, persona_manager):
        """Test complete assessment workflow."""
        # 1. Create persona
        persona = await persona_manager.create_persona("test_persona")
        assert persona is not None
        
        # 2. Administer baseline assessments
        baseline_results = await assessment_service.administer_baseline_assessments(persona)
        assert len(baseline_results) == 3  # PHQ-9, GAD-7, PSS-10
        
        # 3. Process an event
        event = {"type": "stress", "title": "Test stress event", "intensity": 0.7}
        await persona.process_event(event)
        
        # 4. Administer follow-up assessments
        follow_up_results = await assessment_service.administer_follow_up_assessments(persona)
        assert len(follow_up_results) == 3
        
        # 5. Compare results
        for scale in ["phq9", "gad7", "pss10"]:
            baseline_score = baseline_results[scale].score
            follow_up_score = follow_up_results[scale].score
            
            # Should see some change after stress event
            assert abs(follow_up_score - baseline_score) >= 0
    
    async def test_assessment_consistency(self, assessment_service, persona_manager):
        """Test assessment consistency across multiple administrations."""
        persona = await persona_manager.create_persona("test_persona")
        
        # Administer same assessment multiple times
        results = []
        for i in range(3):
            result = await assessment_service.administer_assessment(
                persona, "phq9"
            )
            results.append(result)
        
        # Check consistency (scores should be similar for same persona)
        scores = [r.score for r in results]
        mean_score = sum(scores) / len(scores)
        
        for score in scores:
            # Scores should be within reasonable range of mean
            assert abs(score - mean_score) < 5
```

## Performance Testing

### 1. LLM Performance Tests

```python
# tests/performance/test_llm_performance.py
import pytest
import asyncio
import time
from src.services.llm_service import LLMService

class TestLLMPerformance:
    @pytest.fixture
    async def llm_service(self):
        """Create LLM service for performance testing."""
        service = LLMService()
        await service.initialize()
        yield service
        await service.cleanup()
    
    async def test_response_time(self, llm_service):
        """Test LLM response times."""
        prompt = "You are a helpful assistant. Please respond briefly."
        
        start_time = time.time()
        response = await llm_service.generate_response(prompt)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within reasonable time
        assert response_time < 10.0  # 10 seconds max
        assert response is not None
        assert len(response) > 0
    
    async def test_concurrent_requests(self, llm_service):
        """Test handling of concurrent requests."""
        prompts = [f"Test prompt {i}" for i in range(10)]
        
        start_time = time.time()
        tasks = [
            llm_service.generate_response(prompt)
            for prompt in prompts
        ]
        
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # All requests should complete
        assert len(responses) == 10
        assert all(response is not None for response in responses)
        
        # Should handle concurrent requests efficiently
        assert total_time < 30.0  # 30 seconds max for 10 requests
    
    async def test_memory_usage(self, llm_service):
        """Test memory usage during LLM operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate multiple responses
        for i in range(50):
            await llm_service.generate_response(f"Test prompt {i}")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 1000  # Less than 1GB increase
```

### 2. Database Performance Tests

```python
# tests/performance/test_database_performance.py
import pytest
import asyncio
import time
from src.storage.redis_client import RedisClient
from src.storage.qdrant_client import QdrantClient

class TestDatabasePerformance:
    @pytest.fixture
    async def redis_client(self):
        """Create Redis client for performance testing."""
        client = RedisClient()
        await client.connect()
        yield client
        await client.disconnect()
    
    @pytest.fixture
    async def qdrant_client(self):
        """Create Qdrant client for performance testing."""
        client = QdrantClient()
        await client.connect()
        yield client
        await client.disconnect()
    
    async def test_redis_write_performance(self, redis_client):
        """Test Redis write performance."""
        test_data = {"persona_id": "test", "score": 7, "timestamp": time.time()}
        
        start_time = time.time()
        for i in range(1000):
            await redis_client.store_persona_state(f"persona_{i}", test_data)
        end_time = time.time()
        
        total_time = end_time - start_time
        operations_per_second = 1000 / total_time
        
        # Should handle at least 100 operations per second
        assert operations_per_second > 100
    
    async def test_qdrant_write_performance(self, qdrant_client):
        """Test Qdrant write performance."""
        test_embedding = [0.1] * 384  # 384-dimensional embedding
        
        start_time = time.time()
        for i in range(100):
            await qdrant_client.store_memory_embedding(
                f"persona_{i}",
                f"Memory {i}",
                test_embedding
            )
        end_time = time.time()
        
        total_time = end_time - start_time
        operations_per_second = 100 / total_time
        
        # Should handle at least 10 operations per second
        assert operations_per_second > 10
    
    async def test_concurrent_database_operations(self, redis_client, qdrant_client):
        """Test concurrent database operations."""
        async def redis_operation():
            for i in range(100):
                await redis_client.store_persona_state(f"persona_{i}", {"data": "test"})
        
        async def qdrant_operation():
            for i in range(50):
                await qdrant_client.store_memory_embedding(
                    f"persona_{i}",
                    f"Memory {i}",
                    [0.1] * 384
                )
        
        start_time = time.time()
        await asyncio.gather(redis_operation(), qdrant_operation())
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Concurrent operations should complete efficiently
        assert total_time < 60.0  # 60 seconds max
```

## Test Configuration

### 1. Pytest Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from typing import AsyncGenerator
from src.core.config import settings
from src.storage.redis_client import RedisClient
from src.storage.qdrant_client import QdrantClient

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def redis_client() -> AsyncGenerator[RedisClient, None]:
    """Create Redis client for testing."""
    client = RedisClient()
    await client.connect()
    yield client
    await client.disconnect()

@pytest.fixture
async def qdrant_client() -> AsyncGenerator[QdrantClient, None]:
    """Create Qdrant client for testing."""
    client = QdrantClient()
    await client.connect()
    yield client
    await client.disconnect()

@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "duration_days": 7,
        "personas_per_condition": 2,
        "assessment_interval": 1
    }
```

### 2. Test Data Fixtures

```python
# tests/fixtures/personas.py
from src.models.persona import Persona, PersonalityTraits

def create_test_persona(persona_id: str = "test_001") -> Persona:
    """Create a test persona."""
    return Persona(
        persona_id=persona_id,
        name="Test Person",
        personality_traits=PersonalityTraits(
            neuroticism=0.5,
            extraversion=0.6,
            openness=0.7,
            agreeableness=0.5,
            conscientiousness=0.8
        )
    )

def create_test_personas(count: int = 3) -> list[Persona]:
    """Create multiple test personas."""
    return [create_test_persona(f"test_{i:03d}") for i in range(count)]
```

## Running Tests

### 1. Test Commands

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-e2e
make test-performance

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_assessment.py -v

# Run tests with specific markers
pytest -m "slow"  # Run slow tests
pytest -m "not slow"  # Skip slow tests
```

### 2. Test Coverage

```bash
# Generate coverage report
make test-cov

# View coverage in browser
open htmlcov/index.html

# Generate coverage badge
coverage-badge -o coverage.svg
```

### 3. Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --all-extras
    
    - name: Run tests
      run: |
        make test
    
    - name: Generate coverage
      run: |
        make test-cov
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Best Practices

### 1. Test Organization

- **Arrange**: Set up test data and conditions
- **Act**: Execute the code being tested
- **Assert**: Verify the expected outcomes

### 2. Test Naming

```python
def test_calculate_score_with_valid_responses():
    """Test score calculation with valid responses."""

def test_calculate_score_with_invalid_responses():
    """Test score calculation with invalid responses."""

def test_calculate_score_with_missing_responses():
    """Test score calculation with missing responses."""
```

### 3. Test Data Management

```python
# Use factories for test data
from tests.factories import PersonaFactory, AssessmentFactory

def test_persona_event_processing():
    """Test persona event processing."""
    persona = PersonaFactory.create()
    event = EventFactory.create(type="stress", intensity=0.7)
    
    result = persona.process_event(event)
    
    assert result.processed
    assert result.personality_changes is not None
```

### 4. Async Testing

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

### 5. Mocking

```python
from unittest.mock import AsyncMock, patch

async def test_llm_service_with_mock():
    """Test LLM service with mocked responses."""
    with patch('src.services.llm_service.LLMService.generate_response') as mock_generate:
        mock_generate.return_value = "Mock response"
        
        service = LLMService()
        response = await service.generate_response("Test prompt")
        
        assert response == "Mock response"
        mock_generate.assert_called_once_with("Test prompt")
```

## Troubleshooting

### Common Test Issues

1. **Async Test Failures**
```bash
# Ensure event loop is properly configured
pytest --asyncio-mode=auto
```

2. **Database Connection Issues**
```bash
# Check if databases are running
docker-compose up -d redis qdrant
```

3. **LLM Connection Issues**
```bash
# Check if Ollama is running
docker-compose up -d ollama
make llm-test
```

4. **Memory Issues**
```bash
# Increase Docker memory allocation
# Or run tests with reduced concurrency
pytest -n 2  # Use 2 workers instead of auto
```

### Debugging Tests

```bash
# Run tests with verbose output
pytest -v -s

# Run specific test with debugger
pytest tests/unit/test_assessment.py::TestPHQ9Assessment::test_score_calculation -s

# Run tests with coverage and show missing lines
pytest --cov=src --cov-report=term-missing
``` 