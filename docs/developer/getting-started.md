# Getting Started - Developer Guide

This guide will help you set up the AI Personality Drift Simulation project for development.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Docker & Docker Compose**: Required for containerized development
- **Python 3.12+**: For local development and testing
- **uv**: Modern Python package manager (recommended)
- **Git**: For version control

### System Requirements

- **RAM**: Minimum 8GB, recommended 16GB+ (for LLM operations)
- **Storage**: At least 10GB free space
- **CPU**: Multi-core processor recommended
- **OS**: macOS, Linux, or Windows with Docker support

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/drKeeman/glitch_core.git
cd glitch-core
```

### 2. Environment Setup

Copy the environment template and configure:

```bash
cp env.example .env
# Edit .env with your specific settings
```

### 3. Complete Setup (Recommended)

Run the complete setup script which includes LLM setup:

```bash
make setup
```

This command will:
- Build Docker images
- Start required services
- Download and configure the LLM model
- Run initial tests

### 4. Verify Installation

```bash
# Check if all services are running
make dev-logs

# Run tests to verify everything works
make test
```

## Development Environment

### Starting the Development Environment

```bash
make dev
```

This starts all services:
- **API Server**: http://localhost:8000
- **Redis**: http://localhost:6379
- **Qdrant**: http://localhost:6333
- **Ollama (LLM)**: http://localhost:11434
- **Frontend**: http://localhost:8000 (served by API)

### Development Commands

```bash
# View logs
make dev-logs

# Stop services
make dev-stop

# Rebuild app container
make rebuild-app

# Run tests
make test

# Code formatting
make format

# Linting
make lint
```

## Project Structure

```
glitch-core/
├── src/                    # Main application code
│   ├── api/               # FastAPI application
│   ├── core/              # Configuration and utilities
│   ├── models/            # Data models
│   ├── services/          # Business logic
│   ├── assessment/        # Psychiatric assessment tools
│   ├── interpretability/  # Mechanistic analysis
│   └── utils/             # Utility functions
├── config/                # Configuration files
│   ├── experiments/       # Experiment configurations
│   ├── personas/          # Persona definitions
│   └── simulation/        # Simulation settings
├── tests/                 # Test suite
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks
├── frontend/              # Web dashboard
└── scripts/               # Utility scripts
```

## Key Components

### 1. API Layer (`src/api/`)
- FastAPI application with REST endpoints
- WebSocket support for real-time updates
- CORS configuration for frontend integration

### 2. Core Services (`src/services/`)
- **SimulationEngine**: Main simulation orchestration
- **PersonaManager**: Persona lifecycle management
- **AssessmentService**: Psychiatric assessment administration
- **InterpretabilityService**: Mechanistic analysis

### 3. Data Models (`src/models/`)
- **Persona**: Personality representation
- **Event**: Simulation events (stress, neutral)
- **Assessment**: Psychiatric assessment results
- **Simulation**: Simulation state and configuration

### 4. Storage (`src/storage/`)
- **RedisClient**: Session and cache storage
- **QdrantClient**: Vector database for memory
- **FileStorage**: Data export and archival

## Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Application
APP_NAME=AI Personality Drift Simulation
API_PREFIX=/api/v1
CORS_ORIGINS=["http://localhost:3000"]

# Database
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333

# LLM
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=llama3.1:8b

# Simulation
SIMULATION_TIMEOUT=3600
MAX_CONCURRENT_PERSONAS=10
```

### Configuration Files

- `config/experiments/`: Experiment-specific configurations
- `config/personas/`: Persona baseline definitions
- `config/simulation/`: Simulation parameters

## Testing

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# With coverage
make test-with-deps
```

### Test Structure

```
tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
├── fixtures/          # Test fixtures
└── conftest.py       # Pytest configuration
```

## Debugging

### Viewing Logs

```bash
# All services
make dev-logs

# Specific service
docker-compose logs -f app
docker-compose logs -f ollama
```

### Debug Mode

```bash
# Start with debug mode
docker-compose up -d app
docker-compose exec app python -m debugpy --listen 0.0.0.0:5678 -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Common Issues

1. **Port conflicts**: Ensure ports 8000, 6379, 6333, 11434 are available
2. **Memory issues**: Increase Docker memory allocation for LLM operations
3. **Model download**: Use `make llm-pull` to manually download the LLM model

## Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs for interactive API documentation
2. **Run a simulation**: Use `make sim-run` to start a test simulation
3. **Check the dashboard**: Visit http://localhost:8000 for the web interface
4. **Read the architecture**: See [Architecture Overview](architecture.md) for system design details

## Getting Help

- **Documentation**: Check the relevant docs in this folder
- **Issues**: Create an issue in the repository
- **API Docs**: Visit http://localhost:8000/docs when running
- **Logs**: Use `make dev-logs` to debug issues 