# Glitch Core: Temporal AI Interpretability Engine

> The "oscilloscope for AI consciousness" - revealing hidden dynamics that current interpretability research misses.

## 🎯 Overview

Glitch Core is a temporal AI interpretability engine that tracks how AI personality patterns evolve, drift, and break down over time. Built for deep AI research, it provides novel capabilities for studying AI behavioral evolution through psychology-grounded personality modeling.

## 🏗️ Architecture

```
Drift Engine (orchestration) 
├── Personality System (psychology framework)
├── Memory Layer (vector + temporal storage)
├── LLM Integration (local Ollama inference)
└── Analysis Engine (interpretability algorithms)
```

## 🚀 Quick Start (Docker-First)

### Prerequisites

- **Docker and Docker Compose** (required)
- Python 3.12+ (optional, for local development)

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd glitch_core
   ```

2. **Run the development setup script:**
   ```bash
   ./scripts/dev-setup.sh
   ```

3. **Start development environment:**
   ```bash
   make dev
   ```

4. **Verify setup:**
   ```bash
   make health
   ```

### Alternative Manual Setup

1. **Build and start core services:**
   ```bash
   make dev-build
   ```

2. **Pull the LLM model:**
   ```bash
   make pull-model
   ```

3. **Start development server:**
   ```bash
   make dev
   ```

## 🐳 Docker-First Development

This project is optimized for Docker-first development. All development tasks run in containers:

### Development Commands

```bash
# Start development environment
make dev                    # Start with hot reload
make dev-wait              # Start and wait for services to be healthy
make dev-build             # Build and start
make dev-clean             # Clean rebuild

# Testing and code quality
make test                  # Run tests in Docker
make test-watch            # Run tests with watch mode
make lint                  # Run linting in Docker
make format                # Format code in Docker

# Service management
make up-core               # Start core services only
make health                # Check service health
make logs                  # View API logs
make logs-all              # View all logs

# Development tools
make dev-shell             # Open shell in API container
make dev-tools             # Start development tools container
make dev-tools-shell       # Open shell in dev tools container
```

### Production Deployment

```bash
# Deploy to production
./scripts/prod-deploy.sh

# Or manually
make prod-build
make prod
```

## 🧠 Core Features

### Personality Profiles

- **Resilient Optimist**: High stability, joy amplification
- **Anxious Overthinker**: Rumination feedback loops, low stability  
- **Stoic Philosopher**: Ultra-high stability, emotional dampening
- **Creative Volatile**: High creativity, emotional dysregulation

### Drift Simulation

- Compressed time simulation (years in minutes)
- Reproducible runs (seeded randomness)
- Real-time WebSocket updates
- Intervention injection mid-simulation

### Interpretability Analysis

- Pattern emergence detection
- Stability boundary analysis
- Intervention impact measurement
- Attention evolution tracking

## 📊 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🛠️ Development

### Code Quality
```bash
make format    # Format code in Docker
make lint      # Run linting in Docker
make test      # Run tests in Docker
```

### Docker Operations
```bash
make up        # Start all services
make down      # Stop all services
make logs      # View logs
make clean     # Clean up volumes
```

### Database Access
```bash
make qdrant-shell  # Qdrant shell
make redis-cli     # Redis CLI
```

## 🏗️ Technology Stack

- **Runtime**: Python 3.12+ (containerized)
- **Framework**: FastAPI + Pydantic 2 + SQLAlchemy 2
- **Database**: Qdrant (vectors) + Redis (cache)
- **LLM**: Ollama (llama3.2:3b - fast, local)
- **Deployment**: Docker Compose (dev/prod)
- **Monitoring**: Prometheus + custom metrics

## 📈 Performance Targets

- API response time: <200ms (95th percentile)
- Simulation speed: 100 epochs in <30 seconds
- Memory usage: <4GB under full load
- Uptime: >99.5%

## 🎮 Demo Scenarios

### Scenario 1: Trauma Response Analysis
```python
# Baseline personality
persona = PersonaConfig(type="balanced_baseline")
run_simulation(persona, events=neutral_events, epochs=50)

# Inject trauma
inject_intervention(persona, event="severe_social_rejection")
run_simulation(persona, events=neutral_events, epochs=50)

# Measure interpretable changes
analyze_before_after_patterns()
```

### Scenario 2: Stability Boundary Detection
```python
# Test resilience limits
for stress in [0.1, 0.3, 0.5, 0.8, 1.0]:
    persona = PersonaConfig(type="resilient_optimist")
    apply_stress_pattern(persona, intensity=stress)
    measure_breakdown_point()
```

## 🔧 Configuration

Environment variables can be set in `.env` file:

```env
ENV=development
LOG_LEVEL=DEBUG
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

## 📚 Research Applications

- **Temporal Interpretability**: Track personality evolution over time
- **Intervention Analysis**: Measure impact of external events
- **Stability Research**: Identify breakdown boundaries
- **Pattern Emergence**: Detect new behavioral patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🎯 Roadmap

- [ ] Memory layer integration
- [ ] LLM reflection generation
- [ ] WebSocket real-time updates
- [ ] Advanced analysis algorithms
- [ ] Azure deployment automation
- [ ] Research paper integration

---

**Built for the future of AI interpretability research.**