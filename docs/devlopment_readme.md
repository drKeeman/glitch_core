# Development Guide

## 🐳 Docker-First Development

This project is designed for Docker-first development, meaning all development tasks run in containers. This ensures consistent environments across different machines and eliminates "works on my machine" issues.

## 🚀 Getting Started

### Prerequisites

1. **Docker Desktop** - Install from [docker.com](https://www.docker.com/products/docker-desktop/)
2. **Docker Compose** - Usually included with Docker Desktop
3. **Git** - For version control

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd glitch_core

# Run the development setup script
./scripts/dev-setup.sh

# Start development environment
make dev
```

## 🛠️ Development Workflow

### Daily Development

1. **Start the development environment:**
   ```bash
   make dev
   ```
   This starts the API with hot reload enabled.

2. **Run tests:**
   ```bash
   make test
   ```

3. **Check code quality:**
   ```bash
   make lint
   make format
   ```

4. **View logs:**
   ```bash
   make logs          # API logs only
   make logs-all      # All service logs
   ```

### Code Quality

All code quality tools run in Docker containers:

```bash
# Format code
make format

# Run linting checks
make lint

# Run tests
make test

# Run tests with watch mode
make test-watch
```

### Database Access

```bash
# Access Qdrant shell
make qdrant-shell

# Access Redis CLI
make redis-cli
```

### Development Tools

```bash
# Start development tools container
make dev-tools

# Open shell in dev tools container
make dev-tools-shell

# Open shell in API container
make dev-shell
```

## 🏗️ Architecture

### Services

- **API** (`glitch-api`): FastAPI application with hot reload
- **Qdrant** (`glitch-qdrant`): Vector database for memory storage
- **Redis** (`glitch-redis`): Cache layer for active context
- **Ollama** (`glitch-ollama`): Local LLM inference
- **Nginx** (`glitch-nginx`): Reverse proxy (production only)
- **Prometheus** (`glitch-prometheus`): Monitoring (optional)

### Development vs Production

| Aspect | Development | Production |
|--------|-------------|------------|
| Hot Reload | ✅ Enabled | ❌ Disabled |
| Log Level | DEBUG | INFO |
| Memory Limits | Lower | Higher |
| Workers | 1 | 4 |
| Volume Mounts | Source code mounted | No mounts |

## 🔧 Configuration

### Environment Variables

Copy `env.example` to `.env` and customize:

```bash
cp env.example .env
```

Key variables:
- `ENV`: Environment (development/production)
- `LOG_LEVEL`: Logging verbosity
- `OLLAMA_MODEL`: LLM model to use
- `ALLOWED_ORIGINS`: CORS origins

### Docker Compose Files

- `docker-compose.yml`: Base configuration
- `docker-compose.override.yml`: Development overrides (auto-loaded)
- `docker-compose.prod.yml`: Production overrides

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run tests with watch mode
make test-watch

# Run tests locally (fallback)
make test-local
```

### Test Structure

```
tests/
├── unit/          # Unit tests
├── integration/   # Integration tests
├── scenarios/     # Scenario tests
└── conftest.py    # Test configuration
```

## 📊 Monitoring

### Health Checks

```bash
# Check all services
make health

# Individual service checks
curl http://localhost:8000/health    # API
curl http://localhost:6333/health    # Qdrant
curl http://localhost:11434/api/tags # Ollama
```

### Logs

```bash
# API logs
make logs

# All service logs
make logs-all

# Follow logs in real-time
docker-compose logs -f api
```

## 🚀 Deployment

### Development Deployment

```bash
# Build and start development environment
make dev-build

# Clean rebuild
make dev-clean
```

### Production Deployment

```bash
# Deploy to production
./scripts/prod-deploy.sh

# Or manually
make prod-build
make prod
```

## 🔍 Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   # Check what's using the ports
   lsof -i :8000
   lsof -i :6333
   lsof -i :6379
   lsof -i :11434
   ```

2. **Memory issues:**
   ```bash
   # Check Docker memory usage
   docker stats
   
   # Clean up Docker
   docker system prune -f
   ```

3. **Service not starting:**
   ```bash
   # Check service logs
   docker-compose logs api
   
   # Restart services
   docker-compose restart
   ```

### Debug Commands

```bash
# Check service status
docker-compose ps

# Check resource usage
docker stats

# Inspect container
docker-compose exec api /bin/bash

# View container logs
docker-compose logs -f api
```

## 📚 Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ollama Documentation](https://ollama.ai/docs) 