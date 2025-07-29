# AI Personality Drift Simulation

A research project for studying AI personality drift using mechanistic interpretability.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.12+
- uv package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd glitch-core
   ```

2. **Install dependencies**
   ```bash
   uv sync --all-extras
   ```

3. **Start services**
   ```bash
   make start
   ```

4. **Run tests**
   ```bash
   make test
   ```

## Development

### Available Commands

- `make start` - Start all services (Docker)
- `make test` - Run test suite
- `make lint` - Run linting
- `make format` - Format code
- `make dev` - Start development environment
- `make help` - Show all available commands

### Project Structure

```
src/
├── api/           # FastAPI application
├── core/          # Configuration and utilities
├── storage/       # Redis and Qdrant clients
├── models/        # Data models
├── services/      # Business logic
├── assessment/    # Psychiatric assessment tools
├── interpretability/ # Mechanistic analysis
└── utils/         # Utility functions
```

### Services

- **API**: FastAPI application on port 8000
- **Redis**: Session storage and caching on port 6379
- **Qdrant**: Vector database on port 6333

### API Endpoints

- `GET /api/v1/health` - Health check
- `GET /api/v1/status` - Application status
- `GET /docs` - API documentation

## Configuration

Copy `env.example` to `.env` and modify as needed:

```bash
cp env.example .env
```

## Testing

Run the full test suite:

```bash
make test
```

Run with coverage:

```bash
make test-cov
```

## Docker

Build and start all services:

```bash
make docker-build
make docker-up
```

View logs:

```bash
make docker-logs
```

## Research Goals

This project aims to:

1. **Simulate AI personality drift** under various conditions
2. **Apply mechanistic interpretability** to understand neural changes
3. **Develop assessment tools** for measuring personality changes
4. **Create intervention protocols** for managing drift

## License

MIT License
