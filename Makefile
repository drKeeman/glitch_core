.PHONY: help install dev test lint format clean build up down logs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	uv sync

dev: ## Start development server
	uv run uvicorn src.glitch_core.api.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Run tests
	uv run pytest tests/ -v

lint: ## Run linting
	uv run black --check src/ tests/
	uv run isort --check-only src/ tests/
	uv run mypy src/

format: ## Format code
	uv run black src/ tests/
	uv run isort src/ tests/

clean: ## Clean up
	docker-compose down -v
	docker system prune -f

build: ## Build Docker images
	docker-compose build

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

logs: ## Show logs
	docker-compose logs -f api

logs-all: ## Show all logs
	docker-compose logs -f

# Core services only (no monitoring)
up-core: ## Start core services only
	docker-compose up -d api qdrant redis ollama

# Development with hot reload
dev-docker: ## Start development with Docker
	docker-compose -f docker-compose.yml up -d qdrant redis ollama
	uv run uvicorn src.glitch_core.api.main:app --host 0.0.0.0 --port 8000 --reload

# Health checks
health: ## Check service health
	curl -f http://localhost:8000/health || echo "API not healthy"
	curl -f http://localhost:6333/health || echo "Qdrant not healthy"
	curl -f http://localhost:11434/api/tags || echo "Ollama not healthy"

# Database operations
qdrant-shell: ## Open Qdrant shell
	docker-compose exec qdrant /bin/sh

redis-cli: ## Open Redis CLI
	docker-compose exec redis redis-cli

# Model management
pull-model: ## Pull Ollama model
	curl -X POST http://localhost:11434/api/pull -d '{"name": "llama3.2:3b"}'

list-models: ## List available models
	curl http://localhost:11434/api/tags 