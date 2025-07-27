.PHONY: help install dev test lint format clean build up down logs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Docker-first development commands
dev: ## Start development environment with Docker
	docker-compose up -d qdrant redis ollama
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@echo "🚀 Starting API..."
	docker-compose up api

dev-build: ## Build and start development environment
	docker-compose build
	docker-compose up -d qdrant redis ollama
	docker-compose up api

dev-clean: ## Clean and rebuild development environment
	docker-compose down -v
	docker-compose build --no-cache
	docker-compose up -d qdrant redis ollama
	docker-compose up api

dev-wait: ## Start development environment and wait for services to be healthy
	docker-compose up -d qdrant redis ollama
	@echo "⏳ Waiting for services to be healthy..."
	@until curl -f http://localhost:6333/ > /dev/null 2>&1; do echo "Waiting for Qdrant..."; sleep 5; done
	@until curl -f http://localhost:11434/api/tags > /dev/null 2>&1; do echo "Waiting for Ollama..."; sleep 5; done
	@echo "✅ All services are healthy!"
	@echo "🚀 Starting API..."
	docker-compose up api

# Production deployment
prod: ## Start production environment
	BUILD_TARGET=production docker-compose --profile production up -d

prod-build: ## Build and start production environment
	BUILD_TARGET=production docker-compose --profile production build
	BUILD_TARGET=production docker-compose --profile production up -d

# Core services (for development)
up-core: ## Start core services only
	docker-compose up -d qdrant redis ollama

# Testing in Docker
test: ## Run tests in Docker
	docker-compose run --rm api uv run pytest tests/ -v

test-watch: ## Run tests with watch mode
	docker-compose run --rm api uv run pytest tests/ -v --watch

# Code quality in Docker
lint: ## Run linting in Docker
	docker-compose run --rm api uv run black --check src/ tests/
	docker-compose run --rm api uv run isort --check-only src/ tests/
	docker-compose run --rm api uv run mypy src/

format: ## Format code in Docker
	docker-compose run --rm api uv run black src/ tests/
	docker-compose run --rm api uv run isort src/ tests/

# Docker management
build: ## Build all Docker images
	docker-compose build

build-no-cache: ## Build all Docker images without cache
	docker-compose build --no-cache

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

down-clean: ## Stop all services and remove volumes
	docker-compose down -v

logs: ## Show API logs
	docker-compose logs -f api

logs-all: ## Show all logs
	docker-compose logs -f

# Health checks
health: ## Check service health
	curl -f http://localhost:8000/health || echo "API not healthy"
	curl -f http://localhost:6333/ || echo "Qdrant not healthy"
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

# Development tools
dev-shell: ## Open development shell in container
	docker-compose run --rm api /bin/bash

dev-tools: ## Start development tools container
	docker-compose --profile dev-tools up -d dev-tools

dev-tools-shell: ## Open shell in dev tools container
	docker-compose exec dev-tools /bin/bash

# Scenario execution in Docker
run-scenarios: ## Run all demo scenarios in Docker
	DOCKER_ENV=1 docker-compose exec api uv run python -m scenarios.run_all_scenarios

run-trauma-analysis: ## Run trauma response analysis scenario in Docker
	DOCKER_ENV=1 docker-compose exec api uv run python -m scenarios.trauma_response_analysis

scenarios-help: ## Show scenario help
	@echo "Available scenarios:"
	@echo "  run-scenarios        - Run all scenarios with comprehensive report"
	@echo "  run-trauma-analysis  - Run trauma response analysis only"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Docker environment running"
	@echo "  - All dependencies available in container"
	@echo ""
	@echo "Results will be saved to scenarios/results/"

# Research and analysis
research-report: ## Generate research report from latest results
	@echo "Generating research report..."
	@ls -t scenarios/results/research_report_*.md | head -1 | xargs cat || echo "No research report found"

view-results: ## View latest scenario results
	@echo "Latest results:"
	@ls -t scenarios/results/*.json | head -5 | xargs -I {} echo "  {}"

clean-results: ## Clean up scenario results
	rm -rf scenarios/results/*.json scenarios/results/*.html scenarios/results/*.md

# Monitoring
monitoring: ## Start monitoring services
	docker-compose --profile monitoring up -d

# Local development (fallback)
install: ## Install dependencies locally (fallback)
	uv sync

dev-local: ## Start development server locally (fallback)
	uv run uvicorn src.glitch_core.api.main:app --host 0.0.0.0 --port 8000 --reload

test-local: ## Run tests locally (fallback)
	uv run pytest tests/ -v

lint-local: ## Run linting locally (fallback)
	uv run black --check src/ tests/
	uv run isort --check-only src/ tests/
	uv run mypy src/

format-local: ## Format code locally (fallback)
	uv run black src/ tests/
	uv run isort src/ tests/

# Cleanup
clean: ## Clean up everything
	docker-compose down -v
	docker system prune -f
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -name "*.pyc" -delete 