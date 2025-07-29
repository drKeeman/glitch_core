.PHONY: help install test lint format docker-build docker-up clean start

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@egrep '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync --all-extras

install-dev: ## Install with development dependencies
	uv sync --all-extras --dev

test: ## Run tests
	uv run pytest -v

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html --cov-report=term

test-phase3: ## Run Phase 3 persona test
	uv run python scripts/run_persona_test.py

run-persona-test: ## Run persona test (alias for test-phase3)
	uv run python scripts/run_persona_test.py

lint: ## Run linting
	uv run ruff check src tests
	uv run mypy src

format: ## Format code
	uv run black src tests
	uv run isort src tests
	uv run ruff --fix src tests

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all services
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## View logs
	docker-compose logs -f

start: ## Start all services (with live console)
	docker-compose up

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/

dev: ## Start development environment
	docker-compose up -d redis qdrant
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

all: install-dev lint test docker-build ## Run all quality checks and build 