# Development targets
dev: ## Start development environment
	docker-compose up
	@echo "🚀 Development environment started"
	@echo "📊 API: http://localhost:8000"
	@echo "🔍 Redis: http://localhost:6379"
	@echo "🗄️  Qdrant: http://localhost:6333"
	@echo "🤖 Ollama: http://localhost:11434"

dev-logs: ## View development logs
	docker-compose logs -f

dev-stop: ## Stop development environment
	docker-compose down

docker-down: ## Stop all Docker services
	docker-compose down

build: ## Build Docker images
	@echo "🔨 Building Docker images..."
	docker-compose build
	@echo "✅ Docker images built successfully"

build-no-cache: ## Build Docker images without cache
	@echo "🔨 Building Docker images (no cache)..."
	docker-compose build --no-cache
	@echo "✅ Docker images built successfully"

build-app: ## Build only the app Docker image
	@echo "🔨 Building app Docker image..."
	docker-compose build app
	@echo "✅ App Docker image built successfully"

rebuild-app: ## Rebuild app Docker image (force rebuild)
	@echo "🔨 Rebuilding app Docker image..."
	docker-compose build --no-cache app
	@echo "✅ App Docker image rebuilt successfully"

# Quick Setup target
setup: ## Complete setup with LLM (recommended)
	@echo "🚀 Complete Glitch Core Setup"
	./scripts/quick_setup.sh

setup-with-build: ## Complete setup with build and LLM
	@echo "🚀 Complete Glitch Core Setup with Build"
	make build
	./scripts/quick_setup.sh

# LLM Setup targets
llm-setup: ## Setup Ollama and pull required model
	@echo "🤖 Setting up LLM (Ollama)..."
	docker-compose up -d ollama
	@echo "⏳ Waiting for Ollama to start..."
	@sleep 10
	python scripts/setup_ollama.py

llm-test: ## Test LLM connection and generation
	@echo "🧪 Testing LLM connection..."
	python scripts/test_llm.py

llm-logs: ## View Ollama logs
	docker-compose logs -f ollama

llm-pull: ## Pull LLM model manually
	@echo "📥 Pulling LLM model..."
	docker exec -it glitch-core-ollama ollama pull llama3.1:8b

llm-list: ## List available models
	@echo "📋 Available models:"
	docker exec -it glitch-core-ollama ollama list

# Testing targets
test: ## Run all tests
	docker-compose exec app /app/.venv/bin/pytest tests/ -v

test-with-deps: ## Run all tests (ensures container is running with dev deps)
	docker-compose up -d app
	@echo "⏳ Waiting for app container to be ready..."
	@sleep 5
	docker-compose exec app /app/.venv/bin/pytest tests/ -v

test-unit: ## Run unit tests only
	docker-compose exec app /app/.venv/bin/pytest tests/unit/ -v

test-integration: ## Run integration tests only
	docker-compose exec app /app/.venv/bin/pytest tests/integration/ -v

# Code quality targets
lint: ## Run linting
	docker-compose exec app ruff check src/
	docker-compose exec app black --check src/
	docker-compose exec app isort --check-only src/

format: ## Format code
	docker-compose exec app black src/
	docker-compose exec app isort src/

# Database targets
db-reset: ## Reset all databases
	docker-compose down -v
	docker-compose up -d

# Simulation targets
sim-run: ## Run simulation with live LLM
	@echo "🎯 Running simulation with live LLM..."
	python scripts/run_simulation.py

sim-test: ## Test simulation setup
	@echo "🧪 Testing simulation setup..."
	python scripts/test_simulation.py

# Documentation targets
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	# Add documentation generation commands here

# Cleanup targets
clean: ## Clean up all containers and volumes
	docker-compose down -v --remove-orphans
	docker system prune -f

clean-sim-data: ## Clean up all simulation files and folders from data directory
	@echo "🧹 Cleaning up simulation data..."
	@rm -rf data/exports/simulation_*.json
	@rm -rf data/results/sim_*
	@rm -rf data/raw/sim_*
	@rm -rf data/processed/sim_*
	@echo "✅ Simulation data cleaned up successfully"

clean-models: ## Clean up model cache
	docker exec -it glitch-core-ollama ollama rm llama3.1:8b || true

# Help target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Default target
.DEFAULT_GOAL := help 