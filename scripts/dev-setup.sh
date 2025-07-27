#!/bin/bash

# Development environment setup script
set -e

echo "🚀 Setting up Glitch Core development environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

echo "✅ Docker environment check passed"

# Create necessary directories
mkdir -p scenarios/results
mkdir -p experiments/results
mkdir -p logs

echo "📁 Created necessary directories"

# Build and start core services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting core services..."
docker-compose up -d qdrant redis ollama

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
if curl -f http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant is healthy"
else
    echo "⚠️  Qdrant is not responding yet"
fi

if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is healthy"
else
    echo "⚠️  Ollama is not responding yet"
fi

# Pull default model if not already present
echo "📥 Checking for Ollama model..."
if ! curl -s http://localhost:11434/api/tags | grep -q "llama3.2:3b"; then
    echo "📥 Pulling default model (llama3.2:3b)..."
    curl -X POST http://localhost:11434/api/pull -d '{"name": "llama3.2:3b"}'
else
    echo "✅ Model already available"
fi

echo ""
echo "🎉 Development environment is ready!"
echo ""
echo "Next steps:"
echo "  make dev          - Start the API with hot reload"
echo "  make dev-wait     - Start and wait for services to be healthy"
echo "  make test         - Run tests"
echo "  make lint         - Run linting"
echo "  make format       - Format code"
echo "  make health       - Check service health"
echo ""
echo "Services:"
echo "  API:      http://localhost:8000"
echo "  Qdrant:   http://localhost:6333"
echo "  Redis:    localhost:6379"
echo "  Ollama:   http://localhost:11434"
echo ""
echo "For more commands, run: make help" 