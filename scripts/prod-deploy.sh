#!/bin/bash

# Production deployment script
set -e

echo "🚀 Deploying Glitch Core to production..."

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

# Check for required environment variables
if [ -z "$NGINX_HOST" ]; then
    echo "⚠️  NGINX_HOST not set, using localhost"
    export NGINX_HOST=localhost
fi

if [ -z "$NGINX_PORT" ]; then
    echo "⚠️  NGINX_PORT not set, using 80"
    export NGINX_PORT=80
fi

echo "✅ Environment check passed"

# Stop existing services
echo "🛑 Stopping existing services..."
docker-compose down

# Build production images
echo "🔨 Building production images..."
BUILD_TARGET=production docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Start production services
echo "🚀 Starting production services..."
BUILD_TARGET=production docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile production up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health checks
echo "🔍 Running health checks..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy"
else
    echo "❌ API is not responding"
    exit 1
fi

if curl -f http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant is healthy"
else
    echo "❌ Qdrant is not responding"
    exit 1
fi

if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is healthy"
else
    echo "❌ Ollama is not responding"
    exit 1
fi

echo ""
echo "🎉 Production deployment successful!"
echo ""
echo "Services:"
echo "  API:      http://localhost:8000"
echo "  Qdrant:   http://localhost:6333"
echo "  Redis:    localhost:6379"
echo "  Ollama:   http://localhost:11434"
echo "  Nginx:    http://$NGINX_HOST:$NGINX_PORT"
echo ""
echo "Useful commands:"
echo "  make logs         - View API logs"
echo "  make logs-all     - View all logs"
echo "  make down         - Stop services"
echo "  make health       - Check service health" 