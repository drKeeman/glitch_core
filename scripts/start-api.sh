#!/bin/bash

# Script to start the API after services are ready
set -e

echo "🔍 Checking service health..."

# Wait for Qdrant
echo "⏳ Waiting for Qdrant..."
until curl -f http://localhost:6333/ > /dev/null 2>&1; do
    echo "   Qdrant not ready yet..."
    sleep 5
done
echo "✅ Qdrant is ready!"

# Wait for Ollama
echo "⏳ Waiting for Ollama..."
until curl -f http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "   Ollama not ready yet..."
    sleep 5
done
echo "✅ Ollama is ready!"

# Start API
echo "🚀 Starting API..."
docker-compose up api 