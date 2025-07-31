#!/bin/bash

# Performance optimization script for Glitch Core
# This script restarts services with optimized settings to reduce CPU usage

set -e

echo "🔧 Optimizing performance settings..."

# Stop current services
echo "🛑 Stopping current services..."
docker-compose down

# Wait a moment for cleanup
sleep 5

# Start services with new configuration
echo "🚀 Starting services with optimized settings..."
docker-compose up -d

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to initialize..."
sleep 30

# Check Ollama status
echo "📊 Checking Ollama status..."
docker exec -it glitch-core-ollama ollama list

# Monitor resource usage
echo "📈 Monitoring resource usage..."
docker stats --no-stream glitch-core-ollama

echo "✅ Performance optimization complete!"
echo ""
echo "📋 Optimization changes:"
echo "  • Limited CPU threads to 4"
echo "  • Increased context window to 2048 tokens"
echo "  • Added resource limits (4 CPU cores, 8GB RAM)"
echo "  • Reduced batch size to 1"
echo "  • Added request delays"
echo "  • Optimized sampling parameters"
echo ""
echo "🔍 Monitor with: docker stats glitch-core-ollama"
echo "📝 View logs with: docker logs -f glitch-core-ollama" 