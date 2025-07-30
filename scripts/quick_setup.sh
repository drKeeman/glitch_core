#!/bin/bash

# Quick Setup Script for Glitch Core with Live LLM
# This script sets up the entire environment including Ollama

set -e  # Exit on any error

echo "🚀 Glitch Core Quick Setup"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Start all services
start_services() {
    print_status "Starting all services..."
    docker-compose up -d
    print_success "Services started"
}

# Wait for Ollama to be ready
wait_for_ollama() {
    print_status "Waiting for Ollama to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_success "Ollama is ready"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts - Ollama not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "Ollama failed to start within 60 seconds"
    return 1
}

# Setup LLM
setup_llm() {
    print_status "Setting up LLM..."
    python scripts/setup_ollama.py
}

# Test LLM
test_llm() {
    print_status "Testing LLM connection..."
    python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from scripts.setup_ollama import test_ollama_connection
success = asyncio.run(test_ollama_connection())
exit(0 if success else 1)
"
    if [ $? -eq 0 ]; then
        print_success "LLM test passed"
    else
        print_error "LLM test failed"
        return 1
    fi
}

# Test simulation setup
test_simulation() {
    print_status "Testing simulation setup..."
    python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from src.services.simulation_engine import SimulationEngine
from src.models.simulation import ExperimentalCondition

async def test_sim():
    try:
        engine = SimulationEngine()
        success = await engine.initialize_simulation('experimental_design', ExperimentalCondition.CONTROL)
        print(f'Simulation setup: {success}')
        if success:
            await engine.cleanup()
        return success
    except Exception as e:
        print(f'Simulation test error: {e}')
        return False

success = asyncio.run(test_sim())
exit(0 if success else 1)
"
    if [ $? -eq 0 ]; then
        print_success "Simulation test passed"
    else
        print_error "Simulation test failed"
        return 1
    fi
}

# Main setup function
main() {
    echo ""
    print_status "Starting Glitch Core setup..."
    
    # Check prerequisites
    check_docker
    
    # Start services
    start_services
    
    # Wait for Ollama
    if ! wait_for_ollama; then
        print_error "Setup failed - Ollama did not start properly"
        exit 1
    fi
    
    # Setup LLM
    setup_llm
    
    # Test everything
    if test_llm && test_simulation; then
        echo ""
        print_success "🎉 Setup completed successfully!"
        echo ""
        echo "Your Glitch Core environment is ready:"
        echo "  📊 API: http://localhost:8000"
        echo "  🔍 Redis: http://localhost:6379"
        echo "  🗄️  Qdrant: http://localhost:6333"
        echo "  🤖 Ollama: http://localhost:11434"
        echo ""
        echo "Next steps:"
        echo "  • Run simulation: make sim-run"
        echo "  • View logs: make dev-logs"
        echo "  • Test LLM: make llm-test"
        echo ""
    else
        print_error "Setup failed - some tests did not pass"
        exit 1
    fi
}

# Run main function
main "$@" 