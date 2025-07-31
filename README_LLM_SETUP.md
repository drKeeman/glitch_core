# 🚀 Quick Setup Guide

## One-Command Setup (Recommended)

```bash
make setup
```

This will:
- ✅ Start all Docker services
- ✅ Pull the required LLM model
- ✅ Test the connection
- ✅ Verify everything works

## Manual Setup

If you prefer step-by-step control:

### 1. Start Services
```bash
make dev
```

### 2. Setup LLM
```bash
make llm-setup
```

### 3. Test Everything
```bash
make llm-test
make sim-test
```

## Available Commands

### Development
- `make dev` - Start all services
- `make dev-logs` - View logs
- `make dev-stop` - Stop services

### LLM Management
- `make llm-setup` - Setup Ollama and pull model
- `make llm-test` - Test LLM connection
- `make llm-logs` - View Ollama logs
- `make llm-pull` - Pull model manually
- `make llm-list` - List available models

### Simulation
- `make sim-run` - Run simulation with live LLM
- `make sim-test` - Test simulation setup

### Cleanup
- `make clean` - Clean up all containers
- `make clean-models` - Remove LLM models

## What You Get

After setup, you'll have:

- **📊 API Server**: http://localhost:8000
- **🔍 Redis**: http://localhost:6379  
- **🗄️ Qdrant**: http://localhost:6333
- **🤖 Ollama**: http://localhost:11434

## Troubleshooting

### Docker Issues
```bash
# Check if Docker is running
docker info

# Restart services
make dev-stop
make dev
```

### LLM Issues
```bash
# Check Ollama status
make llm-test

# View Ollama logs
make llm-logs

# Pull model manually
make llm-pull
```

### Simulation Issues
```bash
# Test simulation setup
make sim-test

# Check all services
docker-compose ps
```

## Next Steps

1. **Run Simulation**: `make sim-run`
2. **View Results**: Check the API at http://localhost:8000
3. **Monitor Logs**: `make dev-logs`

## Model Information

- **Model**: `llama3.1:8b`
- **Size**: ~8GB RAM required
- **Performance**: First generation may be slower
- **Alternative**: Use `llama3.1:1b` for testing (change in `src/services/ollama_service.py`)

---

🎉 **You're ready to run AI personality drift experiments with live LLM!** 