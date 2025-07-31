# Ollama Setup for Live LLM

This document explains how to set up Ollama to provide live LLM functionality for the AI Personality Drift Simulation.

## What is Ollama?

Ollama is a framework for running large language models locally. It provides a simple API that allows you to run models like Llama, Mistral, and others on your own hardware.

## Quick Setup

### 1. Start the Ollama Service

The Ollama service is already configured in `docker-compose.yml`. To start it:

```bash
# Start all services including Ollama
docker-compose up -d

# Or start just Ollama
docker-compose up ollama
```

### 2. Pull the Required Model

The simulation uses the `llama3.1:8b` model. We can pull it manually:

```bash
# Connect to the Ollama container
docker exec -it glitch-core-ollama ollama pull llama3.1:8b
```

Or use the setup script:

```bash
python scripts/setup_ollama.py
```

### 3. Verify Setup

Test that everything is working:

```bash
# Test the connection
curl http://localhost:11434/api/tags

# Test generation
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "prompt": "Hello, how are you?",
    "stream": false
  }'
```

## Configuration

The Ollama service is configured with the following settings:

- **URL**: `http://ollama:11434` (internal Docker network)
- **Model**: `llama3.1:8b`
- **Memory**: Persistent storage via Docker volume
- **Health Check**: Automatic monitoring

## Troubleshooting

### Ollama Service Not Starting

1. Check Docker is running
2. Check available disk space (models can be several GB)
3. Check logs: `docker-compose logs ollama`

### Model Not Available

1. Pull the model manually: `docker exec -it glitch-core-ollama ollama pull llama3.1:8b`
2. Check available models: `docker exec -it glitch-core-ollama ollama list`

### Connection Issues

1. Verify the service is running: `docker-compose ps`
2. Check the URL in the environment: `OLLAMA_URL=http://localhost:11434`
3. Test connectivity: `curl http://localhost:11434/api/tags`

## Performance Considerations

- The `llama3.1:8b` model requires approximately 8GB of RAM
- First generation may be slower as the model loads
- Consider using a smaller model for testing (e.g., `llama3.1:1b`)

## Alternative Models

We can use different models by changing the model name in the Ollama service:

```python
# In src/services/ollama_service.py
self.model_name = "llama3.1:1b"  # Smaller model
# or
self.model_name = "mistral:7b"    # Alternative model
```

## Integration with Simulation

The simulation now uses Ollama for:

1. **Persona Response Generation**: Creating realistic responses based on personality traits
2. **Assessment Administration**: Conducting PHQ9, GAD7, and PSS10 assessments
3. **Memory Processing**: Generating contextual responses to events

The Ollama service provides much more realistic and consistent responses compared to the previous mock implementation. 