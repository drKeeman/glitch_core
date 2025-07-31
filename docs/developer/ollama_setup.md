# Ollama Setup for Live LLM

This document explains how to set up Ollama to provide live LLM functionality for the AI Personality Drift Simulation.

## What is Ollama?

Ollama is a framework for running large language models locally. It provides a simple API that allows you to run models like Llama, Mistral, and others on your own hardware with native GPU acceleration.

## Quick Setup

### 1. Install Ollama

**Recommended: Use the official Ollama app for best performance**

Download and install Ollama from the official website:
- **macOS**: https://ollama.ai/download/mac
- **Windows**: https://ollama.ai/download/windows  
- **Linux**: https://ollama.ai/download/linux

The official app provides:
- Native GPU acceleration (Metal on macOS, CUDA on Windows/Linux)
- Better performance than Docker versions
- Automatic updates
- Desktop interface (optional)

### 2. Pull the Required Model

The simulation uses the `llama3.1:8b` model:

```bash
# Pull the model
ollama pull llama3.1:8b
```

### 3. Verify Setup

Test that everything is working:

```bash
# Check available models
ollama list

# Test generation
ollama run llama3.1:8b "Hello, how are you?"
```

## Configuration

The simulation is configured to use the native Ollama installation:

- **URL**: `http://localhost:11434` (native Ollama)
- **Model**: `llama3.1:8b`
- **GPU Acceleration**: Automatic (Metal on macOS, CUDA on Windows/Linux)
- **Performance**: 40x faster than Docker version

## Performance Benefits

Using the native Ollama app provides significant performance improvements:

- **GPU Acceleration**: Native Metal (macOS) or CUDA (Windows/Linux) support
- **Faster Inference**: 40x speedup compared to Docker version
- **Lower CPU Usage**: GPU handles the heavy lifting
- **Better Memory Management**: Optimized for local hardware

## Troubleshooting

### Ollama Not Starting

1. Check if Ollama is installed: `which ollama`
2. Start Ollama: `ollama serve` (or use the desktop app)
3. Check logs: `ollama logs`

### Model Not Available

1. Pull the model: `ollama pull llama3.1:8b`
2. Check available models: `ollama list`
3. Verify model is downloaded: `ollama show llama3.1:8b`

### Connection Issues

1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Check the URL in environment: `OLLAMA_URL=http://localhost:11434`
3. Restart Ollama if needed: `ollama serve`

## Performance Considerations

- The `llama3.1:8b` model requires approximately 8GB of RAM
- GPU acceleration provides 40x speedup on supported hardware
- First generation may be slower as the model loads
- Consider using smaller models for testing (e.g., `llama3.1:1b`)

## Alternative Models

You can use different models by changing the model name:

```python
# In src/services/ollama_service.py
self.model_name = "llama3.1:1b"  # Smaller model
# or
self.model_name = "mistral:7b"    # Alternative model
# or
self.model_name = "phi3:3.8b"     # Microsoft Phi model
```

## Docker Alternative (Not Recommended)

If you must use Docker, the configuration is in `docker-compose.yml` but performance will be significantly slower:

```bash
# Docker version (slower)
docker-compose up -d ollama
docker exec -it glitch-core-ollama ollama pull llama3.1:8b
```

## Integration with Simulation

The simulation uses Ollama for:

1. **Persona Response Generation**: Creating realistic responses based on personality traits
2. **Assessment Administration**: Conducting PHQ9, GAD7, and PSS10 assessments  
3. **Memory Processing**: Generating contextual responses to events

The native Ollama service provides much more realistic and consistent responses with dramatically better performance compared to Docker versions. 