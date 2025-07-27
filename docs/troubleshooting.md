# Troubleshooting Guide

> Common issues and solutions for Glitch Core

## 🚨 Quick Diagnosis

### System Health Check

```bash
# Check all services
make health

# Expected output:
{
  "status": "healthy",
  "checks": {
    "api": true,
    "qdrant": true,
    "redis": true,
    "ollama": true
  }
}
```

### Service Status

```bash
# Check individual services
curl http://localhost:8000/health
curl http://localhost:6333/health  # Qdrant
redis-cli ping  # Redis
curl http://localhost:11434/api/tags  # Ollama
```

## 🔧 Common Issues

### 1. Ollama Not Responding

**Symptoms**:
- LLM reflections not generating
- Timeout errors in logs
- "Ollama service unavailable" in health check

**Solutions**:

```bash
# Check if Ollama is running
docker ps | grep ollama

# Restart Ollama service
make restart-ollama

# Pull the model if missing
make pull-model

# Check Ollama logs
docker logs glitch_core-ollama-1
```

**Manual Fix**:
```bash
# Start Ollama manually
docker run -d --name ollama -p 11434:11434 ollama/ollama:latest

# Pull the model
docker exec -it ollama ollama pull llama3.2:3b
```

### 2. Qdrant Connection Issues

**Symptoms**:
- "Qdrant service unavailable" in health check
- Memory operations failing
- Vector search errors

**Solutions**:

```bash
# Check Qdrant status
curl http://localhost:6333/health

# Restart Qdrant
make restart-qdrant

# Check Qdrant logs
docker logs glitch_core-qdrant-1

# Reset Qdrant data if corrupted
make reset-qdrant
```

**Manual Fix**:
```bash
# Start Qdrant manually
docker run -d --name qdrant \
  -p 6333:6333 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

### 3. Redis Connection Issues

**Symptoms**:
- "Redis service unavailable" in health check
- Cache operations failing
- Active context not working

**Solutions**:

```bash
# Check Redis status
redis-cli ping

# Restart Redis
make restart-redis

# Check Redis logs
docker logs glitch_core-redis-1
```

**Manual Fix**:
```bash
# Start Redis manually
docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine
```

### 4. API Not Starting

**Symptoms**:
- `make dev` fails
- Port 8000 not accessible
- Import errors in logs

**Solutions**:

```bash
# Check Python environment
python --version  # Should be 3.12+
uv --version

# Reinstall dependencies
uv sync --reinstall

# Check for port conflicts
lsof -i :8000

# Start with verbose logging
LOG_LEVEL=DEBUG make dev
```

**Common Import Errors**:
```bash
# Missing dependencies
uv add fastapi pydantic qdrant-client redis

# Environment issues
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Memory Issues

**Symptoms**:
- Slow performance
- Out of memory errors
- Docker containers crashing

**Solutions**:

```bash
# Check memory usage
docker stats

# Increase Docker memory limit
# In Docker Desktop: Settings > Resources > Memory > 8GB

# Clean up unused resources
make clean
docker system prune -f

# Monitor memory usage
htop
```

**Optimization**:
```bash
# Reduce experiment complexity
# In experiment config:
{
  "epochs": 50,  # Reduce from 100
  "events_per_epoch": 5,  # Reduce from 10
  "llm_model": "llama3.2:1b"  # Use smaller model
}
```

### 6. WebSocket Connection Issues

**Symptoms**:
- Real-time updates not working
- WebSocket connection failures
- Browser console errors

**Solutions**:

```bash
# Check WebSocket endpoint
curl -i -N -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==" \
  http://localhost:8000/ws/experiments/test

# Test WebSocket with wscat
npm install -g wscat
wscat -c ws://localhost:8000/ws/experiments/test
```

**Browser Issues**:
```javascript
// Check WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws/experiments/test');

ws.onopen = () => console.log('Connected');
ws.onerror = (error) => console.error('WebSocket error:', error);
ws.onclose = () => console.log('Disconnected');
```

### 7. Experiment Failures

**Symptoms**:
- Experiments stuck in "running" state
- No patterns detected
- Analysis returns empty results

**Solutions**:

```bash
# Check experiment logs
docker logs glitch_core-api-1 | grep experiment_id

# Restart the experiment
curl -X DELETE http://localhost:8000/api/v1/experiments/{experiment_id}
curl -X POST http://localhost:8000/api/v1/experiments/ -d '...'

# Check system resources
docker stats
```

**Debug Experiment**:
```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check experiment status
response = requests.get(f"http://localhost:8000/api/v1/experiments/{experiment_id}")
print(response.json())
```

### 8. Database Corruption

**Symptoms**:
- Inconsistent data
- Qdrant/Redis errors
- Missing experiments or personas

**Solutions**:

```bash
# Backup current data
docker cp glitch_core-qdrant-1:/qdrant/storage ./backup_qdrant
docker cp glitch_core-redis-1:/data ./backup_redis

# Reset databases
make reset-db

# Restore from backup (if needed)
docker cp ./backup_qdrant glitch_core-qdrant-1:/qdrant/storage
docker cp ./backup_redis glitch_core-redis-1:/data
```

## 🔍 Debugging Techniques

### 1. Log Analysis

```bash
# View all logs
make logs

# View specific service logs
docker logs glitch_core-api-1
docker logs glitch_core-qdrant-1
docker logs glitch_core-redis-1
docker logs glitch_core-ollama-1

# Follow logs in real-time
docker logs -f glitch_core-api-1
```

### 2. Network Debugging

```bash
# Check network connectivity
docker network ls
docker network inspect glitch_core_default

# Test service communication
docker exec glitch_core-api-1 curl http://qdrant:6333/health
docker exec glitch_core-api-1 curl http://redis:6379
docker exec glitch_core-api-1 curl http://ollama:11434/api/tags
```

### 3. Performance Profiling

```bash
# Monitor system resources
htop
iotop
nethogs

# Docker resource usage
docker stats --no-stream

# API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
```

### 4. Code Debugging

```python
# Add debug logging
import structlog
logger = structlog.get_logger()

logger.debug("Processing event", event=event, state=current_state)

# Use Python debugger
import pdb; pdb.set_trace()

# Profile specific functions
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... your code ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## 🛠️ Recovery Procedures

### 1. Complete System Reset

```bash
# Stop all services
make down

# Clean up volumes
make clean

# Rebuild and restart
make build
make up

# Verify health
make health
```

### 2. Data Recovery

```bash
# Create backup
make backup

# Restore from backup
make restore

# Verify data integrity
curl http://localhost:8000/api/v1/personas/
curl http://localhost:8000/api/v1/experiments/
```

### 3. Service Recovery

```bash
# Restart specific service
make restart-api
make restart-qdrant
make restart-redis
make restart-ollama

# Restart all services
make restart
```

## 📊 Monitoring and Alerts

### 1. Health Monitoring

```bash
# Set up monitoring script
#!/bin/bash
while true; do
  response=$(curl -s http://localhost:8000/health)
  if [[ $response != *"healthy"* ]]; then
    echo "System unhealthy: $response"
    # Send alert
  fi
  sleep 30
done
```

### 2. Performance Monitoring

```bash
# Monitor key metrics
curl http://localhost:8000/metrics

# Set up Prometheus alerts
# Alert when:
# - API response time > 200ms
# - Memory usage > 80%
# - Failed experiments > 5%
```

### 3. Log Monitoring

```bash
# Monitor for errors
docker logs -f glitch_core-api-1 | grep -i error

# Monitor for warnings
docker logs -f glitch_core-api-1 | grep -i warning

# Monitor experiment progress
docker logs -f glitch_core-api-1 | grep "experiment"
```

## 🚀 Performance Optimization

### 1. System Tuning

```bash
# Increase Docker resources
# Docker Desktop: 8GB RAM, 4 CPUs

# Optimize disk I/O
# Use SSD for Docker volumes

# Network optimization
# Use host networking for local development
```

### 2. Application Tuning

```python
# Optimize experiment settings
experiment_config = {
    "epochs": 50,  # Reduce for faster results
    "events_per_epoch": 5,  # Reduce complexity
    "llm_model": "llama3.2:1b",  # Smaller model
    "memory_limit": 1000,  # Limit memory operations
    "reflection_threshold": 0.8  # Fewer reflections
}
```

### 3. Database Optimization

```bash
# Qdrant optimization
# Increase memory limit
# Use SSD storage
# Optimize collection settings

# Redis optimization
# Increase memory limit
# Enable persistence
# Optimize eviction policy
```

## 📞 Getting Help

### 1. Self-Diagnosis

```bash
# Run diagnostic script
./scripts/diagnose.sh

# Check system requirements
python scripts/check_requirements.py

# Validate configuration
python scripts/validate_config.py
```

### 2. Community Support

- **GitHub Issues**: Report bugs and feature requests
- **Discord**: Join our community for real-time help
- **Documentation**: Check the [Developer Guide](developer-guide.md)

### 3. Debug Information

When reporting issues, include:

```bash
# System information
uname -a
docker --version
python --version

# Service status
make health

# Recent logs
docker logs --tail 100 glitch_core-api-1

# Configuration
cat .env
cat docker-compose.yml
```

---

**Remember: Most issues can be resolved by restarting services or checking the health endpoint. When in doubt, `make restart` is often the quickest solution!** 