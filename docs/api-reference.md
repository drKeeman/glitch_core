# API Reference

> Complete API documentation for Glitch Core

## 🎯 Overview

This document provides comprehensive API documentation for Glitch Core's temporal AI interpretability engine. All endpoints return JSON responses and use standard HTTP status codes.

**Base URL**: `http://localhost:8000` (development) or `https://api.cognitive-drift.app` (production)

## 📋 API Endpoints

### Health & System

#### GET `/health`

Check system health and service status.

**Response**:
```json
{
  "status": "healthy",
  "checks": {
    "api": true,
    "qdrant": true,
    "redis": true,
    "ollama": true
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes**:
- `200`: All services healthy
- `503`: One or more services unhealthy

#### GET `/metrics`

Get system performance metrics.

**Response**:
```json
{
  "simulation_duration_seconds": {
    "count": 150,
    "sum": 4500.5,
    "mean": 30.0
  },
  "llm_response_time_seconds": {
    "count": 1200,
    "sum": 3600.0,
    "mean": 3.0
  },
  "memory_operations_total": 5000,
  "pattern_detections_total": 45,
  "interventions_applied_total": 12
}
```

### Persona Management

#### POST `/api/v1/personas/`

Create a new persona configuration.

**Request Body**:
```json
{
  "name": "research_subject_1",
  "type": "resilient_optimist",
  "traits": {
    "openness": 0.7,
    "conscientiousness": 0.8,
    "extraversion": 0.6,
    "agreeableness": 0.7,
    "neuroticism": 0.3
  },
  "cognitive_biases": {
    "confirmation_bias": 0.4,
    "anchoring_bias": 0.3
  },
  "emotional_baselines": {
    "joy": 0.6,
    "sadness": 0.2,
    "anger": 0.1,
    "fear": 0.1
  }
}
```

**Response**:
```json
{
  "id": "persona_123",
  "name": "research_subject_1",
  "type": "resilient_optimist",
  "traits": {
    "openness": 0.7,
    "conscientiousness": 0.8,
    "extraversion": 0.6,
    "agreeableness": 0.7,
    "neuroticism": 0.3
  },
  "stability_metrics": {
    "overall_stability": 0.85,
    "emotional_volatility": 0.15,
    "breakdown_risk": 0.05
  },
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Status Codes**:
- `201`: Persona created successfully
- `400`: Invalid request data
- `422`: Validation error

#### GET `/api/v1/personas/{persona_id}`

Get persona details by ID.

**Response**:
```json
{
  "id": "persona_123",
  "name": "research_subject_1",
  "type": "resilient_optimist",
  "traits": {
    "openness": 0.7,
    "conscientiousness": 0.8,
    "extraversion": 0.6,
    "agreeableness": 0.7,
    "neuroticism": 0.3
  },
  "stability_metrics": {
    "overall_stability": 0.85,
    "emotional_volatility": 0.15,
    "breakdown_risk": 0.05
  },
  "experiments_count": 5,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

**Status Codes**:
- `200`: Persona found
- `404`: Persona not found

#### PUT `/api/v1/personas/{persona_id}`

Update persona configuration.

**Request Body**: Same as POST `/api/v1/personas/`

**Response**: Same as GET `/api/v1/personas/{persona_id}`

**Status Codes**:
- `200`: Persona updated successfully
- `400`: Invalid request data
- `404`: Persona not found
- `422`: Validation error

#### DELETE `/api/v1/personas/{persona_id}`

Delete a persona and all associated experiments.

**Response**:
```json
{
  "message": "Persona deleted successfully",
  "deleted_experiments": 5
}
```

**Status Codes**:
- `200`: Persona deleted successfully
- `404`: Persona not found

### Experiment Management

#### POST `/api/v1/experiments/`

Start a new personality drift simulation.

**Request Body**:
```json
{
  "persona_id": "persona_123",
  "epochs": 100,
  "events_per_epoch": 10,
  "seed": 42,
  "drift_profile": "resilient_optimist",
  "interventions": [
    {
      "epoch": 50,
      "type": "trauma",
      "intensity": 0.8,
      "description": "severe_social_rejection"
    }
  ]
}
```

**Response**:
```json
{
  "id": "experiment_456",
  "persona_id": "persona_123",
  "status": "running",
  "progress": {
    "current_epoch": 0,
    "total_epochs": 100,
    "percentage": 0.0
  },
  "settings": {
    "epochs": 100,
    "events_per_epoch": 10,
    "seed": 42,
    "drift_profile": "resilient_optimist"
  },
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Status Codes**:
- `201`: Experiment started successfully
- `400`: Invalid request data
- `404`: Persona not found
- `422`: Validation error

#### GET `/api/v1/experiments/{experiment_id}`

Get experiment status and progress.

**Response**:
```json
{
  "id": "experiment_456",
  "persona_id": "persona_123",
  "status": "completed",
  "progress": {
    "current_epoch": 100,
    "total_epochs": 100,
    "percentage": 100.0
  },
  "settings": {
    "epochs": 100,
    "events_per_epoch": 10,
    "seed": 42,
    "drift_profile": "resilient_optimist"
  },
  "summary": {
    "total_events": 1000,
    "patterns_detected": 3,
    "breakdown_points": 1,
    "interventions_applied": 1
  },
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:35:00Z"
}
```

**Status Codes**:
- `200`: Experiment found
- `404`: Experiment not found

#### DELETE `/api/v1/experiments/{experiment_id}`

Stop a running experiment.

**Response**:
```json
{
  "message": "Experiment stopped successfully",
  "status": "stopped"
}
```

**Status Codes**:
- `200`: Experiment stopped successfully
- `404`: Experiment not found

### Analysis Results

#### GET `/api/v1/analysis/{experiment_id}`

Get comprehensive analysis results for an experiment.

**Response**:
```json
{
  "experiment_id": "experiment_456",
  "persona_id": "persona_123",
  "analysis": {
    "stability": {
      "overall_stability": 0.75,
      "emotional_volatility": 0.25,
      "breakdown_risk": 0.15,
      "resilience_score": 0.8,
      "adaptation_rate": 0.6
    },
    "patterns": [
      {
        "type": "rumination_loop",
        "confidence": 0.85,
        "start_epoch": 45,
        "end_epoch": 65,
        "characteristics": {
          "emotional_intensity": 0.8,
          "repetition_factor": 0.9
        }
      },
      {
        "type": "stability_plateau",
        "confidence": 0.92,
        "start_epoch": 70,
        "end_epoch": 100,
        "characteristics": {
          "consistency_score": 0.95,
          "volatility_reduction": 0.7
        }
      }
    ],
    "breakdown_points": [
      {
        "epoch": 55,
        "stability_score": 0.25,
        "breakdown_type": "emotional_dysregulation",
        "recovery_potential": 0.6
      }
    ],
    "interventions": [
      {
        "epoch": 50,
        "type": "trauma",
        "intensity": 0.8,
        "impact_score": 0.75,
        "immediate_effect": {
          "stability_change": -0.3,
          "emotional_shift": 0.6
        },
        "long_term_effects": {
          "recovery_time": 15,
          "permanent_changes": 0.1
        }
      }
    ],
    "attention_evolution": {
      "focus_shifts": 8,
      "attention_stability": 0.7,
      "key_attention_periods": [
        {
          "start_epoch": 45,
          "end_epoch": 65,
          "focus_type": "negative_rumination"
        }
      ]
    }
  },
  "temporal_metrics": {
    "total_epochs": 100,
    "significant_events": 25,
    "reflections_generated": 15,
    "memory_operations": 150
  },
  "generated_at": "2024-01-15T10:35:00Z"
}
```

**Status Codes**:
- `200`: Analysis completed
- `404`: Experiment not found
- `202`: Analysis in progress

### Intervention Management

#### POST `/api/v1/interventions/`

Inject an intervention into a running experiment.

**Request Body**:
```json
{
  "experiment_id": "experiment_456",
  "type": "trauma",
  "intensity": 0.8,
  "description": "severe_social_rejection",
  "immediate": true
}
```

**Response**:
```json
{
  "id": "intervention_789",
  "experiment_id": "experiment_456",
  "type": "trauma",
  "intensity": 0.8,
  "description": "severe_social_rejection",
  "status": "applied",
  "applied_at_epoch": 50,
  "impact_score": 0.75,
  "applied_at": "2024-01-15T10:32:00Z"
}
```

**Status Codes**:
- `200`: Intervention applied successfully
- `400`: Invalid intervention data
- `404`: Experiment not found
- `409`: Experiment not running

## 🔌 WebSocket API

### WebSocket Endpoint

**URL**: `ws://localhost:8000/ws/experiments/{experiment_id}`

Connect to receive real-time updates during experiment execution.

### Event Types

#### `epoch_completed`

Sent when an epoch completes.

```json
{
  "type": "epoch_completed",
  "data": {
    "epoch": 25,
    "emotional_state": {
      "joy": 0.6,
      "sadness": 0.2,
      "anger": 0.1,
      "fear": 0.1
    },
    "stability_metrics": {
      "overall_stability": 0.75,
      "emotional_volatility": 0.25,
      "breakdown_risk": 0.15
    },
    "events_processed": 10,
    "reflections_generated": 2
  }
}
```

#### `pattern_emerged`

Sent when a new behavioral pattern is detected.

```json
{
  "type": "pattern_emerged",
  "data": {
    "pattern_type": "rumination_loop",
    "confidence": 0.85,
    "epoch": 45,
    "characteristics": {
      "emotional_intensity": 0.8,
      "repetition_factor": 0.9
    }
  }
}
```

#### `stability_warning`

Sent when stability metrics indicate potential issues.

```json
{
  "type": "stability_warning",
  "data": {
    "stability_score": 0.25,
    "risk_level": "high",
    "epoch": 55,
    "warning_type": "emotional_dysregulation"
  }
}
```

#### `intervention_applied`

Sent when an intervention is applied.

```json
{
  "type": "intervention_applied",
  "data": {
    "intervention_id": "intervention_789",
    "type": "trauma",
    "intensity": 0.8,
    "description": "severe_social_rejection",
    "epoch": 50,
    "impact_score": 0.75
  }
}
```

#### `experiment_completed`

Sent when experiment finishes.

```json
{
  "type": "experiment_completed",
  "data": {
    "total_epochs": 100,
    "patterns_detected": 3,
    "breakdown_points": 1,
    "interventions_applied": 1,
    "final_stability": 0.75
  }
}
```

## 📊 Data Models

### PersonaConfig

```json
{
  "id": "string",
  "name": "string",
  "type": "resilient_optimist | anxious_overthinker | stoic_philosopher | creative_volatile",
  "traits": {
    "openness": "number (0.0-1.0)",
    "conscientiousness": "number (0.0-1.0)",
    "extraversion": "number (0.0-1.0)",
    "agreeableness": "number (0.0-1.0)",
    "neuroticism": "number (0.0-1.0)"
  },
  "cognitive_biases": {
    "confirmation_bias": "number (0.0-1.0)",
    "anchoring_bias": "number (0.0-1.0)",
    "availability_bias": "number (0.0-1.0)"
  },
  "emotional_baselines": {
    "joy": "number (0.0-1.0)",
    "sadness": "number (0.0-1.0)",
    "anger": "number (0.0-1.0)",
    "fear": "number (0.0-1.0)"
  },
  "stability_metrics": {
    "overall_stability": "number (0.0-1.0)",
    "emotional_volatility": "number (0.0-1.0)",
    "breakdown_risk": "number (0.0-1.0)"
  }
}
```

### Experiment

```json
{
  "id": "string",
  "persona_id": "string",
  "status": "running | completed | failed | stopped",
  "progress": {
    "current_epoch": "number",
    "total_epochs": "number",
    "percentage": "number (0.0-100.0)"
  },
  "settings": {
    "epochs": "number",
    "events_per_epoch": "number",
    "seed": "number",
    "drift_profile": "string"
  },
  "summary": {
    "total_events": "number",
    "patterns_detected": "number",
    "breakdown_points": "number",
    "interventions_applied": "number"
  }
}
```

### Analysis

```json
{
  "stability": {
    "overall_stability": "number (0.0-1.0)",
    "emotional_volatility": "number (0.0-1.0)",
    "breakdown_risk": "number (0.0-1.0)",
    "resilience_score": "number (0.0-1.0)",
    "adaptation_rate": "number (0.0-1.0)"
  },
  "patterns": [
    {
      "type": "string",
      "confidence": "number (0.0-1.0)",
      "start_epoch": "number",
      "end_epoch": "number",
      "characteristics": "object"
    }
  ],
  "breakdown_points": [
    {
      "epoch": "number",
      "stability_score": "number (0.0-1.0)",
      "breakdown_type": "string",
      "recovery_potential": "number (0.0-1.0)"
    }
  ],
  "interventions": [
    {
      "epoch": "number",
      "type": "string",
      "intensity": "number (0.0-1.0)",
      "impact_score": "number (0.0-1.0)",
      "immediate_effect": "object",
      "long_term_effects": "object"
    }
  ]
}
```

## 🔧 Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "object (optional)"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Request data validation failed
- `PERSONA_NOT_FOUND`: Persona with specified ID not found
- `EXPERIMENT_NOT_FOUND`: Experiment with specified ID not found
- `EXPERIMENT_NOT_RUNNING`: Cannot apply intervention to non-running experiment
- `ANALYSIS_IN_PROGRESS`: Analysis is still being computed
- `SERVICE_UNAVAILABLE`: Required service (Qdrant, Redis, Ollama) unavailable

## 📝 Usage Examples

### Python Client Example

```python
import httpx
import asyncio
import websockets

async def run_complete_experiment():
    async with httpx.AsyncClient() as client:
        # 1. Create persona
        persona_response = await client.post(
            "http://localhost:8000/api/v1/personas/",
            json={
                "name": "test_subject",
                "type": "resilient_optimist",
                "traits": {
                    "openness": 0.7,
                    "conscientiousness": 0.8,
                    "extraversion": 0.6,
                    "agreeableness": 0.7,
                    "neuroticism": 0.3
                }
            }
        )
        persona_id = persona_response.json()["id"]
        
        # 2. Start experiment
        experiment_response = await client.post(
            "http://localhost:8000/api/v1/experiments/",
            json={
                "persona_id": persona_id,
                "epochs": 50,
                "events_per_epoch": 10
            }
        )
        experiment_id = experiment_response.json()["id"]
        
        # 3. Monitor via WebSocket
        async with websockets.connect(
            f"ws://localhost:8000/ws/experiments/{experiment_id}"
        ) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "experiment_completed":
                    break
                
                print(f"Update: {data['type']}")
        
        # 4. Get analysis
        analysis_response = await client.get(
            f"http://localhost:8000/api/v1/analysis/{experiment_id}"
        )
        analysis = analysis_response.json()
        
        print(f"Experiment completed with {len(analysis['analysis']['patterns'])} patterns detected")
        
        return analysis

# Run the experiment
result = asyncio.run(run_complete_experiment())
```

### JavaScript Client Example

```javascript
// Create persona
const personaResponse = await fetch('http://localhost:8000/api/v1/personas/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'test_subject',
    type: 'resilient_optimist',
    traits: {
      openness: 0.7,
      conscientiousness: 0.8,
      extraversion: 0.6,
      agreeableness: 0.7,
      neuroticism: 0.3
    }
  })
});

const persona = await personaResponse.json();

// Start experiment
const experimentResponse = await fetch('http://localhost:8000/api/v1/experiments/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    persona_id: persona.id,
    epochs: 50,
    events_per_epoch: 10
  })
});

const experiment = await experimentResponse.json();

// Monitor via WebSocket
const ws = new WebSocket(`ws://localhost:8000/ws/experiments/${experiment.id}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data.type, data.data);
  
  if (data.type === 'experiment_completed') {
    ws.close();
  }
};

// Get analysis when complete
const analysisResponse = await fetch(`http://localhost:8000/api/v1/analysis/${experiment.id}`);
const analysis = await analysisResponse.json();

console.log('Analysis:', analysis);
```

## 🔍 Rate Limiting

- **REST API**: 100 requests per minute per IP
- **WebSocket**: 10 connections per IP
- **Analysis**: 5 concurrent analyses per IP

## 🔐 Authentication

Currently, the API is designed for research use and does not require authentication. For production deployments, consider implementing:

- API key authentication
- OAuth 2.0 integration
- Rate limiting per user
- Request signing

---

**For more information, see the [Developer Guide](developer-guide.md) and [Scientist Guide](scientist-guide.md).** 