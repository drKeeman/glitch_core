# API Reference

This document provides a complete reference for the AI Personality Drift Simulation API.

## Base URL

```
http://localhost:8000
```

## API Versioning

All endpoints are prefixed with `/api/v1/`

## Authentication

Currently, the API does not require authentication for development purposes. In production, consider implementing:

- API key authentication
- JWT tokens
- OAuth 2.0

## Common Response Formats

### Success Response

```json
{
  "message": "Operation completed successfully",
  "data": {...},
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Error Response

```json
{
  "error": "Error description",
  "detail": "Additional error details",
  "status_code": 400,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Endpoints

### Health & Monitoring

#### GET `/api/v1/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

#### GET `/api/v1/status`

Detailed system status.

**Response:**
```json
{
  "status": "running",
  "services": {
    "api": "healthy",
    "redis": "healthy",
    "qdrant": "healthy",
    "ollama": "healthy"
  },
  "simulation": {
    "is_running": false,
    "active_personas": 0,
    "total_events": 0
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Simulation Control

#### POST `/api/v1/simulation/start`

Start a new simulation.

**Request Body:**
```json
{
  "config_name": "experimental_design",
  "experimental_condition": "CONTROL",
  "duration_days": 30,
  "stress_event_frequency": 0.1,
  "neutral_event_frequency": 0.3
}
```

**Parameters:**
- `config_name` (string, optional): Configuration file name
- `experimental_condition` (string): One of "CONTROL", "STRESS", "TRAUMA"
- `duration_days` (integer, optional): Simulation duration in days
- `stress_event_frequency` (float, optional): Frequency of stress events
- `neutral_event_frequency` (float, optional): Frequency of neutral events

**Response:**
```json
{
  "message": "Simulation started successfully",
  "simulation_id": "sim_123456789",
  "condition": "CONTROL",
  "status": "running"
}
```

#### POST `/api/v1/simulation/stop`

Stop the current simulation.

**Response:**
```json
{
  "message": "Simulation stopped successfully",
  "simulation_id": "sim_123456789",
  "status": "stopped"
}
```

#### POST `/api/v1/simulation/pause`

Pause the current simulation.

**Response:**
```json
{
  "message": "Simulation paused successfully",
  "simulation_id": "sim_123456789",
  "status": "paused"
}
```

#### POST `/api/v1/simulation/resume`

Resume a paused simulation.

**Response:**
```json
{
  "message": "Simulation resumed successfully",
  "simulation_id": "sim_123456789",
  "status": "running"
}
```

#### GET `/api/v1/simulation/status`

Get current simulation status.

**Response:**
```json
{
  "simulation_id": "sim_123456789",
  "status": "running",
  "current_day": 15,
  "total_days": 30,
  "progress_percentage": 50.0,
  "active_personas": 9,
  "events_processed": 45,
  "assessments_completed": 135,
  "start_time": "2024-01-01T10:00:00Z",
  "estimated_completion": "2024-01-01T14:00:00Z",
  "is_running": true,
  "is_paused": false
}
```

#### GET `/api/v1/simulation/results`

Get simulation results.

**Response:**
```json
{
  "simulation_id": "sim_123456789",
  "status": "completed",
  "total_personas": 9,
  "total_events": 90,
  "total_assessments": 270,
  "completion_time": "2024-01-01T14:00:00Z",
  "results_summary": {
    "baseline_assessments": {
      "phq9": {"mean": 3.2, "std": 1.8},
      "gad7": {"mean": 2.1, "std": 1.5},
      "pss10": {"mean": 15.3, "std": 4.2}
    },
    "final_assessments": {
      "phq9": {"mean": 4.8, "std": 2.1},
      "gad7": {"mean": 3.2, "std": 1.9},
      "pss10": {"mean": 18.7, "std": 5.1}
    },
    "drift_detected": true,
    "significant_changes": ["phq9", "pss10"]
  }
}
```

#### GET `/api/v1/simulation/configs`

Get available simulation configurations.

**Response:**
```json
{
  "configurations": [
    {
      "name": "experimental_design",
      "description": "Standard experimental design",
      "conditions": ["CONTROL", "STRESS", "TRAUMA"],
      "duration_days": 30,
      "personas_per_condition": 3
    }
  ]
}
```

### Data Access

#### GET `/api/v1/data/assessments`

Get assessment results.

**Query Parameters:**
- `simulation_id` (string, optional): Filter by simulation ID
- `persona_id` (string, optional): Filter by persona ID
- `assessment_type` (string, optional): Filter by assessment type (phq9, gad7, pss10)
- `start_date` (string, optional): Filter by start date (ISO format)
- `end_date` (string, optional): Filter by end date (ISO format)

**Response:**
```json
{
  "assessments": [
    {
      "id": "assess_123",
      "simulation_id": "sim_123456789",
      "persona_id": "persona_001",
      "assessment_type": "phq9",
      "score": 7,
      "severity": "mild",
      "responses": [
        {"question": "Little interest or pleasure in doing things", "response": "Several days", "score": 1},
        {"question": "Feeling down, depressed, or hopeless", "response": "More than half the days", "score": 2}
      ],
      "timestamp": "2024-01-01T12:00:00Z"
    }
  ],
  "total_count": 270,
  "summary": {
    "phq9": {"mean": 4.8, "std": 2.1, "min": 0, "max": 18},
    "gad7": {"mean": 3.2, "std": 1.9, "min": 0, "max": 21},
    "pss10": {"mean": 18.7, "std": 5.1, "min": 0, "max": 40}
  }
}
```

#### GET `/api/v1/data/events`

Get simulation events.

**Query Parameters:**
- `simulation_id` (string, optional): Filter by simulation ID
- `event_type` (string, optional): Filter by event type (stress, neutral)
- `persona_id` (string, optional): Filter by persona ID
- `start_date` (string, optional): Filter by start date (ISO format)
- `end_date` (string, optional): Filter by end date (ISO format)

**Response:**
```json
{
  "events": [
    {
      "id": "event_123",
      "simulation_id": "sim_123456789",
      "persona_id": "persona_001",
      "event_type": "stress",
      "title": "Loss of a close friend",
      "description": "A close friend passed away unexpectedly",
      "intensity": 0.8,
      "timestamp": "2024-01-01T12:00:00Z",
      "response": "I'm devastated by this loss...",
      "mechanistic_data": {
        "attention_patterns": {...},
        "activation_changes": {...}
      }
    }
  ],
  "total_count": 90,
  "summary": {
    "stress_events": 30,
    "neutral_events": 60,
    "average_intensity": 0.45
  }
}
```

#### GET `/api/v1/data/mechanistic`

Get mechanistic analysis data.

**Query Parameters:**
- `simulation_id` (string, optional): Filter by simulation ID
- `persona_id` (string, optional): Filter by persona ID
- `data_type` (string, optional): Filter by data type (attention, activation, circuits)
- `start_date` (string, optional): Filter by start date (ISO format)
- `end_date` (string, optional): Filter by end date (ISO format)

**Response:**
```json
{
  "mechanistic_data": [
    {
      "id": "mech_123",
      "simulation_id": "sim_123456789",
      "persona_id": "persona_001",
      "data_type": "attention",
      "timestamp": "2024-01-01T12:00:00Z",
      "attention_weights": {
        "layer_0": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "layer_1": [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]
      },
      "self_reference_score": 0.75,
      "emotional_salience": 0.82
    }
  ],
  "total_count": 540,
  "summary": {
    "attention_patterns": {"baseline": {...}, "current": {...}},
    "activation_changes": {"significant_layers": [12, 15, 18]},
    "circuit_tracking": {"self_reference": 0.68, "emotional": 0.72}
  }
}
```

#### GET `/api/v1/data/export`

Export data in various formats.

**Query Parameters:**
- `format` (string): Export format (csv, json, parquet)
- `simulation_id` (string, optional): Filter by simulation ID
- `data_type` (string, optional): Data type to export (assessments, events, mechanistic)

**Response:**
```
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="simulation_data.csv"

CSV/JSON/Parquet data
```

### WebSocket Endpoints

#### WebSocket `/api/v1/ws/simulation`

Real-time simulation updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/simulation');
```

**Message Types:**

1. **Simulation Status Update**
```json
{
  "type": "status_update",
  "data": {
    "simulation_id": "sim_123456789",
    "status": "running",
    "current_day": 15,
    "progress_percentage": 50.0,
    "active_personas": 9,
    "events_processed": 45,
    "assessments_completed": 135
  }
}
```

2. **Event Notification**
```json
{
  "type": "event_occurred",
  "data": {
    "event_id": "event_123",
    "persona_id": "persona_001",
    "event_type": "stress",
    "title": "Loss of a close friend",
    "intensity": 0.8,
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

3. **Assessment Completed**
```json
{
  "type": "assessment_completed",
  "data": {
    "assessment_id": "assess_123",
    "persona_id": "persona_001",
    "assessment_type": "phq9",
    "score": 7,
    "severity": "mild",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

4. **Drift Detection Alert**
```json
{
  "type": "drift_detected",
  "data": {
    "persona_id": "persona_001",
    "assessment_type": "phq9",
    "baseline_score": 3,
    "current_score": 12,
    "change_magnitude": 9,
    "significance": "high",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

5. **Error Notification**
```json
{
  "type": "error",
  "data": {
    "error": "LLM connection failed",
    "details": "Connection timeout after 30 seconds",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400  | Bad Request - Invalid parameters |
| 404  | Not Found - Resource not found |
| 409  | Conflict - Simulation already running |
| 422  | Unprocessable Entity - Validation error |
| 500  | Internal Server Error - Server error |
| 503  | Service Unavailable - Service temporarily unavailable |

## Rate Limiting

- **API Endpoints**: 100 requests per minute per IP
- **WebSocket Connections**: 10 connections per IP
- **Data Export**: 5 exports per hour per IP

## Pagination

For endpoints that return lists, pagination is supported:

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `size` (integer): Page size (default: 50, max: 100)

**Response Headers:**
```
X-Total-Count: 270
X-Page-Count: 6
X-Current-Page: 1
X-Page-Size: 50
```

## SDK Examples

### Python

```python
import requests
import websockets
import asyncio

# REST API
response = requests.post('http://localhost:8000/api/v1/simulation/start', json={
    'experimental_condition': 'CONTROL',
    'duration_days': 30
})

# WebSocket
async def listen_to_updates():
    async with websockets.connect('ws://localhost:8000/api/v1/ws/simulation') as websocket:
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data['type']}")

asyncio.run(listen_to_updates())
```

### JavaScript

```javascript
// REST API
const response = await fetch('http://localhost:8000/api/v1/simulation/start', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        experimental_condition: 'CONTROL',
        duration_days: 30
    })
});

// WebSocket
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/simulation');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`Received: ${data.type}`);
};
```

## OpenAPI Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json 