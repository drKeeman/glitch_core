"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "AI Personality Drift Simulation"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data


def test_status_endpoint(client: TestClient):
    """Test status endpoint."""
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    
    data = response.json()
    assert data["app_name"] == "AI Personality Drift Simulation"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data
    assert "environment" in data
    assert "redis_url" in data["environment"]
    assert "qdrant_url" in data["environment"]


def test_openapi_docs(client: TestClient):
    """Test that OpenAPI docs are available."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema(client: TestClient):
    """Test that OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema 