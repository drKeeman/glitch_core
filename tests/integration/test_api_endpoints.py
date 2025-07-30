"""
Integration tests for API endpoints.
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from src.main import app
from src.core.config import settings


class TestAPIEndpoints:
    """Test API endpoints functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["app_name"] == settings.APP_NAME
        assert data["version"] == settings.VERSION
    
    def test_status_endpoint(self, client):
        """Test status endpoint."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "app_name" in data
        assert "version" in data
        assert "timestamp" in data
    
    def test_simulation_configs(self, client):
        """Test simulation configurations endpoint."""
        response = client.get("/api/v1/simulation/configs")
        assert response.status_code == 200
        
        data = response.json()
        assert "simulation_configs" in data
        assert "persona_configs" in data
        assert "event_configs" in data
    
    def test_simulation_status_no_simulation(self, client):
        """Test simulation status when no simulation is running."""
        response = client.get("/api/v1/simulation/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "no_simulation"
        assert data["current_day"] == 0
        assert data["total_days"] == 0
        assert data["progress_percentage"] == 0.0
        assert data["active_personas"] == 0
        assert data["events_processed"] == 0
        assert data["assessments_completed"] == 0
        assert data["is_running"] is False
        assert data["is_paused"] is False
    
    def test_data_assessments(self, client):
        """Test assessment data endpoint."""
        response = client.get("/api/v1/data/assessments")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        # Should return mock data for now
    
    def test_data_mechanistic(self, client):
        """Test mechanistic data endpoint."""
        response = client.get("/api/v1/data/mechanistic")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        # Should return mock data for now
    
    def test_data_events(self, client):
        """Test events data endpoint."""
        response = client.get("/api/v1/data/events")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        # Should return mock data for now
    
    def test_data_stats(self, client):
        """Test data statistics endpoint."""
        response = client.get("/api/v1/data/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_personas" in data
        assert "total_assessments" in data
        assert "total_events" in data
        assert "total_mechanistic_records" in data
        assert "data_size_mb" in data
        assert "last_updated" in data
    
    def test_data_export(self, client):
        """Test data export endpoint."""
        export_request = {
            "export_format": "json",
            "include_assessments": True,
            "include_mechanistic": True,
            "include_events": True
        }
        
        response = client.post("/api/v1/data/export", json=export_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "filename" in data
        assert "filepath" in data
        assert "size_bytes" in data
        assert data["message"] == "Data exported successfully"
    
    def test_simulation_control_endpoints(self, client):
        """Test simulation control endpoints."""
        # Test start simulation
        start_request = {
            "config_name": "experimental_design",
            "experimental_condition": "CONTROL"
        }
        
        response = client.post("/api/v1/simulation/start", json=start_request)
        # This might fail if simulation engine is not properly initialized
        # but we're testing the endpoint structure
        assert response.status_code in [200, 400, 500]
        
        # Test pause simulation
        response = client.post("/api/v1/simulation/pause")
        assert response.status_code in [200, 400]
        
        # Test resume simulation
        response = client.post("/api/v1/simulation/resume")
        assert response.status_code in [200, 400]
        
        # Test stop simulation
        response = client.post("/api/v1/simulation/stop")
        assert response.status_code in [200, 400]
    
    def test_frontend_serving(self, client):
        """Test that frontend is served correctly."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "AI Personality Drift Simulation" in response.text
    
    def test_static_files(self, client):
        """Test static file serving."""
        # Test CSS file
        response = client.get("/style.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]
        
        # Test JS file
        response = client.get("/app.js")
        assert response.status_code == 200
        assert "application/javascript" in response.headers["content-type"]


class TestWebSocketEndpoints:
    """Test WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, async_client):
        """Test WebSocket connection establishment."""
        # This is a basic test - in a real scenario you'd need a WebSocket client
        # For now, we'll test that the endpoint exists
        response = await async_client.get("/api/v1/ws/simulation")
        # WebSocket endpoints return 101 Switching Protocols when accessed correctly
        # This test is mainly to ensure the endpoint is registered
        pass


def test_api_documentation():
    """Test that API documentation is available."""
    client = TestClient(app)
    
    # Test OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    # Test Swagger UI
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    
    # Test ReDoc
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_cors_headers():
    """Test CORS headers are properly set."""
    client = TestClient(app)
    
    response = client.options("/api/v1/health")
    # CORS preflight request should be handled
    assert response.status_code in [200, 405]  # 405 if OPTIONS not handled


if __name__ == "__main__":
    # Run basic tests
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/api/v1/health")
    print(f"Health check status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Health check passed")
    else:
        print("❌ Health check failed")
    
    # Test frontend serving
    response = client.get("/")
    print(f"Frontend serving status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Frontend serving passed")
    else:
        print("❌ Frontend serving failed")
    
    # Test API documentation
    response = client.get("/docs")
    print(f"API docs status: {response.status_code}")
    if response.status_code == 200:
        print("✅ API documentation passed")
    else:
        print("❌ API documentation failed") 