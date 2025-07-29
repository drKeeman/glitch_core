"""
Pytest configuration and fixtures.
"""

import asyncio
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.storage.redis_client import redis_client
from src.storage.qdrant_client import qdrant_client


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def redis_client_fixture():
    """Redis client fixture for testing."""
    await redis_client.connect()
    yield redis_client
    await redis_client.disconnect()


@pytest.fixture
async def qdrant_client_fixture():
    """Qdrant client fixture for testing."""
    await qdrant_client.connect()
    yield qdrant_client
    await qdrant_client.disconnect()


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app) 