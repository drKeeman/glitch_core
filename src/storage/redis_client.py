"""
Redis client for session storage and caching.
"""

import asyncio
from typing import Any, Dict, Optional

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from src.core.config import settings
from src.core.exceptions import DatabaseConnectionError
from src.core.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Async Redis client with connection pooling."""
    
    def __init__(self):
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        """Initialize Redis connection pool and client."""
        try:
            self._pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                max_connections=20,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            logger.info("Redis connection established", url=settings.REDIS_URL)
            
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e), url=settings.REDIS_URL)
            raise DatabaseConnectionError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        logger.info("Redis connection closed")
    
    async def ping(self) -> bool:
        """Test Redis connection."""
        try:
            if not self._client:
                await self.connect()
            await self._client.ping()
            return True
        except Exception as e:
            logger.error("Redis ping failed", error=str(e))
            return False
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set a key-value pair in Redis."""
        try:
            if not self._client:
                await self.connect()
            return await self._client.set(key, value, ex=expire)
        except Exception as e:
            logger.error("Redis set failed", key=key, error=str(e))
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value from Redis."""
        try:
            if not self._client:
                await self.connect()
            return await self._client.get(key)
        except Exception as e:
            logger.error("Redis get failed", key=key, error=str(e))
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        try:
            if not self._client:
                await self.connect()
            return bool(await self._client.delete(key))
        except Exception as e:
            logger.error("Redis delete failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        try:
            if not self._client:
                await self.connect()
            return bool(await self._client.exists(key))
        except Exception as e:
            logger.error("Redis exists failed", key=key, error=str(e))
            return False


# Global Redis client instance
redis_client = RedisClient() 