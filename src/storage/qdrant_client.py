"""
Qdrant client for vector database operations.
"""

from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from src.core.config import settings
from src.core.exceptions import DatabaseConnectionError
from src.core.logging import get_logger

logger = get_logger(__name__)


class QdrantClient:
    """Async Qdrant client for vector storage."""
    
    def __init__(self):
        self._client: Optional[AsyncQdrantClient] = None
    
    async def connect(self) -> None:
        """Initialize Qdrant client connection."""
        try:
            self._client = AsyncQdrantClient(settings.QDRANT_URL)
            
            # Test connection
            await self._client.get_collections()
            logger.info("Qdrant connection established", url=settings.QDRANT_URL)
            
        except Exception as e:
            logger.error("Failed to connect to Qdrant", error=str(e), url=settings.QDRANT_URL)
            raise DatabaseConnectionError(f"Qdrant connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close Qdrant connection."""
        if self._client:
            await self._client.close()
        logger.info("Qdrant connection closed")
    
    async def ping(self) -> bool:
        """Test Qdrant connection."""
        try:
            if not self._client:
                await self.connect()
            await self._client.get_collections()
            return True
        except Exception as e:
            logger.error("Qdrant ping failed", error=str(e))
            return False
    
    async def create_collection(
        self, 
        collection_name: str, 
        vector_size: int = 768,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """Create a new collection in Qdrant."""
        try:
            if not self._client:
                await self.connect()
            
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance,
                )
            )
            logger.info("Collection created", name=collection_name, vector_size=vector_size)
            return True
            
        except Exception as e:
            logger.error("Failed to create collection", name=collection_name, error=str(e))
            return False
    
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            if not self._client:
                await self.connect()
            
            collections = await self._client.get_collections()
            return collection_name in [c.name for c in collections.collections]
            
        except Exception as e:
            logger.error("Failed to check collection existence", name=collection_name, error=str(e))
            return False
    
    async def upsert_points(
        self, 
        collection_name: str, 
        points: List[Dict[str, Any]]
    ) -> bool:
        """Upsert points to a collection."""
        try:
            if not self._client:
                await self.connect()
            
            await self._client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info("Points upserted", collection=collection_name, count=len(points))
            return True
            
        except Exception as e:
            logger.error("Failed to upsert points", collection=collection_name, error=str(e))
            return False
    
    async def search_points(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar points in a collection."""
        try:
            if not self._client:
                await self.connect()
            
            search_result = await self._client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            return [point.dict() for point in search_result]
            
        except Exception as e:
            logger.error("Failed to search points", collection=collection_name, error=str(e))
            return []


# Global Qdrant client instance
qdrant_client = QdrantClient() 