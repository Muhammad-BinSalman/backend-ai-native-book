"""
Qdrant vector database service.

Provides Qdrant client initialization and collection management.
"""
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config.settings import settings


class QdrantService:
    """Qdrant client wrapper for vector operations."""

    def __init__(self) -> None:
        """Initialize Qdrant client."""
        self.client: Optional[QdrantClient] = None
        self.collection_name = settings.qdrant_collection_name

    async def initialize(self) -> None:
        """Initialize Qdrant client and create collection if needed."""
        self.client = QdrantClient(
            url=settings.qdrant_api_endpoint,
            api_key=settings.qdrant_api_key,
        )

        # Create collection if it doesn't exist
        if not await self.collection_exists():
            await self.create_collection()
            print(f"OK Created Qdrant collection: {self.collection_name}")
        else:
            print(f"OK Qdrant collection exists: {self.collection_name}")

    async def collection_exists(self) -> bool:
        """Check if collection exists."""
        if self.client is None:
            raise RuntimeError("Qdrant client not initialized")

        collections = self.client.get_collections().collections
        return any(
            collection.name == self.collection_name for collection in collections
        )

    async def create_collection(self) -> None:
        """Create collection for book chunks."""
        if self.client is None:
            raise RuntimeError("Qdrant client not initialized")

        # Get embedding dimension from Cohere model
        # embed-english-v3.0 produces 1024-dimensional vectors
        vector_size = 1024

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    async def upsert_chunks(
        self, points: List[PointStruct]
    ) -> None:
        """Upsert chunk vectors to Qdrant."""
        if self.client is None:
            raise RuntimeError("Qdrant client not initialized")

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    async def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        book_id: Optional[str] = None,
    ) -> List[PointStruct]:
        """Search for similar chunks."""
        if self.client is None:
            raise RuntimeError("Qdrant client not initialized")

        # Build filter for book_id if provided
        query_filter = None
        if book_id:
            query_filter = Filter(
                must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        ).points

        return results

    async def delete_by_book(self, book_id: str) -> None:
        """Delete all chunks for a specific book."""
        if self.client is None:
            raise RuntimeError("Qdrant client not initialized")

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
            ),
        )

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        if self.client is None:
            raise RuntimeError("Qdrant client not initialized")

        return self.client.get_collection(self.collection_name).model_dump()


# Global Qdrant service instance
qdrant_service = QdrantService()
