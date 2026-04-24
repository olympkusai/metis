"""Qdrant vector database client wrapper for memory and RAG."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import asyncio
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.config import get_settings


@dataclass
class SearchResult:
    """Container for search results."""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


class QdrantVectorStore:
    """Qdrant client wrapper for vector operations."""
    
    # Collection names
    USER_MEMORIES = "user_memories"
    CRYPTO_NEWS = "crypto_news"
    CONVERSATION_HISTORY = "conversation_history"
    
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL. If None, uses from settings.
            api_key: Qdrant API key. If None, uses from settings.
        """
        settings = get_settings()
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        
        # Use async client for async operations
        if self.api_key:
            self.client = AsyncQdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = AsyncQdrantClient(url=self.url)
        self._collections_created = False
    
    async def ensure_collections(self) -> None:
        """Ensure all required collections exist."""
        if self._collections_created:
            return
        
        collections = [
            (self.USER_MEMORIES, 1536, Distance.COSINE),  # OpenAI embeddings
            (self.CRYPTO_NEWS, 1536, Distance.COSINE),
            (self.CONVERSATION_HISTORY, 1536, Distance.COSINE),
        ]
        
        for collection_name, vector_size, distance in collections:
            if not await self.client.collection_exists(collection_name):
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
        
        self._collections_created = True
    
    async def upsert(
        self,
        collection_name: str,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """Upsert a point to a collection.
        
        Args:
            collection_name: Name of the collection
            point_id: Unique point identifier
            vector: Embedding vector
            payload: Metadata payload
        """
        await self.ensure_collections()
        
        await self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
    
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum score threshold
            filter_payload: Optional filter on payload fields
            
        Returns:
            List of search results
        """
        await self.ensure_collections()
        
        query_filter = None
        if filter_payload:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                for key, value in filter_payload.items()
                if value is not None
            ]
            if conditions:
                query_filter = Filter(must=conditions)
        
        results = await self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )
        
        # query_points returns a QueryResponse object, iterate over points
        return [
            SearchResult(
                id=str(point.id),
                score=point.score if hasattr(point, 'score') else 0.0,
                payload=point.payload or {},
                vector=point.vector if hasattr(point, 'vector') else None,
            )
            for point in results.points
        ]
    
    async def delete(
        self,
        collection_name: str,
        point_id: str,
    ) -> None:
        """Delete a point from a collection.
        
        Args:
            collection_name: Name of the collection
            point_id: Point identifier to delete
        """
        await self.ensure_collections()
        
        await self.client.delete(
            collection_name=collection_name,
            points_selector=[point_id],
        )
    
    async def get_by_id(
        self,
        collection_name: str,
        point_id: str,
    ) -> Optional[SearchResult]:
        """Get a point by ID.
        
        Args:
            collection_name: Name of the collection
            point_id: Point identifier
            
        Returns:
            SearchResult if found, None otherwise
        """
        await self.ensure_collections()
        
        try:
            result = await self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
            )
            
            if result:
                point = result[0]
                return SearchResult(
                    id=str(point.id),
                    score=1.0,  # Exact match
                    payload=point.payload or {},
                    vector=point.vector,
                )
        except Exception:
            return None
        
        return None
    
    async def count(self, collection_name: str) -> int:
        """Count points in a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of points in the collection
        """
        await self.ensure_collections()
        
        count_result = await self.client.count(collection_name)
        return count_result.count
    
    async def clear_collection(self, collection_name: str) -> None:
        """Delete all points from a collection.
        
        Args:
            collection_name: Name of the collection
        """
        if await self.client.collection_exists(collection_name):
            scroll_result = await self.client.scroll(
                collection_name=collection_name,
                limit=10000,
            )
            points = [point.id for point in scroll_result[0]]
            if points:
                await self.client.delete(
                    collection_name=collection_name,
                    points_selector=points,
                )


# Singleton instance
_qdrant_store: Optional[QdrantVectorStore] = None


def get_qdrant_store() -> QdrantVectorStore:
    """Get or create the Qdrant store singleton."""
    global _qdrant_store
    if _qdrant_store is None:
        _qdrant_store = QdrantVectorStore()
    return _qdrant_store


def reset_qdrant_store():
    """Reset the Qdrant store singleton (for testing)."""
    global _qdrant_store
    _qdrant_store = None
