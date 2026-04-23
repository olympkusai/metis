"""RAG (Retrieval-Augmented Generation) implementation using Qdrant."""

from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from app.storage.qdrant_client import QdrantVectorStore, get_qdrant_store


class NewsRAG:
    """RAG system for crypto news using Qdrant."""
    
    def __init__(self, qdrant_store: Optional[QdrantVectorStore] = None):
        """Initialize news RAG.
        
        Args:
            qdrant_store: Qdrant store instance. If None, uses singleton.
        """
        self.qdrant = qdrant_store or get_qdrant_store()
        self.embeddings = OpenAIEmbeddings()
    
    async def index_news(
        self,
        news_items: List[Dict[str, Any]],
    ) -> List[str]:
        """Index news articles into Qdrant.
        
        Args:
            news_items: List of news dictionaries with keys:
                - title: News title
                - content: News content/body
                - url: Source URL
                - timestamp: Publication timestamp
                - source: Source name
                
        Returns:
            List of point IDs for indexed news
        """
        point_ids = []
        
        for news in news_items:
            # Combine title and content for embedding
            text = f"{news.get('title', '')}\n\n{news.get('content', '')}"
            
            # Generate embedding
            vector = self.embeddings.embed_query(text)
            
            # Create point ID
            import hashlib
            url = news.get('url', '')
            point_id = f"news:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
            
            # Prepare payload
            payload = {
                "title": news.get("title", ""),
                "content": news.get("content", ""),
                "url": url,
                "timestamp": news.get("timestamp", ""),
                "source": news.get("source", ""),
            }
            
            # Store in Qdrant
            await self.qdrant.upsert(
                collection_name=QdrantVectorStore.CRYPTO_NEWS,
                point_id=point_id,
                vector=vector,
                payload=payload,
            )
            
            point_ids.append(point_id)
        
        return point_ids
    
    async def search_news(
        self,
        query: str,
        top_k: int = 3,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant news articles.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            source_filter: Optional filter by source name
            
        Returns:
            List of news articles with relevance scores
        """
        # Generate query embedding
        query_vector = self.embeddings.embed_query(query)
        
        # Build filter
        filter_payload = None
        if source_filter:
            filter_payload = {"source": source_filter}
        
        # Search
        results = await self.qdrant.search(
            collection_name=QdrantVectorStore.CRYPTO_NEWS,
            query_vector=query_vector,
            limit=top_k,
            filter_payload=filter_payload,
        )
        
        return [
            {
                "title": result.payload.get("title", ""),
                "content": result.payload.get("content", ""),
                "url": result.payload.get("url", ""),
                "timestamp": result.payload.get("timestamp", ""),
                "source": result.payload.get("source", ""),
                "score": result.score,
            }
            for result in results
        ]
    
    async def get_recent_news(
        self,
        limit: int = 10,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent news articles.
        
        Args:
            limit: Maximum number of results
            source_filter: Optional filter by source name
            
        Returns:
            List of recent news articles
        """
        # Use a dummy vector for approximate recent retrieval
        # For production, consider using a timestamp index
        import random
        dummy_vector = [random.random() for _ in range(1536)]
        
        filter_payload = None
        if source_filter:
            filter_payload = {"source": source_filter}
        
        results = await self.qdrant.search(
            collection_name=QdrantVectorStore.CRYPTO_NEWS,
            query_vector=dummy_vector,
            limit=limit,
            filter_payload=filter_payload,
        )
        
        # Sort by timestamp
        sorted_results = sorted(
            results,
            key=lambda r: r.payload.get("timestamp", 0),
            reverse=True,
        )
        
        return [
            {
                "title": result.payload.get("title", ""),
                "content": result.payload.get("content", ""),
                "url": result.payload.get("url", ""),
                "timestamp": result.payload.get("timestamp", ""),
                "source": result.payload.get("source", ""),
                "score": result.score,
            }
            for result in sorted_results[:limit]
        ]
    
    async def delete_news(self, url: str) -> bool:
        """Delete a news article by URL.
        
        Args:
            url: News article URL
            
        Returns:
            True if deleted successfully
        """
        import hashlib
        point_id = f"news:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        
        try:
            await self.qdrant.delete(
                collection_name=QdrantVectorStore.CRYPTO_NEWS,
                point_id=point_id,
            )
            return True
        except Exception:
            return False


# Singleton instance
_news_rag: Optional[NewsRAG] = None


def get_news_rag() -> NewsRAG:
    """Get or create the news RAG singleton."""
    global _news_rag
    if _news_rag is None:
        _news_rag = NewsRAG()
    return _news_rag


def reset_news_rag():
    """Reset the news RAG singleton (for testing)."""
    global _news_rag
    _news_rag = None
