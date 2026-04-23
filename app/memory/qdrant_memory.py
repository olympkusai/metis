"""User memory implementation using Qdrant vector store."""

from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from app.storage.qdrant_client import QdrantVectorStore, get_qdrant_store


class UserMemory:
    """Long-term memory for users using Qdrant."""
    
    def __init__(self, qdrant_store: Optional[QdrantVectorStore] = None):
        """Initialize user memory.
        
        Args:
            qdrant_store: Qdrant store instance. If None, uses singleton.
        """
        self.qdrant = qdrant_store or get_qdrant_store()
        self.embeddings = OpenAIEmbeddings()
    
    async def store_memory(
        self,
        user_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Store a memory for a user.
        
        Args:
            user_id: User identifier
            text: Memory text to store
            metadata: Additional metadata (e.g., timestamp, type)
            
        Returns:
            Point ID of the stored memory
        """
        # Generate embedding
        vector = self.embeddings.embed_query(text)
        
        # Create point ID
        import hashlib
        point_id = f"{user_id}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        
        # Prepare payload
        payload = {
            "user_id": user_id,
            "text": text,
            **metadata,
        }
        
        # Store in Qdrant
        await self.qdrant.upsert(
            collection_name=QdrantVectorStore.USER_MEMORIES,
            point_id=point_id,
            vector=vector,
            payload=payload,
        )
        
        return point_id
    
    async def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a user.
        
        Args:
            user_id: User identifier
            query: Query text to search for
            top_k: Maximum number of memories to retrieve
            memory_type: Optional filter by memory type
            
        Returns:
            List of memory dictionaries with text and score
        """
        # Generate query embedding
        query_vector = self.embeddings.embed_query(query)
        
        # Build filter
        filter_payload = {"user_id": user_id}
        if memory_type:
            filter_payload["type"] = memory_type
        
        # Search
        results = await self.qdrant.search(
            collection_name=QdrantVectorStore.USER_MEMORIES,
            query_vector=query_vector,
            limit=top_k,
            filter_payload=filter_payload,
        )
        
        return [
            {
                "text": result.payload.get("text", ""),
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k not in ["text", "user_id"]},
            }
            for result in results
        ]
    
    async def get_recent_memories(
        self,
        user_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get recent memories for a user (by timestamp).
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory dictionaries
        """
        # Note: Qdrant doesn't support time-based ordering natively
        # For production, consider adding a timestamp index or using a separate time-series DB
        filter_payload = {"user_id": user_id}
        
        # Use a dummy vector to get all user memories
        # In production, you'd want a better approach
        import random
        dummy_vector = [random.random() for _ in range(1536)]
        
        results = await self.qdrant.search(
            collection_name=QdrantVectorStore.USER_MEMORIES,
            query_vector=dummy_vector,
            limit=limit,
            filter_payload=filter_payload,
        )
        
        # Sort by timestamp if available
        sorted_results = sorted(
            results,
            key=lambda r: r.payload.get("timestamp", 0),
            reverse=True,
        )
        
        return [
            {
                "text": result.payload.get("text", ""),
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k not in ["text", "user_id"]},
            }
            for result in sorted_results[:limit]
        ]
    
    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Delete a specific memory.
        
        Args:
            user_id: User identifier
            memory_id: Memory point ID
            
        Returns:
            True if deleted successfully
        """
        try:
            await self.qdrant.delete(
                collection_name=QdrantVectorStore.USER_MEMORIES,
                point_id=memory_id,
            )
            return True
        except Exception:
            return False
    
    async def clear_user_memories(self, user_id: str) -> int:
        """Clear all memories for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of memories deleted
        """
        # Get all user memories
        memories = await self.get_recent_memories(user_id, limit=1000)
        
        # Delete each one
        deleted_count = 0
        for memory in memories:
            # Extract point ID from metadata or reconstruct it
            point_id = memory.get("metadata", {}).get("point_id")
            if point_id:
                success = await self.delete_memory(user_id, point_id)
                if success:
                    deleted_count += 1
        
        return deleted_count


# Singleton instance
_user_memory: Optional[UserMemory] = None


def get_user_memory() -> UserMemory:
    """Get or create the user memory singleton."""
    global _user_memory
    if _user_memory is None:
        _user_memory = UserMemory()
    return _user_memory


def reset_user_memory():
    """Reset the user memory singleton (for testing)."""
    global _user_memory
    _user_memory = None
