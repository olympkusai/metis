"""Conversation history management using Qdrant vector store."""

from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
from enum import Enum
from langchain_openai import OpenAIEmbeddings
from app.storage.qdrant_client import QdrantVectorStore, get_qdrant_store
import uuid
import hashlib


class FeedbackRating(str, Enum):
    """Feedback rating scale."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    NEUTRAL = "neutral"


class MessageRole(str, Enum):
    """Message role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationHistory:
    """Conversation history manager using Qdrant."""
    
    def __init__(self, qdrant_store: Optional[QdrantVectorStore] = None):
        """Initialize conversation history manager.
        
        Args:
            qdrant_store: Qdrant store instance. If None, uses singleton.
        """
        self.qdrant = qdrant_store or get_qdrant_store()
        self.embeddings = OpenAIEmbeddings()
    
    def _generate_message_id(self, session_id: str, content: str) -> str:
        """Generate unique message ID as UUID."""
        # Generate deterministic UUID from session_id and content
        content_hash = hashlib.sha256(f"{session_id}:{content}".encode()).hexdigest()
        return str(uuid.UUID(hex=content_hash[:32]))
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID."""
        return f"{user_id}:{uuid.uuid4().hex}"
    
    async def save_message(
        self,
        user_id: str,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a message to conversation history.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Additional metadata (symbol, timeframe, pipeline_summary, etc.)
            
        Returns:
            Message ID
        """
        # Generate embedding
        vector = self.embeddings.embed_query(content)
        
        # Generate message ID
        message_id = self._generate_message_id(session_id, content)
        
        # Prepare payload with complete metadata
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "role": role.value,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
            "deleted_at": None,
            "feedback": None,
            "feedback_rating": None,
            "feedback_comment": None,
            "metadata": metadata or {},
        }
        
        # Store in Qdrant
        await self.qdrant.upsert(
            collection_name=QdrantVectorStore.CONVERSATION_HISTORY,
            point_id=message_id,
            vector=vector,
            payload=payload,
        )
        
        return message_id
    
    async def get_conversation_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 20,
        include_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a specific session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            include_deleted: Whether to include soft-deleted messages
            
        Returns:
            List of messages ordered by timestamp (oldest first)
        """
        # Get all messages for this session
        filter_payload = {
            "user_id": user_id,
            "session_id": session_id,
        }
        
        if not include_deleted:
            filter_payload["deleted_at"] = None
        
        # Use a dummy vector to retrieve all matching messages
        import random
        dummy_vector = [random.random() for _ in range(1536)]
        
        results = await self.qdrant.search(
            collection_name=QdrantVectorStore.CONVERSATION_HISTORY,
            query_vector=dummy_vector,
            limit=limit * 2,  # Fetch more to account for ordering
            filter_payload=filter_payload,
        )
        
        # Filter by deleted_at if needed (Qdrant filter might not handle None properly)
        messages = []
        for result in results:
            payload = result.payload
            if not include_deleted and payload.get("deleted_at") is not None:
                continue
            messages.append({
                "message_id": str(result.id),
                "role": payload.get("role"),
                "content": payload.get("content"),
                "timestamp": payload.get("timestamp"),
                "metadata": payload.get("metadata", {}),
                "feedback": payload.get("feedback"),
                "feedback_rating": payload.get("feedback_rating"),
                "feedback_comment": payload.get("feedback_comment"),
            })
        
        # Sort by timestamp (oldest first for conversation flow)
        messages.sort(key=lambda m: m["timestamp"])
        
        # Return last N messages
        return messages[-limit:] if len(messages) > limit else messages
    
    async def get_global_context(
        self,
        user_id: str,
        limit: int = 10,
        exclude_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get global context across all sessions.
        
        Args:
            user_id: User identifier
            limit: Maximum number of messages to retrieve
            exclude_session_id: Optional session ID to exclude (current session)
            
        Returns:
            List of recent messages across all sessions
        """
        filter_payload = {
            "user_id": user_id,
            "deleted_at": None,
        }
        
        # Note: Qdrant doesn't support "not equal" directly
        # We'll filter after retrieval
        
        import random
        dummy_vector = [random.random() for _ in range(1536)]
        
        results = await self.qdrant.search(
            collection_name=QdrantVectorStore.CONVERSATION_HISTORY,
            query_vector=dummy_vector,
            limit=limit * 3,  # Fetch more for filtering
            filter_payload=filter_payload,
        )
        
        messages = []
        for result in results:
            payload = result.payload
            if payload.get("deleted_at") is not None:
                continue
            if exclude_session_id and payload.get("session_id") == exclude_session_id:
                continue
            messages.append({
                "message_id": str(result.id),
                "session_id": payload.get("session_id"),
                "role": payload.get("role"),
                "content": payload.get("content"),
                "timestamp": payload.get("timestamp"),
                "metadata": payload.get("metadata", {}),
            })
        
        # Sort by timestamp (newest first for context)
        messages.sort(key=lambda m: m["timestamp"], reverse=True)
        
        return messages[:limit]
    
    async def get_conversation_sessions(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get all conversation sessions for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions to retrieve
            
        Returns:
            List of sessions with metadata
        """
        filter_payload = {
            "user_id": user_id,
            "deleted_at": None,
        }
        
        import random
        dummy_vector = [random.random() for _ in range(1536)]
        
        results = await self.qdrant.search(
            collection_name=QdrantVectorStore.CONVERSATION_HISTORY,
            query_vector=dummy_vector,
            limit=limit * 10,  # Fetch many to aggregate by session
            filter_payload=filter_payload,
        )
        
        # Aggregate by session_id
        sessions: Dict[str, Dict[str, Any]] = {}
        for result in results:
            payload = result.payload
            if payload.get("deleted_at") is not None:
                continue
            
            session_id = payload.get("session_id")
            if session_id not in sessions:
                sessions[session_id] = {
                    "session_id": session_id,
                    "message_count": 0,
                    "first_message": payload.get("timestamp"),
                    "last_message": payload.get("timestamp"),
                    "has_feedback": False,
                }
            
            sessions[session_id]["message_count"] += 1
            sessions[session_id]["last_message"] = max(
                sessions[session_id]["last_message"],
                payload.get("timestamp")
            )
            sessions[session_id]["first_message"] = min(
                sessions[session_id]["first_message"],
                payload.get("timestamp")
            )
            if payload.get("feedback"):
                sessions[session_id]["has_feedback"] = True
        
        # Convert to list and sort by last_message (newest first)
        session_list = list(sessions.values())
        session_list.sort(key=lambda s: s["last_message"], reverse=True)
        
        return session_list[:limit]
    
    async def soft_delete_conversation(self, user_id: str, session_id: str) -> int:
        """Soft delete an entire conversation session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Number of messages deleted
        """
        # Get all messages in session
        messages = await self.get_conversation_history(
            user_id=user_id,
            session_id=session_id,
            limit=1000,
            include_deleted=False,
        )
        
        # Mark each as deleted
        deleted_count = 0
        for message in messages:
            message_id = message["message_id"]
            success = await self.soft_delete_message(user_id, message_id)
            if success:
                deleted_count += 1
        
        return deleted_count
    
    async def soft_delete_message(self, user_id: str, message_id: str) -> bool:
        """Soft delete a specific message.
        
        Args:
            user_id: User identifier
            message_id: Message identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            # Get the message
            result = await self.qdrant.get_by_id(
                collection_name=QdrantVectorStore.CONVERSATION_HISTORY,
                point_id=message_id,
            )
            
            if not result:
                return False
            
            # Verify user ownership
            if result.payload.get("user_id") != user_id:
                return False
            
            # Update with deleted_at timestamp
            result.payload["deleted_at"] = datetime.now(UTC).isoformat()
            
            # Re-upsert with updated payload
            await self.qdrant.upsert(
                collection_name=QdrantVectorStore.CONVERSATION_HISTORY,
                point_id=message_id,
                vector=result.vector or [0.0] * 1536,
                payload=result.payload,
            )
            
            return True
        except Exception:
            return False
    
    async def feedback_message(
        self,
        user_id: str,
        message_id: str,
        rating: FeedbackRating,
        comment: Optional[str] = None,
    ) -> bool:
        """Add feedback to a specific message.
        
        Args:
            user_id: User identifier
            message_id: Message identifier
            rating: Feedback rating
            comment: Optional comment
            
        Returns:
            True if feedback added successfully
        """
        try:
            # Get the message
            result = await self.qdrant.get_by_id(
                collection_name=QdrantVectorStore.CONVERSATION_HISTORY,
                point_id=message_id,
            )
            
            if not result:
                return False
            
            # Verify user ownership
            if result.payload.get("user_id") != user_id:
                return False
            
            # Update with feedback
            result.payload["feedback"] = datetime.now(UTC).isoformat()
            result.payload["feedback_rating"] = rating.value
            result.payload["feedback_comment"] = comment
            
            # Re-upsert with updated payload
            await self.qdrant.upsert(
                collection_name=QdrantVectorStore.CONVERSATION_HISTORY,
                point_id=message_id,
                vector=result.vector or [0.0] * 1536,
                payload=result.payload,
            )
            
            return True
        except Exception:
            return False
    
    async def feedback_conversation(
        self,
        user_id: str,
        session_id: str,
        rating: FeedbackRating,
        comment: Optional[str] = None,
    ) -> int:
        """Add feedback to an entire conversation session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            rating: Feedback rating
            comment: Optional comment
            
        Returns:
            Number of messages updated with feedback
        """
        # Get all messages in session
        messages = await self.get_conversation_history(
            user_id=user_id,
            session_id=session_id,
            limit=1000,
            include_deleted=False,
        )
        
        # Add feedback to each message
        updated_count = 0
        for message in messages:
            message_id = message["message_id"]
            success = await self.feedback_message(
                user_id=user_id,
                message_id=message_id,
                rating=rating,
                comment=comment,
            )
            if success:
                updated_count += 1
        
        return updated_count


# Singleton instance
_conversation_history: Optional[ConversationHistory] = None


def get_conversation_history() -> ConversationHistory:
    """Get or create the conversation history singleton."""
    global _conversation_history
    if _conversation_history is None:
        _conversation_history = ConversationHistory()
    return _conversation_history


def reset_conversation_history():
    """Reset the conversation history singleton (for testing)."""
    global _conversation_history
    _conversation_history = None
