"""Conversation history management using Metis's own Postgres database (db-metis).

Schema follows the "communication" domain documented in
dbml/communication.md (conversations, chat_messages), with a few
Metis-specific extensions: `embedding`/`metadata` columns on chat_messages,
and a separate chat_message_feedback table (feedback is sparse and added
after the message already exists, so it doesn't belong as nullable columns
on every row).

The `embedding` column is reserved for future semantic recall but is not
populated today — no feature consumes it, so we avoid the OpenAI embedding
call (latency + cost) and the pgvector extension dependency. Re-adding it
later is a localized change: `CREATE EXTENSION vector`, alter the column
type to `vector(1536)`, and call `embed_query` in `save_message`.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
from enum import Enum
from metis.storage.pool import DatabasePool
import json
import uuid


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


_conversation_db_pool: Optional[DatabasePool] = None


def set_conversation_db_pool(pool: DatabasePool) -> None:
    """Set the database pool for conversation history (db-metis)."""
    global _conversation_db_pool
    _conversation_db_pool = pool


def _get_conversation_db_pool() -> DatabasePool:
    if _conversation_db_pool is None:
        raise RuntimeError("Conversation DB pool not initialized. Call set_conversation_db_pool() first.")
    return _conversation_db_pool


def _parse_metadata(value: Any) -> Dict[str, Any]:
    """asyncpg returns jsonb as raw text (no codec registered); decode it."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return json.loads(value)


class ConversationHistory:
    """Conversation history manager backed by Metis's own Postgres (db-metis)."""

    def __init__(self, pool: Optional[DatabasePool] = None):
        """Initialize conversation history manager.

        Args:
            pool: Conversation DB pool. If None, uses the singleton set via
                `set_conversation_db_pool()` (done in app/main.py's lifespan).
        """
        self.pool = pool or _get_conversation_db_pool()

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
            session_id: Session identifier (maps to conversations.id)
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Additional metadata (symbol, timeframe, pipeline_summary, etc.)

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        await self.pool.execute(
            """
            INSERT INTO conversations (id, user_id, created_by, created_at, updated_at)
            VALUES ($1, $2, $2, $3, $3)
            ON CONFLICT (id) DO UPDATE SET updated_at = EXCLUDED.updated_at
            """,
            session_id, user_id, now,
        )
        await self.pool.execute(
            """
            INSERT INTO chat_messages
                (id, conversation_id, user_id, role, content, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $7)
            """,
            message_id, session_id, user_id, role.value, content,
            json.dumps(metadata or {}), now,
        )

        return message_id

    async def get_conversation_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 20,
        include_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a specific session, oldest first.

        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            include_deleted: Whether to include soft-deleted messages

        Returns:
            List of messages ordered by timestamp (oldest first)
        """
        deleted_clause = "" if include_deleted else "AND m.deleted_at IS NULL"
        rows = await self.pool.fetch(
            f"""
            SELECT * FROM (
                SELECT m.id, m.role, m.content, m.created_at, m.metadata,
                       f.rating AS feedback_rating, f.comment AS feedback_comment,
                       f.created_at AS feedback_at
                FROM chat_messages m
                LEFT JOIN chat_message_feedback f ON f.message_id = m.id
                WHERE m.conversation_id = $1 AND m.user_id = $2 {deleted_clause}
                ORDER BY m.created_at DESC
                LIMIT $3
            ) sub
            ORDER BY sub.created_at ASC
            """,
            session_id, user_id, limit,
        )
        return [self._row_to_message(r) for r in rows]

    async def get_global_context(
        self,
        user_id: str,
        limit: int = 10,
        exclude_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get global context across all sessions, newest first.

        Args:
            user_id: User identifier
            limit: Maximum number of messages to retrieve
            exclude_session_id: Optional session ID to exclude (current session)

        Returns:
            List of recent messages across all sessions
        """
        if exclude_session_id:
            rows = await self.pool.fetch(
                """
                SELECT m.id, m.conversation_id, m.role, m.content, m.created_at, m.metadata
                FROM chat_messages m
                WHERE m.user_id = $1 AND m.deleted_at IS NULL AND m.conversation_id != $2
                ORDER BY m.created_at DESC
                LIMIT $3
                """,
                user_id, exclude_session_id, limit,
            )
        else:
            rows = await self.pool.fetch(
                """
                SELECT m.id, m.conversation_id, m.role, m.content, m.created_at, m.metadata
                FROM chat_messages m
                WHERE m.user_id = $1 AND m.deleted_at IS NULL
                ORDER BY m.created_at DESC
                LIMIT $2
                """,
                user_id, limit,
            )

        return [
            {
                "message_id": str(r["id"]),
                "session_id": r["conversation_id"],
                "role": r["role"],
                "content": r["content"],
                "timestamp": r["created_at"].isoformat(),
                "metadata": _parse_metadata(r["metadata"]),
            }
            for r in rows
        ]

    async def get_conversation_sessions(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get all conversation sessions for a user, newest first.

        Args:
            user_id: User identifier
            limit: Maximum number of sessions to retrieve

        Returns:
            List of sessions with metadata
        """
        rows = await self.pool.fetch(
            """
            SELECT
                c.id AS session_id,
                COUNT(m.id) AS message_count,
                MIN(m.created_at) AS first_message,
                MAX(m.created_at) AS last_message,
                BOOL_OR(f.message_id IS NOT NULL) AS has_feedback
            FROM conversations c
            LEFT JOIN chat_messages m ON m.conversation_id = c.id AND m.deleted_at IS NULL
            LEFT JOIN chat_message_feedback f ON f.message_id = m.id
            WHERE c.user_id = $1 AND c.deleted_at IS NULL
            GROUP BY c.id
            HAVING COUNT(m.id) > 0
            ORDER BY MAX(m.created_at) DESC
            LIMIT $2
            """,
            user_id, limit,
        )

        return [
            {
                "session_id": r["session_id"],
                "message_count": r["message_count"],
                "first_message": r["first_message"].isoformat() if r["first_message"] else None,
                "last_message": r["last_message"].isoformat() if r["last_message"] else None,
                "has_feedback": bool(r["has_feedback"]),
            }
            for r in rows
        ]

    async def soft_delete_conversation(self, user_id: str, session_id: str) -> int:
        """Soft delete an entire conversation session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Number of messages deleted
        """
        now = datetime.now(UTC)
        await self.pool.execute(
            "UPDATE conversations SET deleted_at = $1 WHERE id = $2 AND user_id = $3 AND deleted_at IS NULL",
            now, session_id, user_id,
        )
        rows = await self.pool.fetch(
            """
            UPDATE chat_messages SET deleted_at = $1
            WHERE conversation_id = $2 AND user_id = $3 AND deleted_at IS NULL
            RETURNING id
            """,
            now, session_id, user_id,
        )
        return len(rows)

    async def soft_delete_message(self, user_id: str, message_id: str) -> bool:
        """Soft delete a specific message.

        Args:
            user_id: User identifier
            message_id: Message identifier

        Returns:
            True if deleted successfully
        """
        now = datetime.now(UTC)
        row = await self.pool.fetchrow(
            """
            UPDATE chat_messages SET deleted_at = $1
            WHERE id = $2 AND user_id = $3 AND deleted_at IS NULL
            RETURNING id
            """,
            now, message_id, user_id,
        )
        return row is not None

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
        owner = await self.pool.fetchval(
            "SELECT user_id FROM chat_messages WHERE id = $1 AND deleted_at IS NULL",
            message_id,
        )
        if owner is None or owner != user_id:
            return False

        await self.pool.execute(
            """
            INSERT INTO chat_message_feedback (message_id, rating, comment, created_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (message_id) DO UPDATE
            SET rating = EXCLUDED.rating, comment = EXCLUDED.comment, created_at = EXCLUDED.created_at
            """,
            message_id, rating.value, comment, datetime.now(UTC),
        )
        return True

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
        messages = await self.get_conversation_history(
            user_id=user_id,
            session_id=session_id,
            limit=1000,
            include_deleted=False,
        )

        updated_count = 0
        for message in messages:
            success = await self.feedback_message(
                user_id=user_id,
                message_id=message["message_id"],
                rating=rating,
                comment=comment,
            )
            if success:
                updated_count += 1

        return updated_count

    @staticmethod
    def _row_to_message(row) -> Dict[str, Any]:
        return {
            "message_id": str(row["id"]),
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["created_at"].isoformat(),
            "metadata": _parse_metadata(row["metadata"]),
            "feedback": row["feedback_at"].isoformat() if row["feedback_at"] else None,
            "feedback_rating": row["feedback_rating"],
            "feedback_comment": row["feedback_comment"],
        }


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
