"""Embedding generation and semantic recall via pgvector.

Generates embeddings with OpenAI's text-embedding-3-small (1536 dims)
and provides similarity search over chat history.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from metis.config import get_settings
from metis.memory.conversation_history import _get_conversation_db_pool

logger = logging.getLogger(__name__)

# Singleton OpenAI client (reuses connection pool)
_openai_client: Optional[AsyncOpenAI] = None

# Embedding model config
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
MAX_INPUT_CHARS = 8000  # API limit is 8191 tokens; chars are a safe proxy


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        settings = get_settings()
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


async def generate_embedding(text: str) -> list[float]:
    """Generate an embedding for a text string.

    Uses text-embedding-3-small (1536 dims, $0.02/1M tokens).
    Truncates input to 8000 chars to stay within API limits.
    """
    if not text or not text.strip():
        return []

    client = _get_openai_client()
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:MAX_INPUT_CHARS],
    )
    return response.data[0].embedding


async def embed_and_store(message_id: str, content: str) -> None:
    """Generate embedding and update the message row.

    Called as a fire-and-forget task after save_message.
    Fails silently — messages without embeddings are picked up
    by the backfill script.
    """
    try:
        embedding = await generate_embedding(content)
        if not embedding:
            return

        pool = _get_conversation_db_pool()
        # asyncpg handles list[float] → vector(1536) via the pgvector codec
        await pool.execute(
            "UPDATE chat_messages SET embedding = $1 WHERE id = $2",
            embedding,
            message_id,
        )
        logger.debug(f"[embedding] Stored embedding for message {message_id}")
    except Exception as e:
        logger.warning(f"[embedding] Failed to embed message {message_id}: {e}")


async def recall_similar_messages(
    query: str,
    user_id: str,
    session_id: str = "",
    limit: int = 3,
    threshold: float = 0.75,
) -> list[dict]:
    """Find semantically similar messages from previous conversations.

    Excludes the current session to avoid duplicating messages that
    are already in the context window.

    Args:
        query: The search query (user's current message)
        user_id: User ID to filter by
        session_id: Current session ID (excluded from search)
        limit: Max number of results
        threshold: Minimum cosine similarity (0-1)

    Returns:
        List of {content, role, created_at, similarity} dicts
    """
    embedding = await generate_embedding(query)
    if not embedding:
        return []

    pool = _get_conversation_db_pool()

    if session_id:
        rows = await pool.fetch(
            """
            SELECT content, role, created_at,
                   1 - (embedding <=> $1) AS similarity
            FROM chat_messages
            WHERE user_id = $2
              AND embedding IS NOT NULL
              AND conversation_id != $3
              AND deleted_at IS NULL
            ORDER BY embedding <=> $1
            LIMIT $4
            """,
            embedding, user_id, session_id, limit * 2,  # fetch more, filter by threshold
        )
    else:
        rows = await pool.fetch(
            """
            SELECT content, role, created_at,
                   1 - (embedding <=> $1) AS similarity
            FROM chat_messages
            WHERE user_id = $2
              AND embedding IS NOT NULL
              AND deleted_at IS NULL
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            embedding, user_id, limit * 2,
        )

    # Filter by threshold and return top results
    results = []
    for row in rows:
        sim = float(row["similarity"])
        if sim >= threshold:
            results.append({
                "content": row["content"],
                "role": row["role"],
                "created_at": row["created_at"],
                "similarity": sim,
            })
        if len(results) >= limit:
            break

    return results
