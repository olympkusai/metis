"""Database migration scripts for Metis database (db-metis)."""

# ─────────────────────────────────────────────────────────────
# Conversation schema — runs on the same db-metis Postgres. Schema
# follows the "communication" domain documented in the cross-service
# dbml/communication.md, with a few Metis-specific extensions
# (embedding, metadata, feedback table).
#
# pgvector was removed from the critical path: no feature consumes the
# `embedding` column today (see metis/memory/conversation_history.py).
# The column is kept nullable and index-free so existing rows aren't
# broken and re-adding pgvector later is a one-line migration
# (`CREATE EXTENSION vector` + ivfflat index + embed_query call).
# ─────────────────────────────────────────────────────────────

CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id varchar PRIMARY KEY,
    user_id varchar NOT NULL,
    created_by varchar NOT NULL,
    title varchar,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    deleted_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations (user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_by ON conversations (created_by);
CREATE INDEX IF NOT EXISTS idx_conversations_deleted_at ON conversations (deleted_at);
"""

# `embedding`/`metadata` are Metis-specific extensions beyond the canonical
# communication.md shape: `embedding` is reserved for future semantic recall
# over chat history (no consumer yet — kept as a nullable plain column so the
# table shape is stable; pgvector is not required to run this migration).
# `metadata` carries per-message pipeline info (reasoning steps, tools used,
# chain-of-thought) already produced by metis/api/chat.py on every assistant reply.
CREATE_CHAT_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id varchar PRIMARY KEY,
    conversation_id varchar NOT NULL REFERENCES conversations(id),
    user_id varchar NOT NULL,
    role varchar NOT NULL,
    content text NOT NULL,
    embedding float8[],
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    deleted_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat_messages (conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id ON chat_messages (user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id_created_at ON chat_messages (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_deleted_at ON chat_messages (deleted_at);
"""

# Feedback is sparse (most messages never get one) and added after the
# message already exists, so it's normalized into its own table instead of
# nullable columns on chat_messages.
CREATE_CHAT_MESSAGE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS chat_message_feedback (
    message_id varchar PRIMARY KEY REFERENCES chat_messages(id),
    rating varchar NOT NULL,
    comment text,
    created_at timestamptz NOT NULL DEFAULT now()
);
"""

# No Metis feature produces notifications yet — table exists to match the
# canonical communication.md domain shape ahead of that feature landing.
CREATE_NOTIFICATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS notifications (
    id varchar PRIMARY KEY,
    user_id varchar NOT NULL,
    type varchar NOT NULL,
    title varchar NOT NULL,
    message text NOT NULL,
    related_resource_type varchar,
    related_resource_id varchar,
    is_read boolean NOT NULL DEFAULT false,
    read_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    deleted_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications (user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_user_id_is_read ON notifications (user_id, is_read);
CREATE INDEX IF NOT EXISTS idx_notifications_related_resource ON notifications (related_resource_type, related_resource_id);
CREATE INDEX IF NOT EXISTS idx_notifications_deleted_at ON notifications (deleted_at);
"""


async def run_conversation_migrations(pool) -> None:
    """Run migrations for the conversation schema (db-metis).

    Args:
        pool: Database connection pool for db-metis.
    """
    await pool.execute(CREATE_CONVERSATIONS_TABLE)
    await pool.execute(CREATE_CHAT_MESSAGES_TABLE)
    await pool.execute(CREATE_CHAT_MESSAGE_FEEDBACK_TABLE)
    await pool.execute(CREATE_NOTIFICATIONS_TABLE)
