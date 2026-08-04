"""Database migration scripts for calculator tables."""

# Migration for make_interval function (PostgreSQL interval function)
CREATE_MAKE_INTERVAL_FUNCTION = """
CREATE OR REPLACE FUNCTION make_interval(mins NUMERIC DEFAULT 0, hours NUMERIC DEFAULT 0, days NUMERIC DEFAULT 0)
RETURNS INTERVAL
LANGUAGE SQL
IMMUTABLE
AS $$
    SELECT 
        (days || ' days')::INTERVAL + 
        (hours || ' hours')::INTERVAL + 
        (mins || ' minutes')::INTERVAL
$$;
"""

# Migration for frequent_calculations table
CREATE_FREQUENT_CALCULATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS frequent_calculations (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    calculation_type VARCHAR(20) NOT NULL,
    name VARCHAR(100) NOT NULL,
    request_count INTEGER DEFAULT 0,
    last_requested_at TIMESTAMP,
    is_persisted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (symbol, interval, calculation_type, name)
);

CREATE INDEX IF NOT EXISTS idx_frequent_calculations_lookup 
    ON frequent_calculations (symbol, interval, calculation_type, name);

CREATE INDEX IF NOT EXISTS idx_frequent_calculations_evaluate 
    ON frequent_calculations (request_count) 
    WHERE is_persisted = FALSE;
"""

# Migration for persisted_calculations table
CREATE_PERSISTED_CALCULATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS persisted_calculations (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    calculation_type VARCHAR(20) NOT NULL,
    name VARCHAR(100) NOT NULL,
    data BYTEA NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (symbol, interval, calculation_type, name)
);

CREATE INDEX IF NOT EXISTS idx_persisted_calculations_cache_lookup 
    ON persisted_calculations (symbol, interval, calculation_type, name, expires_at);
"""

# Migration for market_candles table indexes (table is managed externally, Metis only reads)
# These indexes optimize the query patterns used in MarketCandleQueries
CREATE_MARKET_CANDLES_INDEXES = """
-- Covering index for intraday candle lookups (symbol, interval, close_time)
-- Used by get_candles for 1m candles and aggregation queries
CREATE INDEX IF NOT EXISTS idx_market_candles_intraday_lookup 
    ON market_candles (symbol, interval, close_time)
    INCLUDE (open_time, open_price, high_price, low_price, close_price, base_volume, quote_volume, closed, received_at);

-- Partial index for latest closed candles (symbol, interval, close_time DESC)
-- Used by get_latest_candles for faster retrieval of recent closed candles
CREATE INDEX IF NOT EXISTS idx_market_candles_latest 
    ON market_candles (symbol, interval, close_time DESC)
    WHERE closed = true;

-- Index for day-aligned queries (date_trunc on close_time)
-- Used for daily aggregation and time-based filtering
CREATE INDEX IF NOT EXISTS idx_market_candles_date_trunc 
    ON market_candles (symbol, interval, date_trunc('day', close_time));

-- Composite index for symbol/interval filtering
-- General purpose index for all queries filtering by symbol and interval
CREATE INDEX IF NOT EXISTS idx_market_candles_symbol_interval 
    ON market_candles (symbol, interval);
"""


async def run_migrations(pool) -> None:
    """Run all database migrations.

    Args:
        pool: Database connection pool
    """
    await pool.execute(CREATE_MAKE_INTERVAL_FUNCTION)
    await pool.execute(CREATE_FREQUENT_CALCULATIONS_TABLE)
    await pool.execute(CREATE_PERSISTED_CALCULATIONS_TABLE)
    await pool.execute(CREATE_MARKET_CANDLES_INDEXES)


# ─────────────────────────────────────────────────────────────
# Conversation DB (db-metis) — Metis's own database, separate from
# the external k0s Postgres above. Schema follows the "communication"
# domain documented in the cross-service dbml/communication.md, with a
# few Metis-specific extensions (embedding, metadata, feedback table).
# ─────────────────────────────────────────────────────────────

CREATE_PGVECTOR_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS vector;
"""

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
# communication.md shape: embedding has no consumer yet (reserved for future
# semantic recall over chat history), metadata carries per-message pipeline
# info (reasoning steps, tools used, chain-of-thought) already produced by
# app/api/chat.py on every assistant reply.
CREATE_CHAT_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id varchar PRIMARY KEY,
    conversation_id varchar NOT NULL REFERENCES conversations(id),
    user_id varchar NOT NULL,
    role varchar NOT NULL,
    content text NOT NULL,
    embedding vector(1536),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    deleted_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat_messages (conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id ON chat_messages (user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id_created_at ON chat_messages (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_deleted_at ON chat_messages (deleted_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_embedding ON chat_messages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
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
    """Run migrations for Metis's own conversation database (db-metis).

    Args:
        pool: Database connection pool for the conversation DB (not the
            external k0s pool passed to `run_migrations`).
    """
    await pool.execute(CREATE_PGVECTOR_EXTENSION)
    await pool.execute(CREATE_CONVERSATIONS_TABLE)
    await pool.execute(CREATE_CHAT_MESSAGES_TABLE)
    await pool.execute(CREATE_CHAT_MESSAGE_FEEDBACK_TABLE)
    await pool.execute(CREATE_NOTIFICATIONS_TABLE)
