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
