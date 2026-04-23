"""Database migration scripts for calculator tables."""

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


async def run_migrations(pool) -> None:
    """Run all database migrations.
    
    Args:
        pool: Database connection pool
    """
    await pool.execute(CREATE_FREQUENT_CALCULATIONS_TABLE)
    await pool.execute(CREATE_PERSISTED_CALCULATIONS_TABLE)
