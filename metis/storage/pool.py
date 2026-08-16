"""Database connection pool with retry logic."""
import asyncio
import random
import time
from contextlib import asynccontextmanager
from typing import Any, Callable
import asyncpg
from asyncpg import Pool, Connection


class DatabasePool:
    """Database connection pool with retry logic and health checks."""
    
    def __init__(self, pool: Pool):
        self.pool = pool
    
    @classmethod
    async def create(
        cls,
        dsn: str,
        min_size: int = 2,
        max_size: int = 10,
        command_timeout: float = 60.0,
    ) -> "DatabasePool":
        """Create a new database connection pool.

        Args:
            dsn: Database connection string
            min_size: Minimum pool size
            max_size: Maximum pool size
            command_timeout: Default command timeout in seconds

        Returns:
            DatabasePool instance
        """
        async def _init_conn(conn: Connection) -> None:
            """Register pgvector codec so list[float] ↔ vector(1536) works.

            Uses pgvector.asyncpg.register_vector if available; otherwise
            falls back to a manual codec that converts list[float] to the
            pgvector text format on the wire.
            """
            try:
                from pgvector.asyncpg import register_vector
                await register_vector(conn)
            except ImportError:
                # Manual fallback: register a codec for the 'vector' type
                # that serializes list[float] → '[v1,v2,...]' text format.
                async def _encode(value):
                    if value is None:
                        return None
                    return '[' + ','.join(str(float(v)) for v in value) + ']'

                async def _decode(value):
                    if value is None:
                        return None
                    # value comes as a string like '[v1,v2,...]'
                    if isinstance(value, str):
                        return [float(x) for x in value.strip('[]').split(',') if x]
                    return value

                await conn.set_type_codec(
                    'vector',
                    encoder=_encode,
                    decoder=_decode,
                    schema='pg_catalog',
                    format='text',
                )
            except Exception:
                # pgvector extension may not be installed on this DB;
                # embedding storage will fail at runtime but other queries work.
                pass

        pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            command_timeout=command_timeout,
            init=_init_conn,
        )

        # Test connection
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        return cls(pool)
    
    async def close(self) -> None:
        """Close the connection pool."""
        await self.pool.close()
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        async with self.pool.acquire() as conn:
            yield conn
    
    async def execute(self, query: str, *args: Any) -> str:
        """Execute a SQL query with retry logic.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            Execution status
        """
        return await self._retry_execute(
            lambda: self.pool.execute(query, *args)
        )
    
    async def executemany(self, command: str, args: list[tuple[Any, ...]]) -> None:
        """Execute a command multiple times with different arguments.
        
        Args:
            command: SQL command
            args: List of argument tuples
        """
        await self._retry_execute(
            lambda: self.pool.executemany(command, args)
        )
    
    async def fetch(self, query: str, *args: Any, timeout: float | None = None) -> list[asyncpg.Record]:
        """Fetch rows from a query.
        
        Args:
            query: SQL query
            *args: Query parameters
            timeout: Query timeout
            
        Returns:
            List of records
        """
        return await self._retry_execute(
            lambda: self.pool.fetch(query, *args, timeout=timeout)
        )
    
    async def fetchval(self, query: str, *args: Any, column: int = 0, timeout: float | None = None) -> Any:
        """Fetch a single value from a query.
        
        Args:
            query: SQL query
            *args: Query parameters
            column: Column index to fetch
            timeout: Query timeout
            
        Returns:
            Single value
        """
        return await self._retry_execute(
            lambda: self.pool.fetchval(query, *args, column=column, timeout=timeout)
        )
    
    async def fetchrow(self, query: str, *args: Any, timeout: float | None = None) -> asyncpg.Record | None:
        """Fetch a single row from a query.
        
        Args:
            query: SQL query
            *args: Query parameters
            timeout: Query timeout
            
        Returns:
            Single record or None
        """
        return await self._retry_execute(
            lambda: self.pool.fetchrow(query, *args, timeout=timeout)
        )
    
    async def transaction(self):
        """Execute operations within a transaction."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    # Retry configuration
    RETRY_MAX_ATTEMPTS = 5
    RETRY_BASE_DELAY = 0.2  # 200ms
    RETRY_MAX_DELAY = 15.0  # 15 seconds
    TRANSIENT_MAX_ELAPSED = 600.0  # 10 minutes
    
    def is_transient_error(self, error: Exception) -> bool:
        """Check if an error is transient and should be retried.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is transient
        """
        if error is None:
            return False
        
        error_msg = str(error).lower()
        
        # Network-level problems
        transient_patterns = [
            "failed to connect",
            "connection refused",
            "connection reset",
            "broken pipe",
            "eof",
            "i/o timeout",
            "no route to host",
            "conn closed",
            "closed pool",
            "too many connections",
            "cannot connect now",
            "server closed",
            "database is starting up",
            "terminating connection",
        ]
        
        for pattern in transient_patterns:
            if pattern in error_msg:
                return True
        
        return False
    
    async def _retry_execute(self, fn: Callable) -> Any:
        """Execute a function with retry logic.
        
        Args:
            fn: Async function to execute
            
        Returns:
            Function result
            
        Raises:
            Last error if all retries fail
        """
        last_error = None
        attempt = 0
        delay = self.RETRY_BASE_DELAY
        start_time = time.time()
        transient_cap = start_time + self.TRANSIENT_MAX_ELAPSED
        
        while True:
            try:
                return await fn()
            except Exception as e:
                last_error = e
                attempt += 1
                
                # Check if error is transient
                transient = self.is_transient_error(e)
                
                # Decide whether to keep retrying
                if transient:
                    if time.time() + delay > transient_cap:
                        break
                elif attempt >= self.RETRY_MAX_ATTEMPTS:
                    break
                
                # Sleep with jitter (±25%)
                jitter = random.uniform(-delay / 4, delay / 4)
                wait_time = max(delay + jitter, 0)
                await asyncio.sleep(wait_time)
                
                # Exponential backoff, capped
                delay *= 2
                delay = min(delay, self.RETRY_MAX_DELAY)
        
        raise last_error
    
    async def health_check(self) -> bool:
        """Check database health.
        
        Returns:
            True if database is healthy
        """
        try:
            await self.fetchval("SELECT 1")
            return True
        except Exception:
            return False
    
    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        return {
            "min_size": self.pool.get_min_size(),
            "max_size": self.pool.get_max_size(),
            "size": self.pool.get_size(),
            "idle": self.pool.get_idle_size(),
        }


async def create_pool(
    dsn: str,
    min_size: int = 2,
    max_size: int = 10,
    command_timeout: float = 60.0,
) -> DatabasePool:
    """Create a database connection pool.
    
    Args:
        dsn: Database connection string
        min_size: Minimum pool size
        max_size: Maximum pool size
        command_timeout: Default command timeout in seconds
        
    Returns:
        DatabasePool instance
    """
    return await DatabasePool.create(
        dsn=dsn,
        min_size=min_size,
        max_size=max_size,
        command_timeout=command_timeout,
    )
