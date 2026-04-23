"""Cache layer for frequent calculations."""
from datetime import datetime, timedelta
from typing import Any, Optional
import json

from .pool import DatabasePool
from .models import FrequentCalculation, PersistedCalculation


class CalculationCache:
    """Cache for frequent calculations using database-backed storage."""
    
    def __init__(self, pool: DatabasePool, ttl_hours: int = 24):
        self.pool = pool
        self.ttl = timedelta(hours=ttl_hours)
    
    async def get_cached_result(
        self,
        symbol: str,
        interval: str,
        calculation_type: str,
        name: str
    ) -> Optional[Any]:
        """Get cached calculation result.
        
        Uses covering index idx_persisted_calculations_cache_lookup for index-only scan.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            calculation_type: 'feature' or 'indicator'
            name: Calculation name (e.g., 'rsi_14', 'ma_21')
            
        Returns:
            Cached result or None if not found or expired
        """
        # Use covering index idx_persisted_calculations_cache_lookup
        sql = """
            SELECT data, created_at
            FROM persisted_calculations
            WHERE symbol = $1 AND interval = $2 
                AND calculation_type = $3 AND name = $4
                AND expires_at > NOW()
            LIMIT 1
        """
        
        row = await self.pool.fetchrow(sql, symbol, interval, calculation_type, name)
        if row:
            try:
                return json.loads(row['data'])
            except (json.JSONDecodeError, TypeError):
                return None
        return None
    
    async def cache_result(
        self,
        symbol: str,
        interval: str,
        calculation_type: str,
        name: str,
        result: Any,
    ) -> None:
        """Cache a calculation result.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            calculation_type: 'feature' or 'indicator'
            name: Calculation name
            result: Result to cache (must be JSON-serializable)
        """
        expires_at = datetime.now() + self.ttl
        data = json.dumps(result)
        
        sql = """
            INSERT INTO persisted_calculations
            (symbol, interval, calculation_type, name, data, expires_at, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (symbol, interval, calculation_type, name)
            DO UPDATE SET 
                data = EXCLUDED.data,
                expires_at = EXCLUDED.expires_at,
                created_at = NOW()
        """
        
        await self.pool.execute(sql, symbol, interval, calculation_type, name, data, expires_at)
    
    async def increment_request_count(
        self,
        symbol: str,
        interval: str,
        calculation_type: str,
        name: str
    ) -> None:
        """Track request count for a calculation.
        
        Uses covering index idx_frequent_calculations_lookup_covering for index-only scan on lookup.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            calculation_type: 'feature' or 'indicator'
            name: Calculation name
        """
        sql = """
            INSERT INTO frequent_calculations
            (symbol, interval, calculation_type, name, request_count, last_requested_at, is_persisted, created_at, updated_at)
            VALUES ($1, $2, $3, $4, 1, NOW(), true, NOW(), NOW())
            ON CONFLICT (symbol, interval, calculation_type, name)
            DO UPDATE SET 
                request_count = frequent_calculations.request_count + 1,
                last_requested_at = NOW(),
                updated_at = NOW()
        """
        
        await self.pool.execute(sql, symbol, interval, calculation_type, name)
