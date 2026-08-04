"""Frequent calculation caching and persistence."""
from datetime import datetime, timedelta
from typing import Any
from dataclasses import dataclass
import json

from .types import FeatureResult, IndicatorResult
from ..storage.pool import DatabasePool
from ..storage.models import FrequentCalculation as StorageFrequentCalculation
from ..storage.migrations import run_migrations


@dataclass
class FrequentCalculation:
    """Represents a frequently requested calculation."""
    id: int | None = None
    symbol: str = ""
    interval: str = ""
    calculation_type: str = ""  # 'feature' or 'indicator'
    name: str = ""  # e.g., 'ma_21', 'rsi_14'
    request_count: int = 0
    last_requested_at: datetime | None = None
    is_persisted: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class PersistedCalculation:
    """Represents a persisted calculation result."""
    id: int | None = None
    symbol: str = ""
    interval: str = ""
    calculation_type: str = ""
    name: str = ""
    data: bytes = b""
    expires_at: datetime | None = None
    created_at: datetime | None = None


class FrequentCalculator:
    """Handles frequent calculation caching and persistence.
    
    This class tracks which calculations are frequently requested and persists
    their results to avoid recalculation. It uses an in-memory cache by default
    but can be extended to use a database for persistence.
    """
    
    def __init__(self, db_pool: DatabasePool | None = None):
        """Initialize the FrequentCalculator.
        
        Args:
            db_pool: Database connection pool (optional)
        """
        self.db_pool = db_pool
        self.use_database = db_pool is not None
        
        # Run migrations if database is provided
        if self.db_pool:
            # Run migrations in a separate context to avoid blocking
            import asyncio
            try:
                asyncio.create_task(run_migrations(self.db_pool))
            except RuntimeError:
                # If no event loop, migrations will be run on first use
                self._migrations_run = False
            else:
                self._migrations_run = True
        else:
            self._migrations_run = False
        
        # In-memory cache for frequent calculations
        self._frequent_calculations: dict[tuple[str, str, str, str], FrequentCalculation] = {}
        
        # In-memory cache for persisted results
        self._persisted_results: dict[tuple[str, str, str, str], tuple[bytes, datetime]] = {}
    
    async def _ensure_migrations(self):
        """Ensure database migrations are run."""
        if self.use_database and self.db_pool and not self._migrations_run:
            await run_migrations(self.db_pool)
            self._migrations_run = True
    
    async def check_frequent(
        self,
        symbol: str,
        interval: str,
        calc_type: str,
        name: str
    ) -> bool:
        """Check if a calculation is frequently requested and should be persisted.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            calc_type: Calculation type ('feature' or 'indicator')
            name: Calculation name
            
        Returns:
            True if the calculation is marked as frequent/persisted
        """
        key = (symbol, interval, calc_type, name)
        
        if self.use_database and self.db_pool:
            await self._ensure_migrations()
            sql = """
                SELECT EXISTS(
                    SELECT 1 FROM frequent_calculations 
                    WHERE symbol = $1 AND interval = $2 AND calculation_type = $3 AND name = $4
                )
            """
            exists = await self.db_pool.fetchval(sql, symbol, interval, calc_type, name)
            if exists:
                return True
        
        freq_calc = self._frequent_calculations.get(key)
        return freq_calc is not None and freq_calc.is_persisted
    
    async def get_persisted_result(
        self,
        symbol: str,
        interval: str,
        calc_type: str,
        name: str
    ) -> bytes | None:
        """Retrieve a persisted calculation result.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            calc_type: Calculation type ('feature' or 'indicator')
            name: Calculation name
            
        Returns:
            Persisted data as bytes, or None if not found or expired
        """
        key = (symbol, interval, calc_type, name)
        
        if self.use_database and self.db_pool:
            await self._ensure_migrations()
            sql = """
                SELECT data FROM persisted_calculations 
                WHERE symbol = $1 AND interval = $2 AND calculation_type = $3 AND name = $4 
                AND expires_at > NOW()
                ORDER BY created_at DESC LIMIT 1
            """
            result = await self.db_pool.fetchrow(sql, symbol, interval, calc_type, name)
            if result:
                return result["data"]
        
        cached = self._persisted_results.get(key)
        if cached:
            data, expires_at = cached
            if expires_at > datetime.now():
                return data
            else:
                # Remove expired entry
                del self._persisted_results[key]
        
        return None
    
    async def persist_result(
        self,
        symbol: str,
        interval: str,
        calc_type: str,
        name: str,
        data: bytes,
        ttl: timedelta = timedelta(hours=1)
    ) -> None:
        """Save a calculation result for future use.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            calc_type: Calculation type ('feature' or 'indicator')
            name: Calculation name
            data: Calculation result data (serialized)
            ttl: Time to live for the cached result
        """
        key = (symbol, interval, calc_type, name)
        expires_at = datetime.now() + ttl
        
        if self.use_database and self.db_pool:
            await self._ensure_migrations()
            sql = """
                INSERT INTO persisted_calculations (symbol, interval, calculation_type, name, data, expires_at, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                ON CONFLICT (symbol, interval, calculation_type, name) 
                DO UPDATE SET data = $5, expires_at = $6, created_at = NOW()
            """
            await self.db_pool.execute(sql, symbol, interval, calc_type, name, data, expires_at)
        
        self._persisted_results[key] = (data, expires_at)
    
    async def increment_request_count(
        self,
        symbol: str,
        interval: str,
        calc_type: str,
        name: str
    ) -> None:
        """Increment the request count for a calculation.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            calc_type: Calculation type ('feature' or 'indicator')
            name: Calculation name
        """
        key = (symbol, interval, calc_type, name)
        now = datetime.now()
        
        if self.use_database and self.db_pool:
            await self._ensure_migrations()
            sql = """
                INSERT INTO frequent_calculations (symbol, interval, calculation_type, name, request_count, last_requested_at, created_at, updated_at)
                VALUES ($1, $2, $3, $4, 1, NOW(), NOW(), NOW())
                ON CONFLICT (symbol, interval, calculation_type, name) 
                DO UPDATE SET request_count = frequent_calculations.request_count + 1, 
                             last_requested_at = NOW(),
                             updated_at = NOW()
            """
            await self.db_pool.execute(sql, symbol, interval, calc_type, name)
        
        if key in self._frequent_calculations:
            freq_calc = self._frequent_calculations[key]
            freq_calc.request_count += 1
            freq_calc.last_requested_at = now
            freq_calc.updated_at = now
        else:
            self._frequent_calculations[key] = FrequentCalculation(
                symbol=symbol,
                interval=interval,
                calculation_type=calc_type,
                name=name,
                request_count=1,
                last_requested_at=now,
                is_persisted=False,
                created_at=now,
                updated_at=now
            )
    
    async def evaluate_frequent_calculations(self, threshold: int = 10) -> None:
        """Evaluate which calculations should be persisted based on request count.
        
        Args:
            threshold: Minimum request count to mark as frequent
        """
        if self.use_database and self.db_pool:
            await self._ensure_migrations()
            # Get IDs of calculations to mark as persisted
            sql = """
                SELECT id FROM frequent_calculations 
                WHERE request_count >= $1 AND is_persisted = FALSE
                ORDER BY request_count DESC
            """
            rows = await self.db_pool.fetch(sql, threshold)
            ids = [row["id"] for row in rows]
            
            if ids:
                update_sql = """
                    UPDATE frequent_calculations 
                    SET is_persisted = TRUE, updated_at = NOW() 
                    WHERE id = ANY($1)
                """
                await self.db_pool.execute(update_sql, ids)
        
        now = datetime.now()
        for key, freq_calc in self._frequent_calculations.items():
            if freq_calc.request_count >= threshold and not freq_calc.is_persisted:
                freq_calc.is_persisted = True
                freq_calc.updated_at = now
    
    def serialize_feature_results(self, results: list[FeatureResult]) -> bytes:
        """Serialize feature results to bytes for persistence.
        
        Args:
            results: List of FeatureResult objects
            
        Returns:
            Serialized data as bytes
        """
        data = [
            {
                'name': r.name,
                'value': r.value,
                'timestamp': r.timestamp.isoformat()
            }
            for r in results
        ]
        return json.dumps(data).encode('utf-8')
    
    def deserialize_feature_results(self, data: bytes) -> list[FeatureResult]:
        """Deserialize bytes to feature results.
        
        Args:
            data: Serialized data bytes
            
        Returns:
            List of FeatureResult objects
        """
        parsed = json.loads(data.decode('utf-8'))
        return [
            FeatureResult(
                name=item['name'],
                value=item['value'],
                timestamp=datetime.fromisoformat(item['timestamp'])
            )
            for item in parsed
        ]
    
    def serialize_indicator_results(self, results: list[IndicatorResult]) -> bytes:
        """Serialize indicator results to bytes for persistence.
        
        Args:
            results: List of IndicatorResult objects
            
        Returns:
            Serialized data as bytes
        """
        data = [
            {
                'name': r.name,
                'value': r.value,
                'timestamp': r.timestamp.isoformat()
            }
            for r in results
        ]
        return json.dumps(data).encode('utf-8')
    
    def deserialize_indicator_results(self, data: bytes) -> list[IndicatorResult]:
        """Deserialize bytes to indicator results.
        
        Args:
            data: Serialized data bytes
            
        Returns:
            List of IndicatorResult objects
        """
        parsed = json.loads(data.decode('utf-8'))
        return [
            IndicatorResult(
                name=item['name'],
                value=item['value'],
                timestamp=datetime.fromisoformat(item['timestamp'])
            )
            for item in parsed
        ]
    
    def get_frequent_calculations(self) -> list[FrequentCalculation]:
        """Get all frequent calculations.
        
        Returns:
            List of FrequentCalculation objects
        """
        return list(self._frequent_calculations.values())
    
    def clear_cache(self) -> None:
        """Clear all in-memory caches."""
        self._frequent_calculations.clear()
        self._persisted_results.clear()


def create_frequent_calculator(db_pool: DatabasePool | None = None) -> FrequentCalculator:
    """Create a new FrequentCalculator instance.
    
    Args:
        db_pool: Database connection pool (optional)
        
    Returns:
        FrequentCalculator instance
    """
    return FrequentCalculator(db_pool=db_pool)
