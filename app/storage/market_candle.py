"""Read-only queries for market_candle table.

Metis only reads from market_candle table, never writes.
"""
from datetime import datetime
from typing import List

from .pool import DatabasePool
from .models import MarketCandle


class MarketCandleQueries:
    """Read-only queries for market_candle table."""
    
    def __init__(self, pool: DatabasePool):
        self.pool = pool
    
    async def get_candles(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int | None = None,
    ) -> List[MarketCandle]:
        """Get candles for a symbol and interval within a time range.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of candles to return
            
        Returns:
            List of MarketCandle objects
        """
        sql = """
            SELECT 
                symbol,
                interval,
                open_time,
                close_time,
                open_price,
                high_price,
                low_price,
                close_price,
                base_volume,
                quote_volume,
                trades_count
            FROM market_candles
            WHERE symbol = $1 
                AND interval = $2 
                AND close_time >= $3 
                AND close_time <= $4
            ORDER BY close_time ASC
        """
        
        if limit:
            sql += f" LIMIT {limit}"
        
        rows = await self.pool.fetch(sql, symbol, interval, start_time, end_time)
        return [MarketCandle.from_record(dict(row)) for row in rows]
    
    async def get_latest_candles(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> List[MarketCandle]:
        """Get the latest candles for a symbol and interval.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            limit: Maximum number of candles to return
            
        Returns:
            List of MarketCandle objects
        """
        sql = """
            SELECT 
                symbol,
                interval,
                open_time,
                close_time,
                open_price,
                high_price,
                low_price,
                close_price,
                base_volume,
                quote_volume,
                trades_count
            FROM market_candles
            WHERE symbol = $1 AND interval = $2
            ORDER BY close_time DESC
            LIMIT $3
        """
        
        rows = await self.pool.fetch(sql, symbol, interval, limit)
        return [MarketCandle.from_record(dict(row)) for row in rows]
    
    async def get_candle_by_time(
        self,
        symbol: str,
        interval: str,
        close_time: datetime,
    ) -> MarketCandle | None:
        """Get a specific candle by its close time.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            close_time: Candle close time
            
        Returns:
            MarketCandle object or None if not found
        """
        sql = """
            SELECT 
                symbol,
                interval,
                open_time,
                close_time,
                open_price,
                high_price,
                low_price,
                close_price,
                base_volume,
                quote_volume,
                trades_count
            FROM market_candles
            WHERE symbol = $1 
                AND interval = $2 
                AND close_time = $3
            LIMIT 1
        """
        
        row = await self.pool.fetchrow(sql, symbol, interval, close_time)
        if row:
            return MarketCandle.from_record(dict(row))
        return None
    
    async def count_candles(
        self,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """Count candles for a symbol and interval.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            
        Returns:
            Number of candles
        """
        sql = "SELECT COUNT(*) FROM market_candles WHERE symbol = $1 AND interval = $2"
        args = [symbol, interval]
        
        if start_time:
            sql += " AND close_time >= $3"
            args.append(start_time)
        if end_time:
            idx = len(args) + 1
            sql += f" AND close_time <= ${idx}"
            args.append(end_time)
        
        count = await self.pool.fetchval(sql, *args)
        return int(count) if count else 0
