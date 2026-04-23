"""Read-only queries for market_candle table.

Metis only reads from market_candle table, never writes.
"""
from datetime import datetime
from typing import List

from .pool import DatabasePool
from .models import MarketCandle
from app.utils.timing import timed_async


class MarketCandleQueries:
    """Read-only queries for market_candle table."""
    
    def __init__(self, pool: DatabasePool):
        self.pool = pool
    
    @timed_async("DB Query: get_candles")
    async def get_candles(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int | None = None,
    ) -> List[MarketCandle]:
        """Get candles for a symbol and interval within a time range.
        
        Optimized to use:
        - idx_market_candles_intraday_lookup for 1m candles
        - idx_market_candles_date_trunc for day-aligned queries
        - Materialized views for pre-aggregated data when available
        
        Args:
            symbol: Trading symbol
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d, 1D)
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of candles to return
            
        Returns:
            List of MarketCandle objects
        """
        # Normalize interval to lowercase
        interval = interval.lower()
        # Check if interval is 1m or if we need aggregation
        if interval == "1m":
            # Use covering index idx_market_candles_intraday_lookup for index-only scan
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
                    closed,
                    received_at
                FROM market_candles
                WHERE symbol = $1 
                    AND interval = $2 
                    AND close_time >= $3 
                    AND close_time <= $4
                ORDER BY close_time ASC
            """
            if limit:
                sql += " LIMIT $5"
                rows = await self.pool.fetch(sql, symbol, interval, start_time, end_time, limit)
            else:
                rows = await self.pool.fetch(sql, symbol, interval, start_time, end_time)
        else:
            # Try materialized view for hourly/daily data if available
            if interval == "1h":
                sql = """
                    SELECT 
                        symbol,
                        '1h' as interval,
                        hour as open_time,
                        hour + interval '1 hour' as close_time,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        base_volume,
                        quote_volume,
                        true as closed,
                        NOW() as received_at
                    FROM mv_hourly_ohlcv
                    WHERE symbol = $1 AND interval = '1h'
                        AND hour >= $2 AND hour < $3
                    ORDER BY hour ASC
                """
                if limit:
                    sql += " LIMIT $4"
                    rows = await self.pool.fetch(sql, symbol, start_time, end_time, limit)
                else:
                    rows = await self.pool.fetch(sql, symbol, start_time, end_time)
                
                # Fallback to aggregation if MV is empty or not available
                if not rows:
                    aggregation_sql = self._get_aggregation_sql(interval)
                    sql = f"""
                        {aggregation_sql}
                        WHERE symbol = $1 AND interval = '1m'
                            AND close_time >= $2 
                            AND close_time <= $3
                        GROUP BY symbol, time_bucket
                        ORDER BY time_bucket ASC
                    """
                    if limit:
                        sql += " LIMIT $4"
                        rows = await self.pool.fetch(sql, symbol, start_time, end_time, limit)
                    else:
                        rows = await self.pool.fetch(sql, symbol, start_time, end_time)
            elif interval == "1d" or interval == "1D":
                sql = """
                    SELECT 
                        symbol,
                        '1d' as interval,
                        day as open_time,
                        day + interval '1 day' as close_time,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        base_volume,
                        quote_volume,
                        true as closed,
                        NOW() as received_at
                    FROM mv_daily_ohlcv
                    WHERE symbol = $1 AND interval = '1d'
                        AND day >= $2 AND day < $3
                    ORDER BY day ASC
                """
                if limit:
                    sql += " LIMIT $4"
                    rows = await self.pool.fetch(sql, symbol, start_time, end_time, limit)
                else:
                    rows = await self.pool.fetch(sql, symbol, start_time, end_time)
                
                # Fallback to aggregation if MV is empty or not available
                if not rows:
                    aggregation_sql = self._get_aggregation_sql(interval)
                    sql = f"""
                        {aggregation_sql}
                        WHERE symbol = $1 AND interval = '1m'
                            AND close_time >= $2 
                            AND close_time <= $3
                        GROUP BY symbol, time_bucket
                        ORDER BY time_bucket ASC
                    """
                    if limit:
                        sql += " LIMIT $4"
                        rows = await self.pool.fetch(sql, symbol, start_time, end_time, limit)
                    else:
                        rows = await self.pool.fetch(sql, symbol, start_time, end_time)
            else:
                # Aggregate from 1m using SQL for other intervals
                aggregation_sql = self._get_aggregation_sql(interval)
                sql = f"""
                    {aggregation_sql}
                    WHERE symbol = $1 AND interval = '1m'
                        AND close_time >= $2 
                        AND close_time <= $3
                    GROUP BY symbol, time_bucket
                    ORDER BY time_bucket ASC
                """
                if limit:
                    sql += " LIMIT $4"
                    rows = await self.pool.fetch(sql, symbol, start_time, end_time, limit)
                else:
                    rows = await self.pool.fetch(sql, symbol, start_time, end_time)
        
        return [MarketCandle.from_record(dict(row)) for row in rows]
    
    @timed_async("DB Query: get_latest_candles")
    async def get_latest_candles(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> List[MarketCandle]:
        """Get the latest candles for a symbol and interval.
        
        Optimized to use:
        - idx_market_candles_latest for closed candles
        - Materialized views for pre-aggregated data when available
        
        Args:
            symbol: Trading symbol
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d, 1D)
            limit: Maximum number of candles to return
            
        Returns:
            List of MarketCandle objects
        """
        # Normalize interval to lowercase
        interval = interval.lower()
        # Check if interval is 1m or if we need aggregation
        if interval == "1m":
            # Use partial index idx_market_candles_latest for closed candles only
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
                    closed,
                    received_at
                FROM market_candles
                WHERE symbol = $1 AND interval = $2 AND closed = true
                ORDER BY close_time DESC
                LIMIT $3
            """
            rows = await self.pool.fetch(sql, symbol, interval, limit)
            
            # If no closed candles, try all candles
            if not rows:
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
                        closed,
                        received_at
                    FROM market_candles
                    WHERE symbol = $1 AND interval = $2
                    ORDER BY close_time DESC
                    LIMIT $3
                """
                rows = await self.pool.fetch(sql, symbol, interval, limit)
        else:
            # Try materialized view for hourly/daily data if available
            if interval == "1h":
                sql = """
                    SELECT 
                        symbol,
                        '1h' as interval,
                        hour as open_time,
                        hour + interval '1 hour' as close_time,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        base_volume,
                        quote_volume,
                        true as closed,
                        NOW() as received_at
                    FROM mv_hourly_ohlcv
                    WHERE symbol = $1 AND interval = '1h'
                    ORDER BY hour DESC
                    LIMIT $2
                """
                rows = await self.pool.fetch(sql, symbol, limit)
                
                # Fallback to aggregation if MV is empty or not available
                if not rows:
                    aggregation_sql = self._get_aggregation_sql(interval)
                    sql = f"""
                        {aggregation_sql}
                        WHERE symbol = $1 AND interval = '1m' AND closed = true
                        GROUP BY symbol, time_bucket
                        ORDER BY time_bucket DESC
                        LIMIT $2
                    """
                    rows = await self.pool.fetch(sql, symbol, limit)
                    
                    if not rows:
                        sql = f"""
                            {aggregation_sql}
                            WHERE symbol = $1 AND interval = '1m'
                            GROUP BY symbol, time_bucket
                            ORDER BY time_bucket DESC
                            LIMIT $2
                        """
                        rows = await self.pool.fetch(sql, symbol, limit)
            elif interval == "1d":
                sql = """
                    SELECT 
                        symbol,
                        '1d' as interval,
                        day as open_time,
                        day + interval '1 day' as close_time,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        base_volume,
                        quote_volume,
                        true as closed,
                        NOW() as received_at
                    FROM mv_daily_ohlcv
                    WHERE symbol = $1 AND interval = '1d'
                    ORDER BY day DESC
                    LIMIT $2
                """
                rows = await self.pool.fetch(sql, symbol, limit)
                
                # Fallback to aggregation if MV is empty or not available
                if not rows:
                    aggregation_sql = self._get_aggregation_sql(interval)
                    sql = f"""
                        {aggregation_sql}
                        WHERE symbol = $1 AND interval = '1m' AND closed = true
                        GROUP BY symbol, time_bucket
                        ORDER BY time_bucket DESC
                        LIMIT $2
                    """
                    rows = await self.pool.fetch(sql, symbol, limit)
                    
                    if not rows:
                        sql = f"""
                            {aggregation_sql}
                            WHERE symbol = $1 AND interval = '1m'
                            GROUP BY symbol, time_bucket
                            ORDER BY time_bucket DESC
                            LIMIT $2
                        """
                        rows = await self.pool.fetch(sql, symbol, limit)
            else:
                # Aggregate from 1m using SQL for other intervals
                aggregation_sql = self._get_aggregation_sql(interval)
                sql = f"""
                    {aggregation_sql}
                    WHERE symbol = $1 AND interval = '1m' AND closed = true
                    GROUP BY symbol, time_bucket
                    ORDER BY time_bucket DESC
                    LIMIT $2
                """
                rows = await self.pool.fetch(sql, symbol, limit)
                
                # If no closed candles, try all candles
                if not rows:
                    sql = f"""
                        {aggregation_sql}
                        WHERE symbol = $1 AND interval = '1m'
                        GROUP BY symbol, time_bucket
                        ORDER BY time_bucket DESC
                        LIMIT $2
                    """
                    rows = await self.pool.fetch(sql, symbol, limit)
        
        return [MarketCandle.from_record(dict(row)) for row in rows]
    
    def _get_aggregation_sql(self, interval: str) -> str:
        """Generate SQL aggregation query for the given interval.
        
        Optimized to use covering index idx_market_candles_intraday_lookup
        for index-only scan when possible.
        """
        # Determine time bucket and multiplier
        if interval == "5m":
            bucket = "date_trunc('hour', close_time) + make_interval(mins => floor(extract(minute FROM close_time) / 5) * 5)"
        elif interval == "15m":
            bucket = "date_trunc('hour', close_time) + make_interval(mins => floor(extract(minute FROM close_time) / 15) * 15)"
        elif interval == "1h":
            bucket = "date_trunc('hour', close_time)"
        elif interval == "4h":
            bucket = "date_trunc('hour', close_time) + make_interval(hours => floor(extract(hour FROM close_time) / 4) * 4)"
        elif interval == "1d" or interval == "1D":
            bucket = "date_trunc('day', close_time)"
        else:
            bucket = "date_trunc('hour', close_time)"  # Default to 1h
        
        return f"""
            SELECT 
                symbol,
                '{interval}' as interval,
                MIN(open_time) as open_time,
                MAX(close_time) as close_time,
                MIN(open_price) as open_price,
                MAX(high_price) as high_price,
                MIN(low_price) as low_price,
                MAX(close_price) as close_price,
                SUM(base_volume) as base_volume,
                SUM(quote_volume) as quote_volume,
                BOOL_OR(closed) as closed,
                MAX(received_at) as received_at,
                {bucket} as time_bucket
            FROM market_candles
        """
    
    @timed_async("DB Query: get_candle_by_time")
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
                closed,
                received_at
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
    
    @timed_async("DB Query: count_candles")
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
