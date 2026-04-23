"""Check which intervals have data in database and test aggregation."""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from app.storage.pool import create_pool
from app.storage.market_candle import MarketCandleQueries

async def check_intervals():
    # Create pool using remote database URL
    dsn = "postgres://postgres:BGxE9aWYJP5Ai7rhLkGeQUcnt8Y4hvnq3IM282m7OgEtKIF4QjmMUbIND07qCBR9@88.99.66.165:5432/k0s_prd?sslmode=require"
    pool = await create_pool(dsn)
    repo = MarketCandleQueries(pool)
    
    symbol = 'BTCUSDT'  # Database uses BTCUSDT, not BTC
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)  # Last 30 days
    
    print(f"Checking data availability for {symbol} (last 30 days)")
    print("=" * 60)
    
    # Test SQL aggregation for multi-timeframe targets (with normalization)
    intervals = ['1m', '1h', '4h', '1d', '1D']
    
    for interval in intervals:
        try:
            candles = await repo.get_candles(symbol, interval, start_time, end_time, limit=100)
            print(f"{interval:5s}: {len(candles):4d} candles (SQL aggregation)")
            if candles:
                print(f"       First: {candles[0].close_time}, Last: {candles[-1].close_time}")
        except Exception as e:
            print(f"{interval:5s}: ERROR - {str(e)[:100]}")
    
    await pool.close()

if __name__ == "__main__":
    asyncio.run(check_intervals())
