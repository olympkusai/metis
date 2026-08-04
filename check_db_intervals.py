"""Check which intervals have data in database and test aggregation.

Debug utility. Reads DATABASE_DSN from environment (or .env via metis.config).
Optional CLI args: SYMBOL DAYS (defaults to BTCUSDT, 30 days).
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metis.config import get_settings
from metis.storage.market_candle import MarketCandleQueries
from metis.storage.pool import create_pool


async def check_intervals(symbol: str = "BTCUSDT", days: int = 30) -> None:
    dsn = get_settings().database_dsn
    if not dsn:
        raise SystemExit(
            "DATABASE_DSN is not configured. Set it in .env or as an environment variable."
        )

    pool = await create_pool(dsn)
    try:
        repo = MarketCandleQueries(pool)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        print(f"Checking data availability for {symbol} (last {days} days)")
        print("=" * 60)

        # Test SQL aggregation for multi-timeframe targets (with normalization)
        intervals = ["1m", "1h", "4h", "1d", "1D"]
        for interval in intervals:
            try:
                candles = await repo.get_candles(
                    symbol, interval, start_time, end_time, limit=100
                )
                print(f"{interval:5s}: {len(candles):4d} candles (SQL aggregation)")
                if candles:
                    print(
                        f"       First: {candles[0].close_time}, "
                        f"Last: {candles[-1].close_time}"
                    )
            except Exception as e:
                print(f"{interval:5s}: ERROR - {str(e)[:100]}")
    finally:
        await pool.close()


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    asyncio.run(check_intervals(sym, n_days))
