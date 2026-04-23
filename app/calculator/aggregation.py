"""OHLCV aggregation utility."""
from datetime import datetime, timedelta
from .types import Candle


# Interval to minutes mapping
INTERVAL_TO_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "1D": 1440,  # Support uppercase
    "3d": 4320,
    "1w": 10080,
    "1M": 43200,  # 30 days
}


def aggregate_ohlcv(candles: list[Candle], target_interval: str) -> list[Candle]:
    """Aggregate 1m candles into the target interval.
    
    Args:
        candles: List of 1m candles to aggregate
        target_interval: Target interval (e.g., "5m", "1h", "1d")
        
    Returns:
        List of aggregated candles
        
    Raises:
        ValueError: If no candles provided or invalid target interval
    """
    if len(candles) == 0:
        raise ValueError("no candles provided")
    
    target_minutes = INTERVAL_TO_MINUTES.get(target_interval)
    if target_minutes is None:
        raise ValueError(f"invalid target interval: {target_interval}")
    
    if target_minutes == 1:
        # No aggregation needed for 1m
        return candles
    
    # Group candles by target interval
    grouped: dict[datetime, list[Candle]] = {}
    for candle in candles:
        # Truncate time to target interval
        truncated_time = candle.open_time.replace(
            minute=(candle.open_time.minute // target_minutes) * target_minutes,
            second=0,
            microsecond=0
        )
        if truncated_time not in grouped:
            grouped[truncated_time] = []
        grouped[truncated_time].append(candle)
    
    # Aggregate each group
    result: list[Candle] = []
    for timestamp in sorted(grouped.keys()):
        group = grouped[timestamp]
        
        aggregated = Candle(
            symbol=group[0].symbol,
            interval=target_interval,
            open_time=timestamp,
            close_time=timestamp + timedelta(minutes=target_minutes),
            open_price=group[0].open_price,
            close_price=group[-1].close_price,
            high_price=max(c.high_price for c in group),
            low_price=min(c.low_price for c in group),
            base_volume=sum(c.base_volume for c in group),
            quote_volume=sum(c.quote_volume for c in group),
        )
        
        result.append(aggregated)
    
    return result
