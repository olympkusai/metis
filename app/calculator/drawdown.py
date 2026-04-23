"""Drawdown calculator."""
from .types import Candle, IndicatorResult


class MaxDrawdownCalculator:
    """Calculates maximum drawdown."""
    
    def name(self) -> str:
        return "max_drawdown"
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Calculate maximum drawdown from peak.
        
        Drawdown = (Peak - Trough) / Peak
        """
        if len(candles) < 2:
            raise ValueError("at least 2 candles required for max drawdown calculation")
        
        results: list[IndicatorResult] = []
        
        peak = candles[0].high_price
        max_drawdown = 0.0
        
        for candle in candles:
            # Update peak
            if candle.high_price > peak:
                peak = candle.high_price
            
            # Calculate drawdown from peak
            if peak > 0:
                drawdown = (peak - candle.low_price) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            results.append(IndicatorResult(
                name="max_drawdown",
                value=max_drawdown,
                timestamp=candle.close_time
            ))
        
        return results


def new_max_drawdown_calculator() -> MaxDrawdownCalculator:
    """Create a new max drawdown calculator."""
    return MaxDrawdownCalculator()
