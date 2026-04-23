"""Momentum calculator."""
import numpy as np
from .types import Candle, FeatureResult


class MomentumCalculator:
    """Calculates momentum over a period."""
    
    def __init__(self, period: int):
        self.period = period
    
    def name(self) -> str:
        return f"momentum_{self.period}d"
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Calculate momentum (price change over period).
        
        Momentum = (CurrentClosePrice - ClosePrice N periods ago) / ClosePrice N periods ago
        """
        if len(candles) < self.period:
            raise ValueError(f"not enough candles for momentum calculation (need {self.period}, got {len(candles)})")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate rolling momentum using numpy
        momentum_values = np.zeros(len(candles))
        
        for i in range(self.period, len(candles)):
            current_close = close_prices[i]
            past_close = close_prices[i - self.period]
            
            if past_close == 0:
                momentum_values[i] = 0.0
            else:
                momentum_values[i] = (current_close - past_close) / past_close
        
        # Convert back to FeatureResult objects
        results = [
            FeatureResult(
                name=self.get_name(),
                value=float(momentum_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results
    
    def get_name(self) -> str:
        return f"momentum_{self.period}d"


def momentum_30d() -> MomentumCalculator:
    """Create a calculator for 30-day momentum (43200 minutes)."""
    return MomentumCalculator(43200)  # 30 days * 24 hours * 60 minutes
