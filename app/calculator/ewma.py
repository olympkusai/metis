"""EWMA calculator."""
import numpy as np
from .types import Candle, FeatureResult


class EWMACalculator:
    """Calculates exponentially weighted moving average."""
    
    def __init__(self, span: int):
        self.span = span
    
    def name(self) -> str:
        return f"ewma_{self.span}d"
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Calculate EWMA for the specified span.
        
        EWMA uses alpha = 2 / (span + 1)
        """
        if len(candles) < 2:
            raise ValueError("at least 2 candles required for EWMA calculation")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate EWMA using numpy
        alpha = 2.0 / (self.span + 1)
        ewma_values = np.zeros(len(candles))
        ewma_values[0] = close_prices[0]
        
        for i in range(1, len(candles)):
            ewma_values[i] = alpha * close_prices[i] + (1 - alpha) * ewma_values[i - 1]
        
        # Convert back to FeatureResult objects
        results = [
            FeatureResult(
                name=self.get_name(),
                value=float(ewma_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results
    
    def get_name(self) -> str:
        return f"ewma_{self.span}d"


def ewma_30d() -> EWMACalculator:
    """Create a calculator for 30-day EWMA (43200 minutes)."""
    return EWMACalculator(43200)  # 30 days * 24 hours * 60 minutes
