"""Target calculator."""
import numpy as np
from .types import Candle, FeatureResult


class TargetCalculator:
    """Calculates the target (future return) for prediction."""
    
    def __init__(self, period: int):
        self.period = period
    
    def name(self) -> str:
        return f"target_{self.period}d"
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Calculate the target (return over the next period).
        
        Target = (FutureClosePrice - CurrentClosePrice) / CurrentClosePrice
        """
        if len(candles) <= self.period:
            raise ValueError(f"not enough candles for target calculation (need > {self.period}, got {len(candles)})")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate target using numpy
        target_values = np.zeros(len(candles))
        
        for i in range(len(candles) - self.period):
            current_close = close_prices[i]
            future_close = close_prices[i + self.period]
            
            if current_close == 0:
                target_values[i] = 0.0
            else:
                target_values[i] = (future_close - current_close) / current_close
        
        # Last 'period' candles have no future data (already 0)
        
        # Convert back to FeatureResult objects
        results = [
            FeatureResult(
                name=self.name(),
                value=float(target_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results
