"""Moving average calculator."""
import numpy as np
from .types import Candle, FeatureResult


class MovingAverageCalculator:
    """Calculates moving averages for different periods."""
    
    def __init__(self, period: int, name: str):
        self.period = period
        self.name_ = name
    
    def name(self) -> str:
        return self.name_
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Calculate moving average for the specified period."""
        if len(candles) < self.period:
            raise ValueError(f"not enough candles for moving average calculation (need {self.period}, got {len(candles)})")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate moving average using numpy's convolution
        ma_values = np.zeros(len(candles))
        
        # Use uniform filter for moving average (more efficient than manual loop)
        for i in range(self.period - 1, len(candles)):
            ma_values[i] = np.mean(close_prices[i - self.period + 1:i + 1])
        
        # Convert back to FeatureResult objects
        results = [
            FeatureResult(
                name=self.get_name(),
                value=float(ma_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results
    
    def get_name(self) -> str:
        return self.name_


def ma_7() -> MovingAverageCalculator:
    """Create a calculator for 7-period moving average."""
    return MovingAverageCalculator(7, "ma_7")


def ma_21() -> MovingAverageCalculator:
    """Create a calculator for 21-period moving average."""
    return MovingAverageCalculator(21, "ma_21")


def ma_50() -> MovingAverageCalculator:
    """Create a calculator for 50-period moving average."""
    return MovingAverageCalculator(50, "ma_50")
