"""EMA of returns calculator."""
import numpy as np
from .types import Candle, FeatureResult


class EMAReturnCalculator:
    """Calculates EMA of returns."""
    
    def __init__(self, span: int):
        self.span = span
    
    def name(self) -> str:
        return f"ema_return_{self.span}"
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Calculate EMA of returns for the specified span."""
        if len(candles) < 2:
            raise ValueError("at least 2 candles required for EMA return calculation")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate returns using numpy
        returns = np.zeros(len(candles))
        returns[1:] = np.diff(close_prices) / close_prices[:-1]
        returns[1:] = np.where(close_prices[:-1] == 0, 0.0, returns[1:])
        
        # Calculate EMA using pandas-like approach with numpy
        alpha = 2.0 / (self.span + 1)
        ema_values = np.zeros(len(candles))
        ema_values[0] = returns[0]
        
        for i in range(1, len(candles)):
            ema_values[i] = alpha * returns[i] + (1 - alpha) * ema_values[i - 1]
        
        # Convert back to FeatureResult objects
        results = [
            FeatureResult(
                name=self.get_name(),
                value=float(ema_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results
    
    def get_name(self) -> str:
        return f"ema_return_{self.span}"


def ema_return_60() -> EMAReturnCalculator:
    """Create a calculator for 60-period EMA return."""
    return EMAReturnCalculator(60)
