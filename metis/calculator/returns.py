"""Returns and log returns calculators."""
import math
import numpy as np
from .types import Candle, FeatureResult


class ReturnsCalculator:
    """Calculates simple returns."""
    
    def name(self) -> str:
        return "returns"
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Calculate simple returns for each candle.
        
        Return = (ClosePrice - PreviousClosePrice) / PreviousClosePrice
        """
        if len(candles) < 2:
            raise ValueError("at least 2 candles required for returns calculation")
        
        # Extract close prices using numpy for vectorized operations
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate returns using numpy
        returns = np.zeros(len(candles))
        returns[1:] = np.diff(close_prices) / close_prices[:-1]
        
        # Handle division by zero
        returns[1:] = np.where(close_prices[:-1] == 0, 0.0, returns[1:])
        
        # Convert back to FeatureResult objects
        results = [
            FeatureResult(
                name="returns",
                value=float(returns[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results


class LogReturnsCalculator:
    """Calculates logarithmic returns."""
    
    def name(self) -> str:
        return "log_returns"
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Calculate logarithmic returns for each candle.
        
        Log Return = ln(ClosePrice / PreviousClosePrice)
        """
        if len(candles) < 2:
            raise ValueError("at least 2 candles required for log returns calculation")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate log returns using numpy
        log_returns = np.zeros(len(candles))
        ratios = close_prices[1:] / close_prices[:-1]
        
        # Handle invalid ratios (zero or negative)
        valid_mask = (close_prices[:-1] > 0) & (close_prices[1:] > 0)
        log_returns[1:] = np.where(valid_mask, np.log(ratios), 0.0)
        
        # Convert back to FeatureResult objects
        results = [
            FeatureResult(
                name="log_returns",
                value=float(log_returns[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results
