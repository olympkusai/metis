"""Bollinger Bands calculator."""
import math
import numpy as np
from .types import Candle, IndicatorResult


class BollingerBandsCalculator:
    """Calculates Bollinger Bands width."""
    
    def __init__(self, period: int, std_dev: float):
        self.period = period
        self.std_dev = std_dev
    
    def name(self) -> str:
        return "bb_width"
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Calculate upper, lower, and width of Bollinger Bands."""
        if len(candles) < self.period:
            raise ValueError(f"not enough candles for Bollinger Bands calculation (need {self.period}, got {len(candles)})")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate rolling Bollinger Bands using numpy
        width_values = np.zeros(len(candles))
        
        for i in range(self.period - 1, len(candles)):
            window_prices = close_prices[i - self.period + 1:i + 1]
            mean = np.mean(window_prices)
            std_dev_val = np.std(window_prices, ddof=0)
            
            upper = mean + self.std_dev * std_dev_val
            lower = mean - self.std_dev * std_dev_val
            width_values[i] = (upper - lower) / mean if mean != 0 else 0.0
        
        # Convert back to IndicatorResult objects
        results = [
            IndicatorResult(
                name="bb_width",
                value=float(width_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results


class BBUpperCalculator:
    """Calculates Bollinger Bands upper band."""
    
    def __init__(self, period: int, std_dev: float):
        self.period = period
        self.std_dev = std_dev
    
    def name(self) -> str:
        return "bb_upper"
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Compute the upper Bollinger Band for each candle."""
        if len(candles) < self.period:
            raise ValueError(f"not enough candles for BB upper calculation (need {self.period}, got {len(candles)})")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate rolling upper band using numpy
        upper_values = np.zeros(len(candles))
        
        for i in range(self.period - 1, len(candles)):
            window_prices = close_prices[i - self.period + 1:i + 1]
            mean = np.mean(window_prices)
            std_dev_val = np.std(window_prices, ddof=0)
            upper_values[i] = mean + self.std_dev * std_dev_val
        
        # Convert back to IndicatorResult objects
        results = [
            IndicatorResult(
                name="bb_upper",
                value=float(upper_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results


class BBLowerCalculator:
    """Calculates Bollinger Bands lower band."""
    
    def __init__(self, period: int, std_dev: float):
        self.period = period
        self.std_dev = std_dev
    
    def name(self) -> str:
        return "bb_lower"
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Calculate lower Bollinger Band."""
        if len(candles) < self.period:
            raise ValueError(f"not enough candles for BB lower calculation (need {self.period}, got {len(candles)})")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate rolling lower band using numpy
        lower_values = np.zeros(len(candles))
        
        for i in range(self.period - 1, len(candles)):
            window_prices = close_prices[i - self.period + 1:i + 1]
            mean = np.mean(window_prices)
            std_dev_val = np.std(window_prices, ddof=0)
            lower_values[i] = mean - self.std_dev * std_dev_val
        
        # Convert back to IndicatorResult objects
        results = [
            IndicatorResult(
                name="bb_lower",
                value=float(lower_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results


def bb() -> tuple[BBUpperCalculator, BBLowerCalculator, BollingerBandsCalculator]:
    """Create calculators for standard Bollinger Bands (20, 2)."""
    return BBUpperCalculator(20, 2), BBLowerCalculator(20, 2), BollingerBandsCalculator(20, 2)
