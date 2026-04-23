"""Volatility calculator."""
import math
import numpy as np
from .types import Candle, FeatureResult


class VolatilityCalculator:
    """Calculates volatility for different periods."""
    
    def __init__(self, period: int, name: str = "volatility"):
        self.period = period
        self.name_ = name
    
    def name(self) -> str:
        return self.name_
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Compute the volatility for each candle."""
        if len(candles) < self.period:
            raise ValueError(f"not enough candles for volatility calculation (need {self.period}, got {len(candles)})")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate returns using numpy
        returns = np.zeros(len(candles))
        returns[1:] = np.diff(close_prices) / close_prices[:-1]
        returns[1:] = np.where(close_prices[:-1] == 0, 0.0, returns[1:])
        
        # Calculate rolling volatility using numpy
        volatility_values = np.zeros(len(candles))
        
        for i in range(self.period - 1, len(candles)):
            window_returns = returns[i - self.period + 1:i + 1]
            volatility_values[i] = np.std(window_returns, ddof=0)
        
        # Convert back to FeatureResult objects
        results = [
            FeatureResult(
                name=self.get_name(),
                value=float(volatility_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results
    
    def get_name(self) -> str:
        return self.name_


def volatility_7() -> VolatilityCalculator:
    """Create a calculator for 7-period volatility."""
    return VolatilityCalculator(7, "volatility_7")


def volatility_21() -> VolatilityCalculator:
    """Create a calculator for 21-period volatility."""
    return VolatilityCalculator(21, "volatility_21")
