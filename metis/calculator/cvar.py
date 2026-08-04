"""CVaR calculator."""
import math
import numpy as np
from .types import Candle, IndicatorResult


class CVaRCalculator:
    """Calculates Conditional Value at Risk (CVaR)."""
    
    def __init__(self, confidence: float):
        self.confidence = confidence
    
    def name(self) -> str:
        return f"cvar_{int(self.confidence * 100)}"
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Calculate CVaR at the specified confidence level.
        
        CVaR is the expected loss beyond the Value at Risk (VaR).
        """
        if len(candles) < 2:
            raise ValueError("at least 2 candles required for CVaR calculation")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate returns using numpy
        returns = np.zeros(len(candles))
        returns[1:] = np.diff(close_prices) / close_prices[:-1]
        returns[1:] = np.where(close_prices[:-1] == 0, 0.0, returns[1:])
        
        # Calculate rolling CVaR using numpy
        cvar_values = np.zeros(len(candles))
        
        for i in range(1, len(candles)):
            window_size = min(i + 1, 20)
            window_returns = returns[i - window_size + 1:i + 1]
            
            # Sort returns to find VaR
            sorted_returns = np.sort(window_returns)
            
            # Calculate VaR at confidence level
            var_index = int(math.ceil(len(sorted_returns) * (1 - self.confidence)))
            if var_index >= len(sorted_returns):
                var_index = len(sorted_returns) - 1
            var = sorted_returns[var_index]
            
            # Calculate CVaR (average of returns below VaR)
            below_var = sorted_returns[sorted_returns <= var]
            cvar_values[i] = np.mean(below_var) if len(below_var) > 0 else var
        
        # Convert back to IndicatorResult objects
        results = [
            IndicatorResult(
                name=self.get_name(),
                value=float(cvar_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results
    
    def get_name(self) -> str:
        return f"cvar_{int(self.confidence * 100)}"


def cvar_95() -> CVaRCalculator:
    """Create a calculator for 95% CVaR."""
    return CVaRCalculator(0.95)
