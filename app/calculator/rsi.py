"""RSI calculator."""
import math
import numpy as np
from .types import Candle, IndicatorResult


class RSICalculator:
    """Calculates Relative Strength Index."""
    
    def __init__(self, period: int):
        self.period = period
    
    def name(self) -> str:
        return f"rsi_{self.period}"
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Calculate RSI for the specified period.
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if len(candles) < self.period + 1:
            raise ValueError(f"not enough candles for RSI calculation (need {self.period + 1}, got {len(candles)})")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate price changes using numpy
        changes = np.zeros(len(candles))
        changes[1:] = np.diff(close_prices)
        
        # Separate gains and losses
        gains = np.where(changes > 0, changes, 0.0)
        losses = np.where(changes < 0, -changes, 0.0)
        
        # Calculate rolling RSI using numpy
        rsi_values = np.full(len(candles), 50.0)  # Default to neutral
        
        for i in range(self.period, len(candles)):
            window_gains = gains[i - self.period + 1:i + 1]
            window_losses = losses[i - self.period + 1:i + 1]
            
            avg_gain = np.mean(window_gains)
            avg_loss = np.mean(window_losses)
            
            if avg_loss == 0:
                rsi_values[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))
        
        # Convert back to IndicatorResult objects
        results = [
            IndicatorResult(
                name=self.get_name(),
                value=float(rsi_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results
    
    def get_name(self) -> str:
        return f"rsi_{self.period}"


def rsi_14() -> RSICalculator:
    """Create a calculator for 14-period RSI."""
    return RSICalculator(14)
