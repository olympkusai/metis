"""MACD calculator."""
import numpy as np
from .types import Candle, IndicatorResult


class MACDCalculator:
    """Calculates Moving Average Convergence Divergence."""
    
    def __init__(self, fast_period: int, slow_period: int, signal_period: int):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def name(self) -> str:
        return "macd"
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Calculate MACD line and signal line.
        
        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signalPeriod)
        """
        if len(candles) < self.slow_period + self.signal_period:
            raise ValueError(f"not enough candles for MACD calculation (need {self.slow_period + self.signal_period}, got {len(candles)})")
        
        results: list[IndicatorResult] = []
        
        # Calculate EMAs for fast and slow periods
        fast_ema = self._calculate_ema(candles, self.fast_period)
        slow_ema = self._calculate_ema(candles, self.slow_period)
        
        # Calculate MACD line
        for i in range(len(candles)):
            if i < self.slow_period - 1:
                results.append(IndicatorResult(
                    name="macd",
                    value=0.0,
                    timestamp=candles[i].close_time
                ))
                continue
            
            macd_line = fast_ema[i] - slow_ema[i]
            
            results.append(IndicatorResult(
                name="macd",
                value=macd_line,
                timestamp=candles[i].close_time
            ))
        
        return results
    
    def _calculate_ema(self, candles: list[Candle], period: int) -> list[float]:
        """Calculate EMA for candles using numpy."""
        alpha = 2.0 / (period + 1)
        close_prices = np.array([c.close_price for c in candles])
        
        ema = np.zeros(len(candles))
        ema[0] = close_prices[0]
        
        for i in range(1, len(candles)):
            ema[i] = alpha * close_prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema.tolist()


class MACDSignalCalculator:
    """Calculates MACD signal line."""
    
    def __init__(self, fast_period: int, slow_period: int, signal_period: int):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def name(self) -> str:
        return "macd_signal"
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Calculate MACD signal line."""
        if len(candles) < self.slow_period + self.signal_period:
            raise ValueError(f"not enough candles for MACD signal calculation (need {self.slow_period + self.signal_period}, got {len(candles)})")
        
        results: list[IndicatorResult] = []
        
        # Calculate EMAs for fast and slow periods
        fast_ema = self._calculate_ema(candles, self.fast_period)
        slow_ema = self._calculate_ema(candles, self.slow_period)
        
        # Calculate MACD line
        macd_line: list[float] = []
        for i in range(len(candles)):
            macd_line.append(fast_ema[i] - slow_ema[i])
        
        # Calculate signal line (EMA of MACD)
        signal_line = self._calculate_ema_from_values(macd_line, self.signal_period)
        
        for i in range(len(candles)):
            if i < self.slow_period + self.signal_period - 2:
                results.append(IndicatorResult(
                    name="macd_signal",
                    value=0.0,
                    timestamp=candles[i].close_time
                ))
                continue
            
            results.append(IndicatorResult(
                name="macd_signal",
                value=signal_line[i],
                timestamp=candles[i].close_time
            ))
        
        return results
    
    def _calculate_ema(self, candles: list[Candle], period: int) -> list[float]:
        """Calculate EMA for candles using numpy."""
        alpha = 2.0 / (period + 1)
        close_prices = np.array([c.close_price for c in candles])
        
        ema = np.zeros(len(candles))
        ema[0] = close_prices[0]
        
        for i in range(1, len(candles)):
            ema[i] = alpha * close_prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema.tolist()
    
    def _calculate_ema_from_values(self, values: list[float], period: int) -> list[float]:
        """Calculate EMA from values using numpy."""
        alpha = 2.0 / (period + 1)
        values_array = np.array(values)
        
        ema = np.zeros(len(values))
        ema[0] = values_array[0]
        
        for i in range(1, len(values)):
            ema[i] = alpha * values_array[i] + (1 - alpha) * ema[i - 1]
        
        return ema.tolist()


def macd() -> MACDCalculator:
    """Create a calculator for standard MACD (12, 26, 9)."""
    return MACDCalculator(12, 26, 9)


def macd_signal() -> MACDSignalCalculator:
    """Create a calculator for standard MACD signal (12, 26, 9)."""
    return MACDSignalCalculator(12, 26, 9)
