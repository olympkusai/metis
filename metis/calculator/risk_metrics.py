"""Risk metrics calculators (Sharpe, Calmar)."""
import math
import numpy as np
from .types import Candle, IndicatorResult


class SharpeCalculator:
    """Calculates Sharpe ratio."""
    
    def __init__(self, period: int, risk_free_rate: float = 0.0):
        self.period = period
        self.risk_free_rate = risk_free_rate
    
    def name(self) -> str:
        return "sharpe"
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Calculate Sharpe ratio over the specified period.
        
        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        """
        if len(candles) < self.period:
            raise ValueError(f"not enough candles for Sharpe ratio calculation (need {self.period}, got {len(candles)})")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate returns using numpy
        returns = np.zeros(len(candles))
        returns[1:] = np.diff(close_prices) / close_prices[:-1]
        returns[1:] = np.where(close_prices[:-1] == 0, 0.0, returns[1:])
        
        # Calculate rolling Sharpe ratio using numpy
        sharpe_values = np.zeros(len(candles))
        
        for i in range(self.period - 1, len(candles)):
            window_returns = returns[i - self.period + 1:i + 1]
            mean_return = np.mean(window_returns)
            std_dev = np.std(window_returns, ddof=0)
            sharpe_values[i] = (mean_return - self.risk_free_rate) / std_dev if std_dev != 0 else 0.0
        
        # Convert back to IndicatorResult objects
        results = [
            IndicatorResult(
                name="sharpe",
                value=float(sharpe_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results


class CalmarCalculator:
    """Calculates Calmar ratio."""
    
    def __init__(self, period: int):
        self.period = period
    
    def name(self) -> str:
        return "calmar"
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Calculate Calmar ratio over the specified period.
        
        Calmar = Total Return / Maximum Drawdown
        """
        if len(candles) < self.period:
            raise ValueError(f"not enough candles for Calmar ratio calculation (need {self.period}, got {len(candles)})")
        
        results: list[IndicatorResult] = []
        
        for i in range(len(candles)):
            if i < self.period - 1:
                results.append(IndicatorResult(
                    name="calmar",
                    value=0.0,
                    timestamp=candles[i].close_time
                ))
                continue
            
            # Calculate total return over the period
            start_price = candles[i - self.period + 1].close_price
            end_price = candles[i].close_price
            
            total_return = (end_price - start_price) / start_price if start_price != 0 else 0.0
            
            # Calculate maximum drawdown over the period
            peak = candles[i - self.period + 1].high_price
            max_drawdown = 0.0
            
            for j in range(i - self.period + 1, i + 1):
                if candles[j].high_price > peak:
                    peak = candles[j].high_price
                
                if peak > 0:
                    drawdown = (peak - candles[j].low_price) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
            
            # Calculate Calmar ratio
            calmar = total_return / max_drawdown if max_drawdown != 0 else 0.0
            
            results.append(IndicatorResult(
                name="calmar",
                value=calmar,
                timestamp=candles[i].close_time
            ))
        
        return results


def sharpe(period: int, risk_free_rate: float = 0.0) -> SharpeCalculator:
    """Create a calculator for Sharpe ratio with specified risk-free rate."""
    return SharpeCalculator(period, risk_free_rate)


def calmar(period: int) -> CalmarCalculator:
    """Create a calculator for Calmar ratio."""
    return CalmarCalculator(period)
