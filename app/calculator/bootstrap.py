"""Bootstrap calculator."""
import numpy as np
from .types import Candle, IndicatorResult


class BootstrapCalculator:
    """Calculates bootstrap statistics."""
    
    def __init__(self, samples: int):
        self.samples = samples
    
    def name(self) -> str:
        return f"bootstrap_{self.samples}"
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Calculate bootstrap statistics for returns.
        
        This resamples returns with replacement to estimate distribution.
        """
        if len(candles) < 2:
            raise ValueError("at least 2 candles required for bootstrap calculation")
        
        # Extract close prices using numpy
        close_prices = np.array([c.close_price for c in candles])
        
        # Calculate returns using numpy
        returns = np.zeros(len(candles))
        returns[1:] = np.diff(close_prices) / close_prices[:-1]
        returns[1:] = np.where(close_prices[:-1] == 0, 0.0, returns[1:])
        
        # Calculate rolling bootstrap using numpy
        bootstrap_values = np.zeros(len(candles))
        
        for i in range(len(candles)):
            if i < 20:
                continue
            
            window_size = min(i + 1, 20)
            window_returns = returns[i - window_size + 1:i + 1]
            
            # Bootstrap resampling using numpy
            sample_means = np.zeros(self.samples)
            for s in range(self.samples):
                # Resample with replacement
                sample = np.random.choice(window_returns, size=window_size, replace=True)
                sample_means[s] = np.mean(sample)
            
            # Calculate mean of bootstrap samples
            bootstrap_values[i] = np.mean(sample_means)
        
        # Convert back to IndicatorResult objects
        results = [
            IndicatorResult(
                name=self.get_name(),
                value=float(bootstrap_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results
    
    def get_name(self) -> str:
        return f"bootstrap_{self.samples}"


def bootstrap_20() -> BootstrapCalculator:
    """Create a calculator for 20-sample bootstrap."""
    return BootstrapCalculator(20)
