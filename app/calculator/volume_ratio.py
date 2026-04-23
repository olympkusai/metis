"""Volume ratio calculator."""
import numpy as np
from .types import Candle, FeatureResult


class VolumeRatioCalculator:
    """Calculates the ratio of current volume to average volume."""
    
    def __init__(self, period: int):
        self.period = period
    
    def name(self) -> str:
        return "volume_ratio"
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Calculate volume ratio for the specified period.
        
        Volume Ratio = Current Volume / Average Volume over period
        """
        if len(candles) < self.period:
            raise ValueError(f"not enough candles for volume ratio calculation (need {self.period}, got {len(candles)})")
        
        # Extract volumes using numpy
        volumes = np.array([c.base_volume for c in candles])
        
        # Calculate rolling average using numpy
        ratio_values = np.zeros(len(candles))
        
        for i in range(self.period - 1, len(candles)):
            window_volumes = volumes[i - self.period + 1:i + 1]
            avg_volume = np.mean(window_volumes)
            ratio_values[i] = volumes[i] / avg_volume if avg_volume != 0 else 0.0
        
        # Convert back to FeatureResult objects
        results = [
            FeatureResult(
                name="volume_ratio",
                value=float(ratio_values[i]),
                timestamp=candles[i].close_time
            )
            for i in range(len(candles))
        ]
        
        return results
