"""Time feature calculators."""
from .types import Candle, FeatureResult


class DayOfWeekCalculator:
    """Calculates the day of week feature."""
    
    def name(self) -> str:
        return "day_of_week"
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Calculate day of week (0-6, where 0 is Monday)."""
        results: list[FeatureResult] = []
        
        for candle in candles:
            day_of_week = float(candle.open_time.weekday())
            
            results.append(FeatureResult(
                name="day_of_week",
                value=day_of_week,
                timestamp=candle.close_time
            ))
        
        return results


class MonthCalculator:
    """Calculates the month feature."""
    
    def name(self) -> str:
        return "month"
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Calculate month (1-12)."""
        results: list[FeatureResult] = []
        
        for candle in candles:
            month = float(candle.open_time.month)
            
            results.append(FeatureResult(
                name="month",
                value=month,
                timestamp=candle.close_time
            ))
        
        return results
