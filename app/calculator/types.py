"""Base types for calculator package."""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class Candle:
    """Represents a single OHLCV candle."""
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    base_volume: float
    quote_volume: float


@dataclass
class FeatureResult:
    """Represents a calculated feature value."""
    name: str
    value: float
    timestamp: datetime


@dataclass
class IndicatorResult:
    """Represents a calculated indicator value."""
    name: str
    value: float
    timestamp: datetime


class FeatureCalculator(Protocol):
    """Interface for feature calculators."""
    
    def name(self) -> str:
        """Return the name of the calculator."""
        ...
    
    def calculate(self, candles: list[Candle]) -> list[FeatureResult]:
        """Calculate the feature for the given candles."""
        ...


class IndicatorCalculator(Protocol):
    """Interface for indicator calculators."""
    
    def name(self) -> str:
        """Return the name of the calculator."""
        ...
    
    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Calculate the indicator for the given candles."""
        ...
