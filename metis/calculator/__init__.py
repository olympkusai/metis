"""Calculator package for financial metrics and indicators."""
from .types import Candle, FeatureResult, IndicatorResult, FeatureCalculator, IndicatorCalculator

# Feature calculators
from .returns import ReturnsCalculator, LogReturnsCalculator
from .moving_average import MovingAverageCalculator, ma_7, ma_21, ma_50
from .volatility import VolatilityCalculator, volatility_7, volatility_21
from .volume_ratio import VolumeRatioCalculator
from .time_features import DayOfWeekCalculator, MonthCalculator
from .target import TargetCalculator
from .momentum import MomentumCalculator, momentum_30d
from .ewma import EWMACalculator, ewma_30d
from .ema_return import EMAReturnCalculator, ema_return_60

# Indicator calculators
from .rsi import RSICalculator, rsi_14
from .macd import MACDCalculator, MACDSignalCalculator, macd, macd_signal
from .bollinger_bands import BollingerBandsCalculator, BBUpperCalculator, BBLowerCalculator, bb
from .cvar import CVaRCalculator, cvar_95
from .drawdown import MaxDrawdownCalculator, new_max_drawdown_calculator
from .risk_metrics import SharpeCalculator, CalmarCalculator, sharpe, calmar
from .bootstrap import BootstrapCalculator, bootstrap_20

# Infrastructure
from .batch import BatchCalculator, default_batch_calculator, calculate_batch_async
from .engine import CalculationEngine, CalculationRequest, CalculationResponse, create_calculation_engine
from .frequent import FrequentCalculator, FrequentCalculation, PersistedCalculation, create_frequent_calculator

# Pandas utilities
from .pandas_utils import (
    candles_to_dataframe,
    dataframe_to_feature_results,
    dataframe_to_indicator_results,
    rolling_mean_pandas,
    rolling_std_pandas,
    ewma_pandas,
    calculate_returns_pandas,
    calculate_log_returns_pandas,
)

# Aggregation utility
from .aggregation import aggregate_ohlcv, INTERVAL_TO_MINUTES

__all__ = [
    # Types
    "Candle",
    "FeatureResult",
    "IndicatorResult",
    "FeatureCalculator",
    "IndicatorCalculator",
    
    # Feature calculators
    "ReturnsCalculator",
    "LogReturnsCalculator",
    "MovingAverageCalculator",
    "ma_7",
    "ma_21",
    "ma_50",
    "VolatilityCalculator",
    "volatility_7",
    "volatility_21",
    "VolumeRatioCalculator",
    "DayOfWeekCalculator",
    "MonthCalculator",
    "TargetCalculator",
    "MomentumCalculator",
    "momentum_30d",
    "EWMACalculator",
    "ewma_30d",
    "EMAReturnCalculator",
    "ema_return_60",
    
    # Indicator calculators
    "RSICalculator",
    "rsi_14",
    "MACDCalculator",
    "MACDSignalCalculator",
    "macd",
    "macd_signal",
    "BollingerBandsCalculator",
    "BBUpperCalculator",
    "BBLowerCalculator",
    "bb",
    "CVaRCalculator",
    "cvar_95",
    "MaxDrawdownCalculator",
    "new_max_drawdown_calculator",
    "SharpeCalculator",
    "CalmarCalculator",
    "sharpe",
    "calmar",
    "BootstrapCalculator",
    "bootstrap_20",
    
    # Infrastructure
    "BatchCalculator",
    "default_batch_calculator",
    "calculate_batch_async",
    "CalculationEngine",
    "CalculationRequest",
    "CalculationResponse",
    "create_calculation_engine",
    "FrequentCalculator",
    "FrequentCalculation",
    "PersistedCalculation",
    "create_frequent_calculator",
    
    # Pandas utilities
    "candles_to_dataframe",
    "dataframe_to_feature_results",
    "dataframe_to_indicator_results",
    "rolling_mean_pandas",
    "rolling_std_pandas",
    "ewma_pandas",
    "calculate_returns_pandas",
    "calculate_log_returns_pandas",
    
    # Aggregation
    "aggregate_ohlcv",
    "INTERVAL_TO_MINUTES",
]
