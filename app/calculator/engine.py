"""Calculation engine for orchestrating calculations."""
from datetime import datetime
from typing import Any

from .types import Candle, FeatureResult, IndicatorResult
from .frequent import FrequentCalculator
from .returns import ReturnsCalculator, LogReturnsCalculator
from .moving_average import MovingAverageCalculator, ma_7, ma_21, ma_50
from .volatility import VolatilityCalculator, volatility_7, volatility_21
from .volume_ratio import VolumeRatioCalculator
from .time_features import DayOfWeekCalculator, MonthCalculator
from .target import TargetCalculator
from .momentum import MomentumCalculator, momentum_30d
from .ewma import EWMACalculator, ewma_30d
from .ema_return import EMAReturnCalculator, ema_return_60
from .rsi import RSICalculator, rsi_14
from .macd import MACDCalculator, MACDSignalCalculator, macd, macd_signal
from .bollinger_bands import BollingerBandsCalculator, BBUpperCalculator, BBLowerCalculator, bb
from .cvar import CVaRCalculator, cvar_95
from .drawdown import MaxDrawdownCalculator, new_max_drawdown_calculator
from .risk_metrics import SharpeCalculator, CalmarCalculator, sharpe, calmar
from .bootstrap import BootstrapCalculator, bootstrap_20
from .aggregation import aggregate_ohlcv


class CalculationRequest:
    """Represents a request for feature/indicator calculation."""
    
    def __init__(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        features: list[str] | None = None,
        indicators: list[str] | None = None
    ):
        self.symbol = symbol
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.features = features or []
        self.indicators = indicators or []


class CalculationResponse:
    """Represents the result of a calculation."""
    
    def __init__(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        features: dict[str, list[FeatureResult]] | None = None,
        indicators: dict[str, list[IndicatorResult]] | None = None
    ):
        self.symbol = symbol
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.features = features or {}
        self.indicators = indicators or {}


class CalculationEngine:
    """Handles calculation requests with calculator registry and caching.
    
    All calculations are performed locally using registered calculators.
    No external API calls are made - all features and indicators are computed
    from the provided candle data.
    """
    
    def __init__(self, frequent_calculator: FrequentCalculator | None = None):
        self._feature_calculators: dict[str, Any] = {
            "returns": ReturnsCalculator(),
            "log_returns": LogReturnsCalculator(),
            "ma_7": ma_7(),
            "ma_21": ma_21(),
            "ma_50": ma_50(),
            "volatility_7": volatility_7(),
            "volatility_21": volatility_21(),
            "volume_ratio": VolumeRatioCalculator(20),
            "day_of_week": DayOfWeekCalculator(),
            "month": MonthCalculator(),
            "target": TargetCalculator(1),
            "momentum_30d": momentum_30d(),
            "ewma_30d": ewma_30d(),
            "ema_return_60": ema_return_60(),
        }
        
        self._indicator_calculators: dict[str, Any] = {
            "rsi_14": rsi_14(),
            "macd": macd(),
            "macd_signal": macd_signal(),
            "bb_upper": BBUpperCalculator(20, 2),
            "bb_lower": BBLowerCalculator(20, 2),
            "bb_width": BollingerBandsCalculator(20, 2),
            "cvar_95": cvar_95(),
            "max_drawdown": new_max_drawdown_calculator(),
            "sharpe": sharpe(20),
            "calmar": calmar(20),
            "bootstrap_20": bootstrap_20(),
        }
        
        self.frequent_calculator = frequent_calculator
    
    def get_feature_calculator(self, name: str) -> Any | None:
        """Get a feature calculator by name."""
        return self._feature_calculators.get(name)
    
    def get_indicator_calculator(self, name: str) -> Any | None:
        """Get an indicator calculator by name."""
        return self._indicator_calculators.get(name)
    
    def register_feature_calculator(self, name: str, calculator: Any) -> None:
        """Register a custom feature calculator."""
        self._feature_calculators[name] = calculator
    
    def register_indicator_calculator(self, name: str, calculator: Any) -> None:
        """Register a custom indicator calculator."""
        self._indicator_calculators[name] = calculator
    
    def process_request_sync(
        self,
        req: CalculationRequest,
        candles: list[Candle]
    ) -> CalculationResponse:
        """Process a calculation request (synchronous version for backward compatibility).
        
        All calculations are performed locally using registered calculators.
        No external API calls are made.
        
        Args:
            req: Calculation request
            candles: Candles to calculate on (already fetched and aggregated)
            
        Returns:
            Calculation response with results
        """
        response = CalculationResponse(
            symbol=req.symbol,
            interval=req.interval,
            start_time=req.start_time,
            end_time=req.end_time
        )
        
        # Aggregate if needed
        if req.interval != "1m":
            candles = aggregate_ohlcv(candles, req.interval)
        
        # Calculate features without caching (sync mode)
        for feature_name in req.features:
            calc = self.get_feature_calculator(feature_name)
            if calc is None:
                print(f"Unknown feature: {feature_name}")
                continue
            
            try:
                results = calc.calculate(candles)
                response.features[feature_name] = results
            except Exception as e:
                print(f"Failed to calculate feature {feature_name}: {e}")
        
        # Calculate indicators without caching (sync mode)
        for indicator_name in req.indicators:
            calc = self.get_indicator_calculator(indicator_name)
            if calc is None:
                print(f"Unknown indicator: {indicator_name}")
                continue
            
            try:
                results = calc.calculate(candles)
                response.indicators[indicator_name] = results
            except Exception as e:
                print(f"Failed to calculate indicator {indicator_name}: {e}")
        
        return response
    
    async def process_request(
        self,
        req: CalculationRequest,
        candles: list[Candle]
    ) -> CalculationResponse:
        """Process a calculation request with caching support.
        
        All calculations are performed locally using registered calculators.
        No external API calls are made. Caching is handled by FrequentCalculator.
        
        Args:
            req: Calculation request
            candles: Candles to calculate on (already fetched and aggregated)
            
        Returns:
            Calculation response with results
        """
        response = CalculationResponse(
            symbol=req.symbol,
            interval=req.interval,
            start_time=req.start_time,
            end_time=req.end_time
        )
        
        # Aggregate if needed
        if req.interval != "1m":
            candles = aggregate_ohlcv(candles, req.interval)
        
        # Calculate features with caching support
        for feature_name in req.features:
            # Check cache first if frequent calculator is available
            if self.frequent_calculator:
                is_frequent = await self.frequent_calculator.check_frequent(req.symbol, req.interval, "feature", feature_name)
                if is_frequent:
                    cached = await self.frequent_calculator.get_persisted_result(req.symbol, req.interval, "feature", feature_name)
                    if cached:
                        response.features[feature_name] = self.frequent_calculator.deserialize_feature_results(cached)
                        continue
            
            # Increment request count
            if self.frequent_calculator:
                await self.frequent_calculator.increment_request_count(req.symbol, req.interval, "feature", feature_name)
            
            # Calculate
            calc = self.get_feature_calculator(feature_name)
            if calc is None:
                print(f"Unknown feature: {feature_name}")
                continue
            
            try:
                results = calc.calculate(candles)
                response.features[feature_name] = results
                
                # Persist if this is a frequent calculation
                if self.frequent_calculator:
                    is_frequent = await self.frequent_calculator.check_frequent(req.symbol, req.interval, "feature", feature_name)
                    if is_frequent:
                        serialized = self.frequent_calculator.serialize_feature_results(results)
                        await self.frequent_calculator.persist_result(req.symbol, req.interval, "feature", feature_name, serialized)
            except Exception as e:
                print(f"Failed to calculate feature {feature_name}: {e}")
        
        # Calculate indicators with caching support
        for indicator_name in req.indicators:
            # Check cache first if frequent calculator is available
            if self.frequent_calculator:
                is_frequent = await self.frequent_calculator.check_frequent(req.symbol, req.interval, "indicator", indicator_name)
                if is_frequent:
                    cached = await self.frequent_calculator.get_persisted_result(req.symbol, req.interval, "indicator", indicator_name)
                    if cached:
                        response.indicators[indicator_name] = self.frequent_calculator.deserialize_indicator_results(cached)
                        continue
            
            # Increment request count
            if self.frequent_calculator:
                await self.frequent_calculator.increment_request_count(req.symbol, req.interval, "indicator", indicator_name)
            
            # Calculate
            calc = self.get_indicator_calculator(indicator_name)
            if calc is None:
                print(f"Unknown indicator: {indicator_name}")
                continue
            
            try:
                results = calc.calculate(candles)
                response.indicators[indicator_name] = results
                
                # Persist if this is a frequent calculation
                if self.frequent_calculator:
                    is_frequent = await self.frequent_calculator.check_frequent(req.symbol, req.interval, "indicator", indicator_name)
                    if is_frequent:
                        serialized = self.frequent_calculator.serialize_indicator_results(results)
                        await self.frequent_calculator.persist_result(req.symbol, req.interval, "indicator", indicator_name, serialized)
            except Exception as e:
                print(f"Failed to calculate indicator {indicator_name}: {e}")
        
        return response
    
    def list_available_features(self) -> list[str]:
        """List all available feature calculators."""
        return list(self._feature_calculators.keys())
    
    def list_available_indicators(self) -> list[str]:
        """List all available indicator calculators."""
        return list(self._indicator_calculators.keys())


def create_calculation_engine(frequent_calculator: FrequentCalculator | None = None) -> CalculationEngine:
    """Create a new calculation engine with default calculators.
    
    Args:
        frequent_calculator: Optional FrequentCalculator for caching
        
    Returns:
        CalculationEngine instance
    """
    return CalculationEngine(frequent_calculator=frequent_calculator)
