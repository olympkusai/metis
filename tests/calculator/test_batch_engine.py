"""Tests for BatchCalculator and CalculationEngine."""
import pytest
from datetime import datetime, timedelta

from metis.calculator.types import Candle
from metis.calculator.batch import BatchCalculator, default_batch_calculator
from metis.calculator.engine import CalculationEngine, CalculationRequest, create_calculation_engine
from metis.calculator.returns import ReturnsCalculator
from metis.calculator.rsi import rsi_14
from metis.calculator.moving_average import ma_7


def generate_test_candles(count: int) -> list[Candle]:
    """Generate test candles with realistic price movements."""
    candles: list[Candle] = []
    base_price = 100.0
    base_time = datetime.now() - timedelta(minutes=count)
    
    for i in range(count):
        change = (base_price * 0.01) * ((i % 3) - 1) / 3
        base_price += change
        
        candle = Candle(
            symbol="BTCUSDT",
            interval="1m",
            open_time=base_time + timedelta(minutes=i),
            close_time=base_time + timedelta(minutes=i + 1),
            open_price=base_price,
            high_price=base_price * 1.005,
            low_price=base_price * 0.995,
            close_price=base_price + change,
            base_volume=1000.0 + (i % 10) * 100,
            quote_volume=(base_price + change) * (1000.0 + (i % 10) * 100),
        )
        candles.append(candle)
    
    return candles


class TestBatchCalculator:
    """Tests for BatchCalculator."""
    
    def test_batch_calculation(self):
        candles = generate_test_candles(50)
        calc = default_batch_calculator()
        
        calculators = [ReturnsCalculator(), rsi_14(), ma_7()]
        feature_results, indicator_results = calc.calculate_batch(candles, calculators)
        
        assert len(feature_results) == 2  # ReturnsCalculator and ma_7
        assert len(indicator_results) == 1  # rsi_14
        assert "returns" in feature_results
        assert "ma_7" in feature_results
        assert "rsi_14" in indicator_results
        
    def test_batch_calculation_with_timeout(self):
        candles = generate_test_candles(100)
        calc = BatchCalculator(max_concurrent=2, timeout=5.0)
        
        calculators = [ReturnsCalculator(), rsi_14(), ma_7()]
        feature_results, indicator_results = calc.calculate_batch(candles, calculators)
        
        assert len(feature_results) == 2
        assert len(indicator_results) == 1


class TestCalculationEngine:
    """Tests for CalculationEngine."""
    
    def test_engine_creation(self):
        engine = create_calculation_engine()
        
        assert engine is not None
        assert len(engine.list_available_features()) > 0
        assert len(engine.list_available_indicators()) > 0
        
    def test_get_feature_calculator(self):
        engine = create_calculation_engine()
        
        calc = engine.get_feature_calculator("returns")
        assert calc is not None
        assert calc.name() == "returns"
        
        calc = engine.get_feature_calculator("unknown")
        assert calc is None
        
    def test_get_indicator_calculator(self):
        engine = create_calculation_engine()
        
        calc = engine.get_indicator_calculator("rsi_14")
        assert calc is not None
        assert calc.name() == "rsi_14"
        
        calc = engine.get_indicator_calculator("unknown")
        assert calc is None
        
    def test_process_request(self):
        candles = generate_test_candles(50)
        engine = create_calculation_engine()
        
        req = CalculationRequest(
            symbol="BTCUSDT",
            interval="1m",
            start_time=datetime.now() - timedelta(minutes=50),
            end_time=datetime.now(),
            features=["returns", "ma_7"],
            indicators=["rsi_14"]
        )
        
        response = engine.process_request_sync(req, candles)
        
        assert response.symbol == "BTCUSDT"
        assert response.interval == "1m"
        assert len(response.features) == 2
        assert len(response.indicators) == 1
        assert "returns" in response.features
        assert "ma_7" in response.features
        assert "rsi_14" in response.indicators
        
    def test_register_custom_calculator(self):
        engine = create_calculation_engine()
        
        custom_calc = ReturnsCalculator()
        engine.register_feature_calculator("custom_returns", custom_calc)
        
        calc = engine.get_feature_calculator("custom_returns")
        assert calc is not None
        assert calc.name() == "returns"
