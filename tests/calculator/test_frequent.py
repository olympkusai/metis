"""Tests for FrequentCalculator."""
import pytest
import asyncio
from datetime import datetime, timedelta

from app.calculator.types import Candle, FeatureResult, IndicatorResult
from app.calculator.frequent import FrequentCalculator, FrequentCalculation, PersistedCalculation, create_frequent_calculator
from app.calculator.returns import ReturnsCalculator


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


class TestFrequentCalculator:
    """Tests for FrequentCalculator."""
    
    def test_create_frequent_calculator(self):
        calc = create_frequent_calculator()
        assert calc is not None
        assert not calc.use_database
    
    @pytest.mark.asyncio
    async def test_increment_request_count(self):
        calc = create_frequent_calculator()
        
        await calc.increment_request_count("BTCUSDT", "1m", "feature", "returns")
        
        frequent = calc.get_frequent_calculations()
        assert len(frequent) == 1
        assert frequent[0].request_count == 1
        assert frequent[0].symbol == "BTCUSDT"
        assert frequent[0].name == "returns"
        
    @pytest.mark.asyncio
    async def test_increment_multiple_times(self):
        calc = create_frequent_calculator()
        
        for _ in range(5):
            await calc.increment_request_count("BTCUSDT", "1m", "feature", "returns")
        
        frequent = calc.get_frequent_calculations()
        assert len(frequent) == 1
        assert frequent[0].request_count == 5
        
    @pytest.mark.asyncio
    async def test_check_frequent_not_persisted(self):
        calc = create_frequent_calculator()
        
        await calc.increment_request_count("BTCUSDT", "1m", "feature", "returns")
        
        is_frequent = await calc.check_frequent("BTCUSDT", "1m", "feature", "returns")
        assert not is_frequent
        
    @pytest.mark.asyncio
    async def test_check_frequent_persisted(self):
        calc = create_frequent_calculator()
        
        await calc.increment_request_count("BTCUSDT", "1m", "feature", "returns")
        await calc.evaluate_frequent_calculations(threshold=1)
        
        is_frequent = await calc.check_frequent("BTCUSDT", "1m", "feature", "returns")
        assert is_frequent
        
    @pytest.mark.asyncio
    async def test_evaluate_frequent_calculations(self):
        calc = create_frequent_calculator()
        
        # Add some calculations
        await calc.increment_request_count("BTCUSDT", "1m", "feature", "returns")
        await calc.increment_request_count("BTCUSDT", "1m", "feature", "ma_7")
        await calc.increment_request_count("BTCUSDT", "1m", "feature", "ma_7")
        
        # Evaluate with threshold of 2
        await calc.evaluate_frequent_calculations(threshold=2)
        
        frequent = calc.get_frequent_calculations()
        returns_calc = [f for f in frequent if f.name == "returns"][0]
        ma7_calc = [f for f in frequent if f.name == "ma_7"][0]
        
        assert not returns_calc.is_persisted
        assert ma7_calc.is_persisted
        
    @pytest.mark.asyncio
    async def test_persist_and_get_result(self):
        calc = create_frequent_calculator()
        
        data = b"test data"
        await calc.persist_result("BTCUSDT", "1m", "feature", "returns", data)
        
        retrieved = await calc.get_persisted_result("BTCUSDT", "1m", "feature", "returns")
        assert retrieved == data
        
    @pytest.mark.asyncio
    async def test_get_expired_result(self):
        calc = create_frequent_calculator()
        
        data = b"test data"
        ttl = timedelta(milliseconds=100)
        await calc.persist_result("BTCUSDT", "1m", "feature", "returns", data, ttl)
        
        # Wait for expiration
        import asyncio
        await asyncio.sleep(0.2)
        
        retrieved = await calc.get_persisted_result("BTCUSDT", "1m", "feature", "returns")
        assert retrieved is None
        
    def test_serialize_deserialize_feature_results(self):
        calc = create_frequent_calculator()
        
        results = [
            FeatureResult(name="returns", value=0.01, timestamp=datetime.now()),
            FeatureResult(name="returns", value=0.02, timestamp=datetime.now()),
        ]
        
        serialized = calc.serialize_feature_results(results)
        deserialized = calc.deserialize_feature_results(serialized)
        
        assert len(deserialized) == 2
        assert deserialized[0].name == "returns"
        assert deserialized[0].value == 0.01
        
    def test_serialize_deserialize_indicator_results(self):
        calc = create_frequent_calculator()
        
        results = [
            IndicatorResult(name="rsi_14", value=50.0, timestamp=datetime.now()),
            IndicatorResult(name="rsi_14", value=55.0, timestamp=datetime.now()),
        ]
        
        serialized = calc.serialize_indicator_results(results)
        deserialized = calc.deserialize_indicator_results(serialized)
        
        assert len(deserialized) == 2
        assert deserialized[0].name == "rsi_14"
        assert deserialized[0].value == 50.0
        
    @pytest.mark.asyncio
    async def test_clear_cache(self):
        calc = create_frequent_calculator()
        
        await calc.increment_request_count("BTCUSDT", "1m", "feature", "returns")
        await calc.persist_result("BTCUSDT", "1m", "feature", "returns", b"data")
        
        assert len(calc.get_frequent_calculations()) == 1
        assert await calc.get_persisted_result("BTCUSDT", "1m", "feature", "returns") == b"data"
        
        calc.clear_cache()
        
        assert len(calc.get_frequent_calculations()) == 0
        assert await calc.get_persisted_result("BTCUSDT", "1m", "feature", "returns") is None
