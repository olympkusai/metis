"""Comprehensive tests for all calculator classes."""
import pytest
from datetime import datetime, timedelta
import math

from app.calculator.types import Candle, FeatureResult, IndicatorResult
from app.calculator.returns import ReturnsCalculator, LogReturnsCalculator
from app.calculator.moving_average import MovingAverageCalculator, ma_7, ma_21, ma_50
from app.calculator.volatility import VolatilityCalculator, volatility_7, volatility_21
from app.calculator.volume_ratio import VolumeRatioCalculator
from app.calculator.time_features import DayOfWeekCalculator, MonthCalculator
from app.calculator.target import TargetCalculator
from app.calculator.momentum import MomentumCalculator, momentum_30d
from app.calculator.ewma import EWMACalculator, ewma_30d
from app.calculator.ema_return import EMAReturnCalculator, ema_return_60
from app.calculator.rsi import RSICalculator, rsi_14
from app.calculator.macd import MACDCalculator, MACDSignalCalculator, macd, macd_signal
from app.calculator.bollinger_bands import BollingerBandsCalculator, BBUpperCalculator, BBLowerCalculator, bb
from app.calculator.cvar import CVaRCalculator, cvar_95
from app.calculator.drawdown import MaxDrawdownCalculator, new_max_drawdown_calculator
from app.calculator.risk_metrics import SharpeCalculator, CalmarCalculator, sharpe, calmar
from app.calculator.bootstrap import BootstrapCalculator, bootstrap_20
from app.calculator.aggregation import aggregate_ohlcv, INTERVAL_TO_MINUTES


def generate_test_candles(count: int) -> list[Candle]:
    """Generate test candles with realistic price movements."""
    candles: list[Candle] = []
    base_price = 100.0
    base_time = datetime.now() - timedelta(minutes=count)
    
    for i in range(count):
        # Simulate realistic price movement
        change = (base_price * 0.01) * ((i % 3) - 1) / 3  # Small random movement
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


class TestReturnsCalculator:
    """Tests for ReturnsCalculator."""
    
    def test_returns_calculation(self):
        candles = generate_test_candles(10)
        calc = ReturnsCalculator()
        results = calc.calculate(candles)
        
        assert len(results) == 10
        assert results[0].value == 0.0  # First candle has no return
        assert results[0].name == "returns"
        
    def test_returns_with_zero_price(self):
        candles = [
            Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                   open_price=0, high_price=0, low_price=0, close_price=0, base_volume=0, quote_volume=0),
            Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                   open_price=100, high_price=100, low_price=100, close_price=100, base_volume=0, quote_volume=0),
        ]
        calc = ReturnsCalculator()
        results = calc.calculate(candles)
        
        assert results[1].value == 0.0  # Zero previous price should return 0
        
    def test_returns_error_on_insufficient_data(self):
        candles = generate_test_candles(1)
        calc = ReturnsCalculator()
        with pytest.raises(ValueError, match="at least 2 candles"):
            calc.calculate(candles)


class TestLogReturnsCalculator:
    """Tests for LogReturnsCalculator."""
    
    def test_log_returns_calculation(self):
        candles = generate_test_candles(10)
        calc = LogReturnsCalculator()
        results = calc.calculate(candles)
        
        assert len(results) == 10
        assert results[0].value == 0.0  # First candle has no return
        assert results[0].name == "log_returns"
        
    def test_log_returns_with_negative_price(self):
        candles = [
            Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                   open_price=100, high_price=100, low_price=100, close_price=100, base_volume=0, quote_volume=0),
            Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                   open_price=-100, high_price=-100, low_price=-100, close_price=-100, base_volume=0, quote_volume=0),
        ]
        calc = LogReturnsCalculator()
        results = calc.calculate(candles)
        
        assert results[1].value == 0.0  # Negative price should return 0


class TestMovingAverageCalculator:
    """Tests for MovingAverageCalculator."""
    
    def test_ma_7_calculation(self):
        candles = generate_test_candles(50)
        calc = ma_7()
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data
        assert results[6].value > 0  # Should have value after warmup
        
    def test_ma_21_calculation(self):
        candles = generate_test_candles(50)
        calc = ma_21()
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data
        assert results[20].value > 0  # Should have value after warmup
        
    def test_ma_50_calculation(self):
        candles = generate_test_candles(100)
        calc = ma_50()
        results = calc.calculate(candles)
        
        assert len(results) == 100
        assert results[0].value == 0.0  # Not enough data
        assert results[49].value > 0  # Should have value after warmup
        
    def test_ma_error_on_insufficient_data(self):
        candles = generate_test_candles(5)
        calc = ma_21()
        with pytest.raises(ValueError, match="not enough candles"):
            calc.calculate(candles)
        
    def test_ma_constant_price(self):
        candles = [Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                        open_price=100, high_price=100, low_price=100, close_price=100, base_volume=0, quote_volume=0)
                  for _ in range(10)]
        calc = MovingAverageCalculator(5, "ma_5")
        results = calc.calculate(candles)
        
        assert math.isclose(results[5].value, 100.0, rel_tol=0.01)


class TestVolatilityCalculator:
    """Tests for VolatilityCalculator."""
    
    def test_volatility_7_calculation(self):
        candles = generate_test_candles(50)
        calc = volatility_7()
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data
        assert results[6].value >= 0  # Should have non-negative value after warmup
        
    def test_volatility_21_calculation(self):
        candles = generate_test_candles(50)
        calc = volatility_21()
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data
        assert results[20].value >= 0  # Should have non-negative value after warmup
        
    def test_volatility_constant_price(self):
        candles = [Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                        open_price=100, high_price=100, low_price=100, close_price=100, base_volume=0, quote_volume=0)
                  for _ in range(10)]
        calc = VolatilityCalculator(5)
        results = calc.calculate(candles)
        
        assert math.isclose(results[5].value, 0.0, abs_tol=0.01)


class TestVolumeRatioCalculator:
    """Tests for VolumeRatioCalculator."""
    
    def test_volume_ratio_calculation(self):
        candles = generate_test_candles(50)
        calc = VolumeRatioCalculator(20)
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data
        assert results[19].value > 0  # Should have value after warmup


class TestTimeFeatures:
    """Tests for time feature calculators."""
    
    def test_day_of_week_calculation(self):
        candles = generate_test_candles(10)
        calc = DayOfWeekCalculator()
        results = calc.calculate(candles)
        
        assert len(results) == 10
        for r in results:
            assert 0 <= r.value <= 6  # Monday=0, Sunday=6
            assert r.name == "day_of_week"
            
    def test_month_calculation(self):
        candles = generate_test_candles(10)
        calc = MonthCalculator()
        results = calc.calculate(candles)
        
        assert len(results) == 10
        for r in results:
            assert 1 <= r.value <= 12
            assert r.name == "month"


class TestTargetCalculator:
    """Tests for TargetCalculator."""
    
    def test_target_calculation(self):
        candles = generate_test_candles(50)
        calc = TargetCalculator(1)
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[-1].value == 0.0  # Last candle has no future data
        
    def test_target_error_on_insufficient_data(self):
        candles = generate_test_candles(1)
        calc = TargetCalculator(1)
        with pytest.raises(ValueError, match="not enough candles"):
            calc.calculate(candles)


class TestMomentumCalculator:
    """Tests for MomentumCalculator."""
    
    def test_momentum_calculation(self):
        candles = generate_test_candles(50000)  # Need enough for 30d momentum
        calc = momentum_30d()
        results = calc.calculate(candles)
        
        assert len(results) == 50000
        assert results[0].value == 0.0  # Not enough data
        assert results[43200].value != 0.0  # Should have value after warmup


class TestEWMACalculator:
    """Tests for EWMACalculator."""
    
    def test_ewma_calculation(self):
        candles = generate_test_candles(50000)
        calc = ewma_30d()
        results = calc.calculate(candles)
        
        assert len(results) == 50000
        assert results[0].value > 0  # Should have value from first candle


class TestEMAReturnCalculator:
    """Tests for EMAReturnCalculator."""
    
    def test_ema_return_calculation(self):
        candles = generate_test_candles(100)
        calc = ema_return_60()
        results = calc.calculate(candles)
        
        assert len(results) == 100
        assert results[0].value == 0.0  # First return is 0


class TestRSICalculator:
    """Tests for RSICalculator."""
    
    def test_rsi_14_calculation(self):
        candles = generate_test_candles(50)
        calc = rsi_14()
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 50.0  # Neutral RSI for warmup
        assert results[14].value >= 0  # RSI should be non-negative
        assert results[14].value <= 100  # RSI should be <= 100
        
    def test_rsi_all_gains(self):
        candles = [Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                        open_price=float(i), high_price=float(i), low_price=float(i), close_price=float(i), base_volume=0, quote_volume=0)
                  for i in range(100, 120)]
        calc = rsi_14()
        results = calc.calculate(candles)
        
        assert math.isclose(results[19].value, 100.0, abs_tol=1.0)
        
    def test_rsi_all_losses(self):
        candles = [Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                        open_price=float(i), high_price=float(i), low_price=float(i), close_price=float(i), base_volume=0, quote_volume=0)
                  for i in range(120, 100, -1)]
        calc = rsi_14()
        results = calc.calculate(candles)
        
        assert math.isclose(results[19].value, 0.0, abs_tol=1.0)


class TestMACDCalculator:
    """Tests for MACDCalculator."""
    
    def test_macd_calculation(self):
        candles = generate_test_candles(50)
        calc = macd()
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data
        assert results[25].value != 0.0  # Should have value after warmup
        
    def test_macd_signal_calculation(self):
        candles = generate_test_candles(50)
        calc = macd_signal()
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data


class TestBollingerBands:
    """Tests for Bollinger Bands calculators."""
    
    def test_bb_upper_calculation(self):
        candles = generate_test_candles(50)
        calc = BBUpperCalculator(20, 2)
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data
        assert results[19].value > 0  # Should have value after warmup
        
    def test_bb_lower_calculation(self):
        candles = generate_test_candles(50)
        calc = BBLowerCalculator(20, 2)
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data
        
    def test_bb_width_calculation(self):
        candles = generate_test_candles(50)
        calc = BollingerBandsCalculator(20, 2)
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data
        assert results[19].value >= 0  # Width should be non-negative


class TestCVaRCalculator:
    """Tests for CVaRCalculator."""
    
    def test_cvar_95_calculation(self):
        candles = generate_test_candles(50)
        calc = cvar_95()
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data


class TestMaxDrawdownCalculator:
    """Tests for MaxDrawdownCalculator."""
    
    def test_max_drawdown_calculation(self):
        candles = generate_test_candles(50)
        calc = new_max_drawdown_calculator()
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value >= 0  # Drawdown should be non-negative


class TestRiskMetrics:
    """Tests for risk metrics calculators."""
    
    def test_sharpe_calculation(self):
        candles = generate_test_candles(50)
        calc = sharpe(20)
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data
        
    def test_calmar_calculation(self):
        candles = generate_test_candles(50)
        calc = calmar(20)
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data


class TestBootstrapCalculator:
    """Tests for BootstrapCalculator."""
    
    def test_bootstrap_20_calculation(self):
        candles = generate_test_candles(50)
        calc = bootstrap_20()
        results = calc.calculate(candles)
        
        assert len(results) == 50
        assert results[0].value == 0.0  # Not enough data
        assert results[20].value != 0.0  # Should have value after warmup (index 20, not 19)


class TestAggregation:
    """Tests for OHLCV aggregation."""
    
    def test_aggregate_5m(self):
        candles = generate_test_candles(100)
        aggregated = aggregate_ohlcv(candles, "5m")
        
        assert len(aggregated) <= 22  # Should be approximately 100/5, allow for time alignment
        for candle in aggregated:
            assert candle.interval == "5m"
            assert candle.high_price >= candle.low_price
            assert candle.base_volume > 0
            
    def test_aggregate_1h(self):
        candles = generate_test_candles(100)
        aggregated = aggregate_ohlcv(candles, "1h")
        
        assert len(aggregated) <= 2  # Should be approximately 100/60
        for candle in aggregated:
            assert candle.interval == "1h"
            
    def test_aggregate_1m_no_change(self):
        candles = generate_test_candles(10)
        aggregated = aggregate_ohlcv(candles, "1m")
        
        assert len(aggregated) == 10  # No aggregation for 1m
        
    def test_aggregate_error_on_empty_candles(self):
        with pytest.raises(ValueError, match="no candles provided"):
            aggregate_ohlcv([], "5m")
            
    def test_aggregate_error_on_invalid_interval(self):
        candles = generate_test_candles(10)
        with pytest.raises(ValueError, match="invalid target interval"):
            aggregate_ohlcv(candles, "invalid")


class TestCalculationCorrectness:
    """Tests for calculation correctness with known values."""
    
    def test_simple_returns_correctness(self):
        candles = [
            Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                   open_price=100, high_price=100, low_price=100, close_price=100, base_volume=0, quote_volume=0),
            Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                   open_price=105, high_price=105, low_price=105, close_price=105, base_volume=0, quote_volume=0),
            Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                   open_price=110, high_price=110, low_price=110, close_price=110, base_volume=0, quote_volume=0),
        ]
        calc = ReturnsCalculator()
        results = calc.calculate(candles)
        
        assert results[0].value == 0.0
        assert math.isclose(results[1].value, 0.05, rel_tol=0.001)  # (105-100)/100
        assert math.isclose(results[2].value, 0.0476, rel_tol=0.001)  # (110-105)/105
        
    def test_simple_ma_correctness(self):
        candles = [
            Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                   open_price=10, high_price=10, low_price=10, close_price=10, base_volume=0, quote_volume=0),
            Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                   open_price=20, high_price=20, low_price=20, close_price=20, base_volume=0, quote_volume=0),
            Candle(symbol="TEST", interval="1m", open_time=datetime.now(), close_time=datetime.now(),
                   open_price=30, high_price=30, low_price=30, close_price=30, base_volume=0, quote_volume=0),
        ]
        calc = MovingAverageCalculator(3, "ma_3")
        results = calc.calculate(candles)
        
        assert results[0].value == 0.0
        assert results[1].value == 0.0
        assert math.isclose(results[2].value, 20.0, rel_tol=0.01)  # (10+20+30)/3
