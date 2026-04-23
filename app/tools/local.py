"""Local tools using CalculationEngine - NO EXTERNAL API CALLS."""
from datetime import datetime, timedelta
from typing import Any
import json
import asyncio

from langchain_core.tools import tool

from app.calculator.engine import CalculationEngine, CalculationRequest, create_calculation_engine
from app.calculator.types import Candle
from app.storage import DatabasePool, MarketCandleQueries, MarketCandle


# Singleton CalculationEngine
_calculation_engine: CalculationEngine | None = None
_db_pool: DatabasePool | None = None


def get_calculation_engine() -> CalculationEngine:
    """Get or create the CalculationEngine singleton."""
    global _calculation_engine
    if _calculation_engine is None:
        _calculation_engine = create_calculation_engine()
    return _calculation_engine


def set_db_pool(pool: DatabasePool):
    """Set the database pool for local tools."""
    global _db_pool
    _db_pool = pool


def _convert_market_candles_to_candles(market_candles: list[MarketCandle]) -> list[Candle]:
    """Convert MarketCandle to Calculator Candle format."""
    candles = []
    for mc in market_candles:
        candles.append(Candle(
            open_time=mc.open_time,
            close_time=mc.close_time,
            open=mc.open_price,
            high=mc.high_price,
            low=mc.low_price,
            close=mc.close_price,
            volume=mc.base_volume,
            quote_volume=mc.quote_volume,
            trades=mc.trades_count,
        ))
    return candles


async def _get_candles_from_db(
    symbol: str,
    interval: str,
    count: int = 100
) -> list[Candle]:
    """Get candles from database using MarketCandleQueries."""
    if _db_pool is None:
        raise RuntimeError("Database pool not configured. Call set_db_pool() first.")
    
    queries = MarketCandleQueries(_db_pool)
    
    # Calculate time range based on interval
    interval_ms = {
        "1m": 1 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }.get(interval, 60 * 60 * 1000)
    
    end_time = datetime.now()
    start_time = end_time - timedelta(milliseconds=count * interval_ms)
    
    market_candles = await queries.get_candles(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        limit=count
    )
    
    return _convert_market_candles_to_candles(market_candles)


@tool
async def get_live_price(symbol: str, interval: str = "1m") -> str:
    """Get current price from local database (NO EXTERNAL API)."""
    try:
        candles = await _get_candles_from_db(symbol, interval, count=1)
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        last = candles[-1]
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "close": last.close,
            "high": last.high,
            "low": last.low,
            "volume": last.volume,
            "timestamp": last.close_time,
        })
    except Exception as e:
        return json.dumps({"error": f"Error getting price: {e}"})


@tool
async def get_indicators(symbol: str, interval: str = "1m") -> str:
    """Get market indicators from local database (NO EXTERNAL API)."""
    try:
        candles = await _get_candles_from_db(symbol, interval, count=100)
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        price_now = closes[-1]
        price_open = closes[0]
        pct_change = (price_now - price_open) / price_open * 100 if price_open != 0 else 0
        
        # Estimate 24h volume based on interval
        candles_24h_map = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}
        n_candles_24h = candles_24h_map.get(interval, 24)
        vol_24h = sum(volumes[-n_candles_24h:]) if len(volumes) >= n_candles_24h else sum(volumes)
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "candle_count": len(candles),
            "current_price": price_now,
            "pct_change": round(pct_change, 4),
            "total_volume": vol_24h,
            "high": max(highs),
            "low": min(lows),
            "first_timestamp": candles[0].close_time,
            "last_timestamp": candles[-1].close_time,
        })
    except Exception as e:
        return json.dumps({"error": f"Error getting indicators: {e}"})


@tool
async def calculate_risk(symbol: str, interval: str = "1m") -> str:
    """Calculate risk metrics using local CalculationEngine (NO EXTERNAL API)."""
    try:
        engine = get_calculation_engine()
        candles = await _get_candles_from_db(symbol, interval, count=100)
        
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        req = CalculationRequest(
            symbol=symbol,
            interval=interval,
            start_time=candles[0].close_time,
            end_time=candles[-1].close_time,
            indicators=["cvar_95", "sharpe", "max_drawdown", "volatility_21"]
        )
        
        response = await engine.process_request(req, candles)
        
        results = {}
        for ind_name, ind_results in response.indicators.items():
            if ind_results:
                results[ind_name] = ind_results[-1].value
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "cvar_95": results.get('cvar_95'),
            "sharpe": results.get('sharpe'),
            "max_drawdown": results.get('max_drawdown'),
            "volatility_21": results.get('volatility_21'),
        })
    except Exception as e:
        return json.dumps({"error": f"Error calculating risk: {e}"})


@tool
async def get_feature_rsi(symbol: str, interval: str = "1m") -> str:
    """Get RSI using local CalculationEngine (NO EXTERNAL API)."""
    try:
        engine = get_calculation_engine()
        candles = await _get_candles_from_db(symbol, interval, count=100)
        
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        req = CalculationRequest(
            symbol=symbol,
            interval=interval,
            start_time=candles[0].close_time,
            end_time=candles[-1].close_time,
            indicators=["rsi_14"]
        )
        
        response = await engine.process_request(req, candles)
        
        if "rsi_14" not in response.indicators or not response.indicators["rsi_14"]:
            return json.dumps({"error": f"RSI not available for {symbol} ({interval})."})
        
        val = response.indicators["rsi_14"][-1].value
        regime = "overbought" if val > 70 else "oversold" if val < 30 else "neutral"
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "rsi_14": val,
            "regime": regime,
        })
    except Exception as e:
        return json.dumps({"error": f"Error getting RSI: {e}"})


@tool
async def get_feature_macd(symbol: str, interval: str = "1m") -> str:
    """Get MACD using local CalculationEngine (NO EXTERNAL API)."""
    try:
        engine = get_calculation_engine()
        candles = await _get_candles_from_db(symbol, interval, count=100)
        
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        req = CalculationRequest(
            symbol=symbol,
            interval=interval,
            start_time=candles[0].close_time,
            end_time=candles[-1].close_time,
            indicators=["macd", "macd_signal"]
        )
        
        response = await engine.process_request(req, candles)
        
        macd_line = response.indicators.get("macd", [])[-1].value if response.indicators.get("macd") else 0
        signal = response.indicators.get("macd_signal", [])[-1].value if response.indicators.get("macd_signal") else 0
        histogram = macd_line - signal
        crossover = "bullish" if histogram > 0 and macd_line > signal else "bearish" if histogram < 0 and macd_line < signal else "none"
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "macd_line": macd_line,
            "signal": signal,
            "histogram": histogram,
            "crossover": crossover,
        })
    except Exception as e:
        return json.dumps({"error": f"Error getting MACD: {e}"})


@tool
async def get_feature_bollinger(symbol: str, interval: str = "1m") -> str:
    """Get Bollinger Bands using local CalculationEngine (NO EXTERNAL API)."""
    try:
        engine = get_calculation_engine()
        candles = await _get_candles_from_db(symbol, interval, count=100)
        
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        req = CalculationRequest(
            symbol=symbol,
            interval=interval,
            start_time=candles[0].close_time,
            end_time=candles[-1].close_time,
            indicators=["bb_upper", "bb_lower"]
        )
        
        response = await engine.process_request(req, candles)
        
        upper = response.indicators.get("bb_upper", [])[-1].value if response.indicators.get("bb_upper") else 0
        lower = response.indicators.get("bb_lower", [])[-1].value if response.indicators.get("bb_lower") else 0
        
        if not (upper > 0 and lower > 0 and upper > lower):
            return json.dumps({"error": f"Invalid Bollinger Bands for {symbol} ({interval})."})
        
        middle = (upper + lower) / 2
        current_price = candles[-1].close
        pct_b = (current_price - lower) / (upper - lower)
        pct_b = max(0.0, min(1.0, pct_b))
        width = (upper - lower) / upper
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "pct_b": round(pct_b, 4),
            "width": width,
            "breakout": pct_b > 1.0 or pct_b < 0.0,
        })
    except Exception as e:
        return json.dumps({"error": f"Error getting Bollinger Bands: {e}"})


@tool
async def get_feature_volatility(symbol: str, interval: str = "1m") -> str:
    """Get volatility using local CalculationEngine (NO EXTERNAL API)."""
    try:
        engine = get_calculation_engine()
        candles = await _get_candles_from_db(symbol, interval, count=100)
        
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        req = CalculationRequest(
            symbol=symbol,
            interval=interval,
            start_time=candles[0].close_time,
            end_time=candles[-1].close_time,
            features=["volatility_7"]
        )
        
        response = await engine.process_request(req, candles)
        
        if "volatility_7" not in response.features or not response.features["volatility_7"]:
            return json.dumps({"error": f"Volatility not available for {symbol} ({interval})."})
        
        val = response.features["volatility_7"][-1].value
        import math
        
        annualization_map = {
            "1m": math.sqrt(252 * 1440),
            "1h": math.sqrt(252 * 24),
            "1d": math.sqrt(252),
        }
        annualized = val * annualization_map.get(interval, 1.0)
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "volatility_raw": val,
            "volatility_annualized": annualized,
        })
    except Exception as e:
        return json.dumps({"error": f"Error getting volatility: {e}"})


@tool
async def get_feature_sharpe(symbol: str, interval: str = "1m") -> str:
    """Get Sharpe ratio using local CalculationEngine (NO EXTERNAL API)."""
    try:
        engine = get_calculation_engine()
        candles = await _get_candles_from_db(symbol, interval, count=100)
        
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        req = CalculationRequest(
            symbol=symbol,
            interval=interval,
            start_time=candles[0].close_time,
            end_time=candles[-1].close_time,
            indicators=["sharpe"]
        )
        
        response = await engine.process_request(req, candles)
        
        if "sharpe" not in response.indicators or not response.indicators["sharpe"]:
            return json.dumps({"error": f"Sharpe not available for {symbol} ({interval})."})
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "sharpe": response.indicators["sharpe"][-1].value,
        })
    except Exception as e:
        return json.dumps({"error": f"Error getting Sharpe: {e}"})


@tool
async def get_feature_cvar(symbol: str, interval: str = "1m") -> str:
    """Get CVaR 95% using local CalculationEngine (NO EXTERNAL API)."""
    try:
        engine = get_calculation_engine()
        candles = await _get_candles_from_db(symbol, interval, count=100)
        
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        req = CalculationRequest(
            symbol=symbol,
            interval=interval,
            start_time=candles[0].close_time,
            end_time=candles[-1].close_time,
            indicators=["cvar_95"]
        )
        
        response = await engine.process_request(req, candles)
        
        if "cvar_95" not in response.indicators or not response.indicators["cvar_95"]:
            return json.dumps({"error": f"CVaR not available for {symbol} ({interval})."})
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "cvar_95": response.indicators["cvar_95"][-1].value,
        })
    except Exception as e:
        return json.dumps({"error": f"Error getting CVaR: {e}"})


@tool
async def get_feature_max_drawdown(symbol: str, interval: str = "1m") -> str:
    """Get max drawdown using local CalculationEngine (NO EXTERNAL API)."""
    try:
        engine = get_calculation_engine()
        candles = await _get_candles_from_db(symbol, interval, count=100)
        
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        req = CalculationRequest(
            symbol=symbol,
            interval=interval,
            start_time=candles[0].close_time,
            end_time=candles[-1].close_time,
            indicators=["max_drawdown"]
        )
        
        response = await engine.process_request(req, candles)
        
        if "max_drawdown" not in response.indicators or not response.indicators["max_drawdown"]:
            return json.dumps({"error": f"Max drawdown not available for {symbol} ({interval})."})
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "max_drawdown": response.indicators["max_drawdown"][-1].value,
        })
    except Exception as e:
        return json.dumps({"error": f"Error getting max drawdown: {e}"})


@tool
async def get_feature_sma(symbol: str, period: int = 20, interval: str = "1m") -> str:
    """Get SMA using local CalculationEngine (NO EXTERNAL API)."""
    try:
        engine = get_calculation_engine()
        candles = await _get_candles_from_db(symbol, interval, count=100)
        
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        period_map = {7: "ma_7", 20: "ma_21", 21: "ma_21", 50: "ma_50"}
        feat = period_map.get(period, "ma_21")
        
        req = CalculationRequest(
            symbol=symbol,
            interval=interval,
            start_time=candles[0].close_time,
            end_time=candles[-1].close_time,
            features=[feat]
        )
        
        response = await engine.process_request(req, candles)
        
        if feat not in response.features or not response.features[feat]:
            return json.dumps({"error": f"SMA({period}) not available for {symbol} ({interval})."})
        
        raw = response.features[feat][-1].value
        current_price = candles[-1].close
        ratio = raw / current_price if current_price > 0 else 0
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "period": period,
            "sma": raw,
            "sma_to_price_ratio": ratio,
        })
    except Exception as e:
        return json.dumps({"error": f"Error getting SMA: {e}"})


@tool
async def get_feature_ema_return(symbol: str, interval: str = "1m") -> str:
    """Get EMA return using local CalculationEngine (NO EXTERNAL API)."""
    try:
        engine = get_calculation_engine()
        candles = await _get_candles_from_db(symbol, interval, count=100)
        
        if not candles:
            return json.dumps({"error": f"No candles found for {symbol} ({interval})."})
        
        req = CalculationRequest(
            symbol=symbol,
            interval=interval,
            start_time=candles[0].close_time,
            end_time=candles[-1].close_time,
            features=["ema_return_60"]
        )
        
        response = await engine.process_request(req, candles)
        
        if "ema_return_60" not in response.features or not response.features["ema_return_60"]:
            return json.dumps({"error": f"EMA Return not available for {symbol} ({interval})."})
        
        return json.dumps({
            "symbol": symbol,
            "interval": interval,
            "ema_return_60": response.features["ema_return_60"][-1].value,
        })
    except Exception as e:
        return json.dumps({"error": f"Error getting EMA return: {e}"})
