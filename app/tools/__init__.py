"""
Tools otimizadas para LangChain com chamadas assíncronas paralelas à API.
Inclui timers detalhados para profiling de performance.
"""

from langchain_core.tools import tool
import time as _time
import os
import json
import asyncio
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from app.api_client import K0sApiClient
from app.config import get_settings

# API client configuration
settings = get_settings()
API_BASE_URL = str(settings.api_base_url)

# Performance tuning: connection pool limits
MAX_CONCURRENT_REQUESTS = settings.max_concurrent_requests
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
_client_lock = asyncio.Lock()


@dataclass
class TimingMetrics:
    """Métricas de tempo detalhadas para profiling."""
    total_ms: float = 0
    api_calls_ms: Dict[str, float] = field(default_factory=dict)
    normalization_ms: float = 0
    validation_ms: float = 0
    phases: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_phase(self, name: str, elapsed_ms: float, details: str = ""):
        self.phases.append({
            "name": name,
            "elapsed_ms": round(elapsed_ms, 2),
            "details": details
        })
    
    def to_dict(self) -> Dict:
        return {
            "total_ms": round(self.total_ms, 2),
            "api_calls_ms": {k: round(v, 2) for k, v in self.api_calls_ms.items()},
            "normalization_ms": round(self.normalization_ms, 2),
            "validation_ms": round(self.validation_ms, 2),
            "phases": self.phases
        }


_INTERVAL_MS = {
    "1m":  1        * 60 * 1_000,
    "3m":  3        * 60 * 1_000,
    "5m":  5        * 60 * 1_000,
    "15m": 15       * 60 * 1_000,
    "30m": 30       * 60 * 1_000,
    "1h":  1  * 60  * 60 * 1_000,
    "2h":  2  * 60  * 60 * 1_000,
    "4h":  4  * 60  * 60 * 1_000,
    "6h":  6  * 60  * 60 * 1_000,
    "8h":  8  * 60  * 60 * 1_000,
    "12h": 12 * 60  * 60 * 1_000,
    "1d":  1  * 24  * 60 * 60 * 1_000,
    "3d":  3  * 24  * 60 * 60 * 1_000,
    "1w":  7  * 24  * 60 * 60 * 1_000,
    "1M":  30 * 24  * 60 * 60 * 1_000,
}
_MIN_CANDLES = 100

# Minimum window (in candles) required for each feature/indicator
_MIN_WINDOW = {
    "returns": 2, "log_returns": 2, "ma_7": 7, "ma_21": 21, "ma_50": 50,
    "volatility_7": 7, "volatility_21": 21, "volume_ratio": 21,
    "day_of_week": 1, "month": 1, "target_1d": 2,
    "momentum_30d": 43200, "ewma_30d": 43200, "ema_return_60": 60,
    "rsi_14": 15, "macd": 35, "macd_signal": 35,
    "bb_upper": 20, "bb_lower": 20, "bb_width": 20,
    "sharpe": 20, "calmar": 20, "cvar_95": 2, "max_drawdown": 2, "bootstrap_20": 20,
}

_WARMUP_FILTER = {"target_1d", "ma_7", "ma_21", "ma_50", "volatility_7", "volatility_21",
                  "returns", "log_returns", "bb_upper", "bb_lower", "bb_width"}
_RISK_INDICATORS = {"sharpe", "calmar", "cvar_95", "max_drawdown", "bootstrap_20"}


async def _get_api_client() -> K0sApiClient:
    """Get or create HTTP API client instance with connection pooling (thread-safe)."""
    if not hasattr(_get_api_client, '_client'):
        async with _client_lock:
            if not hasattr(_get_api_client, '_client'):
                _get_api_client._client = K0sApiClient(base_url=API_BASE_URL, timeout=8.0)
    return _get_api_client._client


def _default_timestamps(interval: str, from_ts=None, to_ts=None, feature_or_indicator=None) -> Tuple[int, int]:
    """Retorna (from_ts, to_ts) garantindo pelo menos o mínimo de candles necessário."""
    if not to_ts:
        to_ts = int(_time.time() * 1000)
    if not from_ts:
        candle_ms = _INTERVAL_MS.get(interval, _INTERVAL_MS["1m"])
        min_candles = _MIN_WINDOW.get(feature_or_indicator, _MIN_CANDLES)
        from_ts = to_ts - (min_candles * candle_ms)
    print(f"[TOOLS] _default_timestamps: interval={interval}, feature={feature_or_indicator}, from_ts={from_ts}, to_ts={to_ts}, window={min_candles if feature_or_indicator else _MIN_CANDLES} candles")
    return from_ts, to_ts


def _normalize_symbol(symbol: str) -> str:
    """Normaliza o símbolo para o formato BTCUSDT."""
    s = symbol.upper().strip()
    if s.endswith("USDT"):
        return s
    if s.endswith("USD"):
        return s[:-3] + "USDT"
    return s + "USDT"


def normalize_indicator(indicator_name: str, value: float, current_price: float) -> float:
    """Normaliza valor do indicador conforme seu tipo.

    Normalização é aplicada para tornar indicadores comparáveis entre diferentes
    ativos com preços diferentes. A lógica é:

    - Indicadores já normalizados (returns, volatilidade, ratios): mantém valor original
    - Indicadores de preço absoluto (MA, EWMA): dividido pelo preço atual para normalizar
    - Indicadores de diferença de preço (MACD): dividido pelo preço atual
    - Indicadores de banda (Bollinger): mantidos em valor absoluto para cálculo de %B
    - Indicadores de risco (CVaR, Sharpe, Drawdown): mantidos em valor absoluto

    Args:
        indicator_name: Nome do indicador
        value: Valor bruto do indicador
        current_price: Preço atual do ativo para normalização

    Returns:
        Valor normalizado do indicador
    """
    if indicator_name in ["returns", "log_returns", "volatility_7", "volatility_21",
                          "volume_ratio", "bb_width", "sharpe", "calmar", "max_drawdown",
                          "day_of_week", "month", "target_1d", "momentum_30d",
                          "ema_return_60", "bootstrap_20"]:
        return value
    if indicator_name == "rsi_14":
        return value
    if indicator_name in ["ma_7", "ma_21", "ma_50", "ewma_30d"]:
        return value / current_price if current_price > 0 else 0
    if indicator_name in ["bb_upper", "bb_lower"]:
        return value
    if indicator_name in ["macd", "macd_signal"]:
        return value / current_price if current_price > 0 else 0
    if indicator_name == "cvar_95":
        return value
    return value


# ============ FUNÇÕES CORE ASSÍNCRONAS OTIMIZADAS ============

async def _get_current_price_async(symbol: str, metrics: Optional[TimingMetrics] = None) -> float:
    """Obtém o preço atual do símbolo via API REST (async)."""
    start = _time.time()
    phase_name = f"get_price_{symbol}"
    
    try:
        async with _semaphore:
            client = await _get_api_client()
            to_ts = int(_time.time() * 1000)
            from_ts = to_ts - 5 * 60 * 1_000  # 5 minutos
            
            api_start = _time.time()
            response = await client.get_ohlcv(
                symbol=symbol,
                interval="1m",
                from_time=from_ts,
                to_time=to_ts,
                limit=1
            )
            api_ms = (_time.time() - api_start) * 1000
            
            if metrics:
                metrics.api_calls_ms[phase_name] = api_ms
                metrics.add_phase(phase_name, api_ms, "1 candle OHLCV")
            
            if response and response.candles:
                return response.candles[-1].close
    except Exception as e:
        if metrics:
            metrics.add_phase(phase_name, (_time.time() - start) * 1000, f"error: {str(e)}")
    return 0.0


async def _get_feature_parallel(
    symbol: str, 
    feature: str, 
    interval: str = "1m",
    current_price: Optional[float] = None,
    metrics: Optional[TimingMetrics] = None
) -> tuple:
    """Busca feature específica via API REST com métricas detalhadas."""
    total_start = _time.time()
    
    min_window = _MIN_WINDOW.get(feature)
    if not min_window:
        return [], 400, 0, []
    
    from_ts, to_ts = _default_timestamps(interval, feature_or_indicator=feature)
    
    # Get current price if not provided (parallel optimization)
    price_start = _time.time()
    if current_price is None:
        current_price = await _get_current_price_async(symbol, metrics)
    price_ms = (_time.time() - price_start) * 1000
    
    try:
        async with _semaphore:
            client = await _get_api_client()
            api_start = _time.time()
            response = await client.get_features(
                symbol=symbol,
                interval=interval,
                features=[feature],
                from_time=from_ts,
                to_time=to_ts
            )
            api_ms = (_time.time() - api_start) * 1000
            
            if metrics:
                metrics.api_calls_ms[f"feature_{feature}"] = api_ms
        
        if not response:
            total_ms = (_time.time() - total_start) * 1000
            return [], 500, total_ms, []
        
        # Process data
        process_start = _time.time()
        data = []
        anomalies = []
        
        for feature_result in response.features:
            if feature_result.name == feature:
                timestamp_ms = feature_result.timestamp
                
                # Validation
                if timestamp_ms > int(_time.time() * 1000) + 300000:  # 5 min tolerance
                    anomalies.append(f"Feature {feature}: timestamp no futuro")
                    continue
                
                # Filter 0 only for warm-up indicators
                if feature not in _WARMUP_FILTER or feature_result.value != 0:
                    # Normalization
                    norm_start = _time.time()
                    normalized_value = normalize_indicator(feature, feature_result.value, current_price)
                    if metrics:
                        metrics.normalization_ms += (_time.time() - norm_start) * 1000
                    
                    data.append({
                        "feature_value": normalized_value,
                        "feature_value_raw": feature_result.value,
                        "timestamp": timestamp_ms,
                    })
        
        process_ms = (_time.time() - process_start) * 1000
        total_ms = (_time.time() - total_start) * 1000
        
        if metrics:
            metrics.add_phase(f"process_feature_{feature}", process_ms, f"{len(data)} values")
        
        if not data:
            return [], 400, total_ms, anomalies
        
        return data, 200, total_ms, anomalies
        
    except Exception as e:
        total_ms = (_time.time() - total_start) * 1000
        if metrics:
            metrics.add_phase(f"feature_{feature}_error", total_ms, str(e))
        return [], 500, total_ms, []


async def _get_indicator_parallel(
    symbol: str, 
    indicator: str, 
    interval: str = "1m",
    current_price: Optional[float] = None,
    metrics: Optional[TimingMetrics] = None
) -> tuple:
    """Busca indicador específico via API REST com métricas detalhadas."""
    total_start = _time.time()
    
    min_window = _MIN_WINDOW.get(indicator)
    if not min_window:
        return [], 400, 0, []
    
    from_ts, to_ts = _default_timestamps(interval, feature_or_indicator=indicator)
    
    # Get current price if not provided
    if current_price is None:
        current_price = await _get_current_price_async(symbol, metrics)
    
    try:
        async with _semaphore:
            client = await _get_api_client()
            api_start = _time.time()
            response = await client.get_indicators(
                symbol=symbol,
                interval=interval,
                indicators=[indicator],
                from_time=from_ts,
                to_time=to_ts
            )
            api_ms = (_time.time() - api_start) * 1000
            
            if metrics:
                metrics.api_calls_ms[f"indicator_{indicator}"] = api_ms
        
        if not response:
            total_ms = (_time.time() - total_start) * 1000
            return [], 500, total_ms, []
        
        # Process data
        process_start = _time.time()
        data = []
        anomalies = []
        
        for indicator_result in response.indicators:
            if indicator_result.name == indicator:
                timestamp_ms = indicator_result.timestamp
                
                # Validation
                if timestamp_ms > int(_time.time() * 1000) + 300000:
                    anomalies.append(f"Indicator {indicator}: timestamp no futuro")
                    continue
                
                # Don't filter 0 for risk indicators
                if indicator in _RISK_INDICATORS or indicator not in _WARMUP_FILTER or indicator_result.value != 0:
                    # Normalization
                    normalized_value = normalize_indicator(indicator, indicator_result.value, current_price)
                    
                    data.append({
                        "indicator_value": normalized_value,
                        "indicator_value_raw": indicator_result.value,
                        "timestamp": timestamp_ms,
                    })
        
        process_ms = (_time.time() - process_start) * 1000
        total_ms = (_time.time() - total_start) * 1000
        
        if metrics:
            metrics.validation_ms += process_ms
            metrics.add_phase(f"process_indicator_{indicator}", process_ms, f"{len(data)} values")
        
        if not data:
            return [], 400, total_ms, anomalies
        
        return data, 200, total_ms, anomalies
        
    except Exception as e:
        total_ms = (_time.time() - total_start) * 1000
        if metrics:
            metrics.add_phase(f"indicator_{indicator}_error", total_ms, str(e))
        return [], 500, total_ms, []


# ============ FUNÇÕES PARALELAS OTIMIZADAS ============

async def _get_multiple_features_parallel(
    symbol: str,
    features: List[str],
    interval: str = "1m",
    metrics: Optional[TimingMetrics] = None
) -> Dict[str, Tuple]:
    """Busca múltiplas features em paralelo."""
    start = _time.time()
    
    # Get price once for all features
    price = await _get_current_price_async(symbol, metrics)
    
    # Create tasks for all features
    tasks = [
        _get_feature_parallel(symbol, feature, interval, price, metrics)
        for feature in features
    ]
    
    # Execute all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Map results
    result_map = {}
    for feature, result in zip(features, results):
        if isinstance(result, Exception):
            result_map[feature] = ([], 500, 0, [str(result)])
        else:
            result_map[feature] = result
    
    if metrics:
        metrics.add_phase("batch_features", (_time.time() - start) * 1000, f"{len(features)} features")
    
    return result_map


async def _get_multiple_indicators_parallel(
    symbol: str,
    indicators: List[str],
    interval: str = "1m",
    metrics: Optional[TimingMetrics] = None
) -> Dict[str, Tuple]:
    """Busca múltiplos indicadores em paralelo."""
    start = _time.time()
    
    # Get price once for all indicators
    price = await _get_current_price_async(symbol, metrics)
    
    # Create tasks for all indicators
    tasks = [
        _get_indicator_parallel(symbol, indicator, interval, price, metrics)
        for indicator in indicators
    ]
    
    # Execute all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Map results
    result_map = {}
    for indicator, result in zip(indicators, results):
        if isinstance(result, Exception):
            result_map[indicator] = ([], 500, 0, [str(result)])
        else:
            result_map[indicator] = result
    
    if metrics:
        metrics.add_phase("batch_indicators", (_time.time() - start) * 1000, f"{len(indicators)} indicators")
    
    return result_map


# ============ TOOLS EXPOSTAS ============

@tool
async def get_live_price(symbol: str, interval: str = "1m") -> str:
    """Retorna o preço atual de uma criptomoeda via API REST."""
    metrics = TimingMetrics()
    start = _time.time()

    norm_symbol = _normalize_symbol(symbol)
    print(f"[TOOLS] get_live_price: symbol={symbol} -> {norm_symbol}, interval={interval}")
    to_ts = int(_time.time() * 1000)
    from_ts = to_ts - 5 * _INTERVAL_MS.get(interval, _INTERVAL_MS["1m"])

    try:
        async with _semaphore:
            client = await _get_api_client()
            api_start = _time.time()
            response = await client.get_ohlcv(
                symbol=norm_symbol,
                interval=interval,
                from_time=from_ts,
                to_time=to_ts,
                limit=5
            )
            api_ms = (_time.time() - api_start) * 1000
            metrics.api_calls_ms["ohlcv"] = api_ms
            print(f"[TOOLS] get_live_price API call took {api_ms:.2f}ms")

        if response and response.candles:
            last = response.candles[-1]
            result = {
                "symbol": norm_symbol,
                "interval": interval,
                "close": last.close,
                "high": last.high,
                "low": last.low,
                "volume": last.volume,
                "timestamp": last.close_time,
                "_timing": metrics.to_dict()
            }
            return json.dumps(result)
        else:
            return json.dumps({"error": f"Sem dados recentes para {norm_symbol} ({interval})."})
    except Exception as e:
        return json.dumps({"error": f"Erro ao obter preço: {e}", "_timing": metrics.to_dict()})


@tool
async def get_indicators(symbol: str, interval: str = "1m") -> str:
    """Retorna indicadores de mercado (OHLCV, variação, volume) dos últimos 100 candles."""
    metrics = TimingMetrics()
    start = _time.time()

    norm_symbol = _normalize_symbol(symbol)
    print(f"[TOOLS] get_indicators: symbol={symbol} -> {norm_symbol}, interval={interval}")
    from_ts, to_ts = _default_timestamps(interval, from_ts=None, to_ts=None)

    try:
        async with _semaphore:
            client = await _get_api_client()
            api_start = _time.time()
            response = await client.get_ohlcv(
                symbol=norm_symbol,
                interval=interval,
                from_time=from_ts,
                to_time=to_ts,
                limit=100
            )
            api_ms = (_time.time() - api_start) * 1000
            metrics.api_calls_ms["ohlcv_100"] = api_ms
            print(f"[TOOLS] get_indicators API call took {api_ms:.2f}ms")

        if not response or not response.candles:
            return json.dumps({"error": f"Sem candles para {norm_symbol} ({interval})."})

        process_start = _time.time()
        candles = response.candles
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        price_now = closes[-1]
        price_open = closes[0]
        pct_change = (price_now - price_open) / price_open * 100 if price_open != 0 else 0

        candles_24h_map = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}
        n_candles_24h = candles_24h_map.get(interval, 24)
        vol_24h = sum(volumes[-n_candles_24h:]) if len(volumes) >= n_candles_24h else sum(volumes)

        process_ms = (_time.time() - process_start) * 1000
        total_ms = (_time.time() - start) * 1000
        metrics.total_ms = total_ms
        metrics.add_phase("process_candles", process_ms, f"{len(candles)} candles")

        result = {
            "symbol": norm_symbol,
            "interval": interval,
            "candle_count": len(candles),
            "current_price": price_now,
            "pct_change": round(pct_change, 4),
            "total_volume": vol_24h,
            "high": max(highs),
            "low": min(lows),
            "first_timestamp": candles[0].close_time,
            "last_timestamp": candles[-1].close_time,
            "_timing": metrics.to_dict()
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Erro ao obter indicadores: {e}"})


@tool
async def calculate_risk(symbol: str, interval: str = "1m") -> str:
    """Calcula métricas de risco (VaR, CVaR, Sharpe, drawdown) em paralelo."""
    metrics = TimingMetrics()
    start = _time.time()

    norm_symbol = _normalize_symbol(symbol)
    print(f"[TOOLS] calculate_risk: symbol={symbol} -> {norm_symbol}, interval={interval}")

    # Busca indicadores e volatilidade em PARALELO real
    indicators = ["cvar_95", "sharpe", "max_drawdown"]
    results_map, vol_result = await asyncio.gather(
        _get_multiple_indicators_parallel(norm_symbol, indicators, interval, metrics),
        _get_feature_parallel(norm_symbol, "volatility_21", interval, None, metrics),
    )

    results = {}
    for ind, (data, st, ms, _) in results_map.items():
        if st == 200 and data and data[-1].get("indicator_value") is not None:
            results[ind] = data[-1].get("indicator_value")

    if vol_result[1] == 200 and vol_result[0] and vol_result[0][-1].get("feature_value") is not None:
        results["volatility_21"] = vol_result[0][-1].get("feature_value")

    total_ms = (_time.time() - start) * 1000
    metrics.total_ms = total_ms
    print(f"[TOOLS] calculate_risk total time: {total_ms:.2f}ms")

    if not results:
        return json.dumps({
            "error": f"Métricas de risco não disponíveis para {norm_symbol} ({interval}).",
            "_timing": metrics.to_dict()
        })

    result = {
        "symbol": norm_symbol,
        "interval": interval,
        "cvar_95": results.get('cvar_95'),
        "sharpe": results.get('sharpe'),
        "max_drawdown": results.get('max_drawdown'),
        "volatility_21": results.get('volatility_21'),
        "_timing": metrics.to_dict()
    }
    return json.dumps(result)


@tool
async def get_feature_rsi(symbol: str, interval: str = "1m") -> str:
    """Obtém o valor atual do RSI (14 períodos)."""
    metrics = TimingMetrics()
    norm_symbol = _normalize_symbol(symbol)
    print(f"[TOOLS] get_feature_rsi: symbol={symbol} -> {norm_symbol}, interval={interval}")

    data, status, ms, anomalies = await _get_indicator_parallel(norm_symbol, "rsi_14", interval, None, metrics)
    metrics.total_ms = ms
    print(f"[TOOLS] get_feature_rsi total time: {ms:.2f}ms")

    if status == 200 and data and data[-1].get("indicator_value") is not None:
        val = data[-1].get("indicator_value")
        regime = "overbought" if val > 70 else "oversold" if val < 30 else "neutral"
        return json.dumps({
            "symbol": norm_symbol,
            "interval": interval,
            "rsi_14": val,
            "regime": regime,
            "_timing": metrics.to_dict()
        })

    return json.dumps({
        "error": f"RSI não disponível para {norm_symbol} ({interval}).",
        "anomalies": anomalies,
        "_timing": metrics.to_dict()
    })


@tool
async def get_feature_macd(symbol: str, interval: str = "1m") -> str:
    """Obtém os valores MACD (line, signal, histogram) em paralelo."""
    metrics = TimingMetrics()
    start = _time.time()

    norm_symbol = _normalize_symbol(symbol)
    print(f"[TOOLS] get_feature_macd: symbol={symbol} -> {norm_symbol}, interval={interval}")

    indicators = ["macd", "macd_signal"]
    results_map = await _get_multiple_indicators_parallel(norm_symbol, indicators, interval, metrics)

    results = {}
    anomalies = []
    for ind, (data, st, ms, anoms) in results_map.items():
        if st == 200 and data and data[-1].get("indicator_value") is not None:
            val = data[-1].get("indicator_value")
            results[ind] = val
            if val == 0.0:
                anomalies.append(f"MACD {ind} é exatamente 0.0")
        anomalies.extend(anoms)

    if not results:
        return json.dumps({
            "error": f"MACD não disponível para {norm_symbol} ({interval}).",
            "_timing": metrics.to_dict()
        })

    macd_line = results.get('macd', 0)
    signal = results.get('macd_signal', 0)
    histogram = macd_line - signal
    crossover = "bullish" if histogram > 0 and macd_line > signal else \
                "bearish" if histogram < 0 and macd_line < signal else "none"

    metrics.total_ms = (_time.time() - start) * 1000
    print(f"[TOOLS] get_feature_macd total time: {metrics.total_ms:.2f}ms")

    return json.dumps({
        "symbol": norm_symbol,
        "interval": interval,
        "macd_line": macd_line,
        "signal": signal,
        "histogram": histogram,
        "crossover": crossover,
        "anomalies": anomalies if anomalies else None,
        "_timing": metrics.to_dict()
    })


@tool
async def get_feature_bollinger(symbol: str, interval: str = "1m") -> str:
    """Obtém as Bollinger Bands (upper, lower) em paralelo."""
    metrics = TimingMetrics()
    start = _time.time()

    norm_symbol = _normalize_symbol(symbol)
    print(f"[TOOLS] get_feature_bollinger: symbol={symbol} -> {norm_symbol}, interval={interval}")

    indicators = ["bb_upper", "bb_lower"]
    results_map = await _get_multiple_indicators_parallel(norm_symbol, indicators, interval, metrics)

    results = {}
    anomalies = []
    for ind, (data, st, ms, anoms) in results_map.items():
        if st == 200 and data and data[-1].get("indicator_value") is not None:
            results[ind] = data[-1].get("indicator_value")
        anomalies.extend(anoms)

    if not results:
        return json.dumps({
            "error": f"Bollinger Bands não disponível para {norm_symbol} ({interval}).",
            "_timing": metrics.to_dict()
        })

    upper = results.get('bb_upper', 0)
    lower = results.get('bb_lower', 0)

    # Sanity: Bollinger Bands exigem upper > lower > 0. Dados parciais/degenerados
    # devem ser reportados como erro para não poluir o pipeline.
    if not (upper > 0 and lower > 0 and upper > lower):
        return json.dumps({
            "error": f"Bollinger Bands inválido para {norm_symbol} ({interval}): "
                     f"upper={upper}, lower={lower}",
            "anomalies": anomalies,
            "_timing": metrics.to_dict()
        })
    width = (upper - lower) / upper

    middle = (upper + lower) / 2 if (upper > 0 and lower > 0) else 0.0
    current_price = await _get_current_price_async(norm_symbol, metrics)

    if upper > lower and upper > 0:
        pct_b = (current_price - lower) / (upper - lower)
        raw_pct_b = pct_b
        pct_b = max(0.0, min(1.0, pct_b))
    else:
        pct_b = 0.5
        raw_pct_b = 0.5

    metrics.total_ms = (_time.time() - start) * 1000
    print(f"[TOOLS] get_feature_bollinger total time: {metrics.total_ms:.2f}ms")

    return json.dumps({
        "symbol": norm_symbol,
        "interval": interval,
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "pct_b": round(pct_b, 4),
        "width": width,
        "breakout": raw_pct_b > 1.0 or raw_pct_b < 0.0,
        "anomalies": anomalies if anomalies else None,
        "_timing": metrics.to_dict()
    })


@tool
async def get_feature_volatility(symbol: str, interval: str = "1m") -> str:
    """Obtém a volatilidade (7 períodos)."""
    metrics = TimingMetrics()
    norm_symbol = _normalize_symbol(symbol)
    print(f"[TOOLS] get_feature_volatility: symbol={symbol} -> {norm_symbol}, interval={interval}")

    data, status, ms, anomalies = await _get_feature_parallel(norm_symbol, "volatility_7", interval, None, metrics)
    metrics.total_ms = ms
    print(f"[TOOLS] get_feature_volatility total time: {ms:.2f}ms")

    if status == 200 and data and data[-1].get("feature_value") is not None:
        val = data[-1].get("feature_value")
        import math

        annualization_map = {
            "1m": math.sqrt(252 * 1440),
            "1h": math.sqrt(252 * 24),
            "1d": math.sqrt(252),
            "1D": math.sqrt(52),
            "1w": math.sqrt(52)
        }
        annualized = val * annualization_map.get(interval, 1.0)

        return json.dumps({
            "symbol": norm_symbol,
            "interval": interval,
            "volatility_raw": val,
            "volatility_annualized": annualized,
            "data_interval": interval,
            "_timing": metrics.to_dict()
        })

    return json.dumps({
        "error": f"Volatilidade não disponível para {norm_symbol} ({interval}).",
        "_timing": metrics.to_dict()
    })


@tool
async def get_feature_sharpe(symbol: str, interval: str = "1m") -> str:
    """Obtém o Sharpe ratio."""
    metrics = TimingMetrics()
    norm_symbol = _normalize_symbol(symbol)

    data, status, ms, anomalies = await _get_indicator_parallel(norm_symbol, "sharpe", interval, None, metrics)
    metrics.total_ms = ms

    if status == 200 and data and data[-1].get("indicator_value") is not None:
        return json.dumps({
            "symbol": norm_symbol,
            "interval": interval,
            "sharpe": data[-1].get("indicator_value"),
            "_timing": metrics.to_dict()
        })

    return json.dumps({
        "error": f"Sharpe não disponível para {norm_symbol} ({interval}).",
        "_timing": metrics.to_dict()
    })


@tool
async def get_feature_cvar(symbol: str, interval: str = "1m") -> str:
    """Obtém o CVaR 95%."""
    metrics = TimingMetrics()
    norm_symbol = _normalize_symbol(symbol)

    data, status, ms, anomalies = await _get_indicator_parallel(norm_symbol, "cvar_95", interval, None, metrics)
    metrics.total_ms = ms

    if status == 200 and data and data[-1].get("indicator_value") is not None:
        return json.dumps({
            "symbol": norm_symbol,
            "interval": interval,
            "cvar_95": data[-1].get("indicator_value"),
            "_timing": metrics.to_dict()
        })

    return json.dumps({
        "error": f"CVaR não disponível para {norm_symbol} ({interval}).",
        "_timing": metrics.to_dict()
    })


@tool
async def get_feature_max_drawdown(symbol: str, interval: str = "1m") -> str:
    """Obtém o máximo drawdown."""
    metrics = TimingMetrics()
    norm_symbol = _normalize_symbol(symbol)

    data, status, ms, anomalies = await _get_indicator_parallel(norm_symbol, "max_drawdown", interval, None, metrics)
    metrics.total_ms = ms

    if status == 200 and data and data[-1].get("indicator_value") is not None:
        return json.dumps({
            "symbol": norm_symbol,
            "interval": interval,
            "max_drawdown": data[-1].get("indicator_value"),
            "_timing": metrics.to_dict()
        })

    return json.dumps({
        "error": f"Max drawdown não disponível para {norm_symbol} ({interval}).",
        "_timing": metrics.to_dict()
    })


@tool
async def get_feature_sma(symbol: str, period: int = 20, interval: str = "1m") -> str:
    """Obtém o SMA."""
    metrics = TimingMetrics()
    norm_symbol = _normalize_symbol(symbol)

    period_map = {7: "ma_7", 20: "ma_21", 21: "ma_21", 50: "ma_50"}
    feat = period_map.get(period, "ma_21")

    data, status, ms, anomalies = await _get_feature_parallel(norm_symbol, feat, interval, None, metrics)
    metrics.total_ms = ms

    if status == 200 and data and data[-1].get("feature_value_raw") is not None:
        raw = data[-1].get("feature_value_raw")
        ratio = data[-1].get("feature_value")  # normalizado (sma/price)
        return json.dumps({
            "symbol": norm_symbol,
            "interval": interval,
            "period": period,
            "sma": raw,
            "sma_to_price_ratio": ratio,
            "_timing": metrics.to_dict()
        })

    return json.dumps({
        "error": f"SMA({period}) não disponível para {norm_symbol} ({interval}).",
        "_timing": metrics.to_dict()
    })


@tool
async def get_feature_ema_return(symbol: str, interval: str = "1m") -> str:
    """Obtém o EMA de retornos (60 períodos)."""
    metrics = TimingMetrics()
    norm_symbol = _normalize_symbol(symbol)

    data, status, ms, anomalies = await _get_feature_parallel(norm_symbol, "ema_return_60", interval, None, metrics)
    metrics.total_ms = ms

    if status == 200 and data and data[-1].get("feature_value") is not None:
        return json.dumps({
            "symbol": norm_symbol,
            "interval": interval,
            "ema_return_60": data[-1].get("feature_value"),
            "_timing": metrics.to_dict()
        })

    return json.dumps({
        "error": f"EMA Return não disponível para {norm_symbol} ({interval}).",
        "_timing": metrics.to_dict()
    })


@tool
async def get_ohlcv_history(symbol: str, interval: str = "1m", from_ts: int = None, to_ts: int = None) -> str:
    """Obtém dados históricos de candles OHLCV via API REST."""
    metrics = TimingMetrics()
    start = _time.time()

    from_ts, to_ts = _default_timestamps(interval, from_ts, to_ts)
    norm_symbol = _normalize_symbol(symbol)

    print(f"[TOOLS] get_ohlcv_history: symbol={symbol} -> {norm_symbol}, interval={interval}, from_ts={from_ts}, to_ts={to_ts}")

    try:
        async with _semaphore:
            client = await _get_api_client()
            api_start = _time.time()
            response = await client.get_ohlcv(
                symbol=norm_symbol,
                interval=interval,
                from_time=from_ts,
                to_time=to_ts,
                limit=100
            )
            api_ms = (_time.time() - api_start) * 1000
            metrics.api_calls_ms["ohlcv_history"] = api_ms
            print(f"[TOOLS] get_ohlcv_history API call took {api_ms:.2f}ms")

        if not response or not response.candles:
            return json.dumps({
                "error": f"Nenhum dado encontrado para {norm_symbol} no período especificado.",
                "_timing": metrics.to_dict()
            })

        candles = response.candles
        result = {
            "symbol": norm_symbol,
            "interval": interval,
            "candles_preview": [
                {
                    "time": c.close_time,
                    "O": c.open,
                    "H": c.high,
                    "L": c.low,
                    "C": c.close,
                    "V": c.volume
                }
                for c in candles[:5]
            ],
            "total_candles": len(candles),
            "_timing": metrics.to_dict()
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Erro ao obter histórico: {str(e)}"})


@tool
def get_metrics() -> str:
    """Obtém métricas do sistema."""
    return json.dumps({
        "api_base_url": API_BASE_URL,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "status": "ok"
    })




# Lista de todas as tools disponíveis
all_tools = [
    get_live_price,
    get_indicators,
    calculate_risk,
    get_ohlcv_history,
    get_metrics,
    get_feature_rsi,
    get_feature_macd,
    get_feature_bollinger,
    get_feature_volatility,
    get_feature_sharpe,
    get_feature_cvar,
    get_feature_max_drawdown,
    get_feature_sma,
    get_feature_ema_return,
]
