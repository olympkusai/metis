from langchain_core.tools import tool
import time as _time
import os
import json
from typing import Any, Dict
from app.grpc_client import CalculatorClient

# gRPC client configuration
GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = int(os.getenv("GRPC_PORT", "8081"))
GRPC_API_KEY = os.getenv("GRPC_API_KEY", "test-key-1")

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
    # Features
    "returns": 2,
    "log_returns": 2,
    "ma_7": 7,
    "ma_21": 21,
    "ma_50": 50,
    "volatility_7": 7,
    "volatility_21": 21,
    "volume_ratio": 21,
    "day_of_week": 1,
    "month": 1,
    "target_1d": 2,
    "momentum_30d": 43200,  # 30 dias * 24h * 60min
    "ewma_30d": 43200,      # 30 dias * 24h * 60min
    "ema_return_60": 60,
    # Indicators
    "rsi_14": 15,
    "macd": 35,
    "macd_signal": 35,
    "bb_upper": 20,
    "bb_lower": 20,
    "bb_width": 20,
    "sharpe": 20,
    "calmar": 20,
    "cvar_95": 2,
    "max_drawdown": 2,
    "bootstrap_20": 20,
}

# Features/indicators where 0 indicates warm-up (should filter)
_WARMUP_FILTER = {
    "target_1d", "ma_7", "ma_21", "ma_50", "volatility_7", "volatility_21",
    "returns", "log_returns", "bb_upper", "bb_lower", "bb_width"
}

# Risk indicators where 0 is legitimate (should not filter)
_RISK_INDICATORS = {
    "sharpe", "calmar", "cvar_95", "max_drawdown", "bootstrap_20"
}

def _get_grpc_client():
    """Get or create gRPC client instance."""
    if not hasattr(_get_grpc_client, '_client'):
        _get_grpc_client._client = CalculatorClient(host=GRPC_HOST, port=GRPC_PORT, api_key=GRPC_API_KEY)
    return _get_grpc_client._client

def _r(method: str, result: str, elapsed_ms: float = None) -> str:
    """Prefixa o resultado com método e tempo de resposta."""
    meta = f"[{method}]"
    if elapsed_ms is not None:
        meta += f" ({elapsed_ms:.0f}ms)"
    return f"{meta}\n{result}"

def _get_feature(symbol: str, feature: str, interval: str = "1m") -> tuple:
    """Busca feature específica via gRPC. Retorna (data, status, elapsed_ms, anomalies)."""
    import time
    start = time.time()
    client = _get_grpc_client()
    
    min_window = _MIN_WINDOW.get(feature)
    if not min_window:
        return [], 400, 0, []
    
    from_ts, to_ts = _default_timestamps(interval, feature_or_indicator=feature)
    
    # Get current price for normalization
    current_price = _get_current_price(symbol)
    
    try:
        response = client.get_features(
            symbol=symbol,
            interval=interval,
            features=[feature],
            from_time=from_ts,
            to_time=to_ts
        )
        elapsed_ms = (time.time() - start) * 1000
        
        if response:
            data = []
            anomalies = []
            for feature_result in response.features:
                if feature_result.name == feature:
                    timestamp_ms = int(feature_result.timestamp.seconds * 1000)
                    
                    # Validate timestamp
                    is_valid, error_msg = _validate_timestamp(timestamp_ms)
                    if not is_valid:
                        anomalies.append(f"Feature {feature}: {error_msg}")
                        continue
                    
                    # Filter 0 only for warm-up indicators
                    if feature not in _WARMUP_FILTER or feature_result.value != 0:
                        # Apply normalization
                        normalized_value = normalize_indicator(feature, feature_result.value, current_price)
                        data.append({
                            "feature_value": normalized_value,
                            "feature_value_raw": feature_result.value,
                            "timestamp": timestamp_ms,
                        })
            
            if not data:
                return [], 400, elapsed_ms, anomalies
            
            return data, 200, elapsed_ms, anomalies
        else:
            return [], 500, elapsed_ms, []
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return [], 500, elapsed_ms, []

def _get_indicator(symbol: str, indicator: str, interval: str = "1m") -> tuple:
    """Busca indicador específico via gRPC. Retorna (data, status, elapsed_ms, anomalies)."""
    import time
    start = time.time()
    client = _get_grpc_client()
    
    min_window = _MIN_WINDOW.get(indicator)
    if not min_window:
        return [], 400, 0, []
    
    from_ts, to_ts = _default_timestamps(interval, feature_or_indicator=indicator)
    
    # Get current price for normalization
    current_price = _get_current_price(symbol)
    
    try:
        response = client.get_indicators(
            symbol=symbol,
            interval=interval,
            indicators=[indicator],
            from_time=from_ts,
            to_time=to_ts
        )
        elapsed_ms = (time.time() - start) * 1000
        
        if response:
            data = []
            anomalies = []
            for indicator_result in response.indicators:
                if indicator_result.name == indicator:
                    timestamp_ms = int(indicator_result.timestamp.seconds * 1000)
                    
                    # Validate timestamp
                    is_valid, error_msg = _validate_timestamp(timestamp_ms)
                    if not is_valid:
                        anomalies.append(f"Indicator {indicator}: {error_msg}")
                        continue
                    
                    # Don't filter 0 for risk indicators or legitimate zero values
                    if indicator in _RISK_INDICATORS or indicator not in _WARMUP_FILTER or indicator_result.value != 0:
                        # Apply normalization
                        normalized_value = normalize_indicator(indicator, indicator_result.value, current_price)
                        data.append({
                            "indicator_value": normalized_value,
                            "indicator_value_raw": indicator_result.value,
                            "timestamp": timestamp_ms,
                        })
            
            if not data:
                return [], 400, elapsed_ms, anomalies
            
            return data, 200, elapsed_ms, anomalies
        else:
            return [], 500, elapsed_ms, []
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return [], 500, elapsed_ms, []

def _default_timestamps(interval: str, from_ts=None, to_ts=None, feature_or_indicator=None):
    """Retorna (from_ts, to_ts) garantindo pelo menos o mínimo de candles necessário."""
    if not to_ts:
        to_ts = int(_time.time() * 1000)
    if not from_ts:
        candle_ms = _INTERVAL_MS.get(interval, _INTERVAL_MS["1m"])
        min_candles = _MIN_WINDOW.get(feature_or_indicator, _MIN_CANDLES)
        from_ts = to_ts - (min_candles * candle_ms)
    return from_ts, to_ts

def _validate_timestamp(timestamp_ms: int, tolerance_seconds: int = 300) -> tuple[bool, str]:
    """Valida se o timestamp não está no futuro (com tolerância).
    
    Args:
        timestamp_ms: Timestamp em milissegundos
        tolerance_seconds: Tolerância em segundos para clock drift
    
    Returns:
        (is_valid, error_message)
    """
    current_ts = int(_time.time() * 1000)
    max_allowed_ts = current_ts + (tolerance_seconds * 1000)
    
    if timestamp_ms > max_allowed_ts:
        # Calculate how far in the future
        future_seconds = (timestamp_ms - current_ts) / 1000
        return False, f"Timestamp {timestamp_ms} is {future_seconds:.0f}s in the future (current: {current_ts})"
    
    return True, ""

def _get_interval_for_timeframe(timeframe: str) -> str:
    """Retorna o intervalo consistente para o timeframe de análise.
    
    Args:
        timeframe: "intraday", "daily", ou "weekly"
    
    Returns:
        Interval string (e.g., "1m", "1h", "1D")
    """
    timeframe_map = {
        "intraday": "1m",
        "daily": "1h",
        "weekly": "1D"
    }
    return timeframe_map.get(timeframe.lower(), "1h")

def _normalize_symbol(symbol: str) -> str:
    """Normaliza o símbolo para o formato BTCUSDT."""
    s = symbol.upper().strip()
    if s.endswith("USDT"):
        return s
    if s.endswith("USD"):
        return s[:-3] + "USDT"
    return s + "USDT"

def _get_current_price(symbol: str) -> float:
    """Obtém o preço atual do símbolo via gRPC."""
    try:
        client = _get_grpc_client()
        to_ts = int(_time.time() * 1000)
        from_ts = to_ts - 5 * 60 * 1_000  # 5 minutos
        response = client.get_ohlcv(
            symbol=symbol,
            interval="1m",
            from_time=from_ts,
            to_time=to_ts,
            limit=1
        )
        if response and response.candles:
            return response.candles[-1].close
    except Exception:
        pass
    return 0.0

def normalize_indicator(indicator_name: str, value: float, current_price: float) -> float:
    """Normaliza valor do indicador conforme seu tipo.
    
    Args:
        indicator_name: Nome do indicador/feature
        value: Valor bruto do indicador
        current_price: Preço atual do ativo para normalização baseada em preço
    
    Returns:
        Valor normalizado do indicador
    """
    # Já normalizados - não precisam de alteração
    if indicator_name in ["returns", "log_returns", "volatility_7", "volatility_21",
                          "volume_ratio", "bb_width", "sharpe", "calmar", "max_drawdown",
                          "day_of_week", "month", "target_1d", "momentum_30d",
                          "ema_return_60", "bootstrap_20"]:
        return value
    
    # RSI: mantido em escala 0-100 (padrão da indústria)
    if indicator_name == "rsi_14":
        return value
    
    # Preço-bruto: dividir pelo preço atual (exceto Bollinger Bands, mantidos em preço absoluto)
    if indicator_name in ["ma_7", "ma_21", "ma_50", "ewma_30d"]:
        return value / current_price if current_price > 0 else 0
    
    # Bollinger Bands: mantidos em preço absoluto para cálculo correto de %B
    if indicator_name in ["bb_upper", "bb_lower"]:
        return value
    
    # MACD: dividir pelo preço atual
    if indicator_name in ["macd", "macd_signal"]:
        return value / current_price if current_price > 0 else 0
    
    # CVaR: já é % mas pode ser ajustado
    if indicator_name == "cvar_95":
        return value  # já está em %
    
    return value

@tool
def get_live_price(symbol: str, interval: str = "1m") -> str:
    """Retorna o preço atual de uma criptomoeda via gRPC.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal do candle (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    import time
    start = time.time()
    symbol = _normalize_symbol(symbol)
    to_ts = int(_time.time() * 1000)
    from_ts = to_ts - 5 * _INTERVAL_MS.get(interval, _INTERVAL_MS["1m"])
    
    try:
        client = _get_grpc_client()
        response = client.get_ohlcv(
            symbol=symbol,
            interval=interval,
            from_time=from_ts,
            to_time=to_ts,
            limit=5
        )
        elapsed_ms = (time.time() - start) * 1000
        
        if response and response.candles:
            last = response.candles[-1]
            result = {
                "symbol": symbol,
                "interval": interval,
                "close": last.close,
                "high": last.high,
                "low": last.low,
                "volume": last.volume,
                "timestamp": last.close_time
            }
            return json.dumps(result)
        else:
            return json.dumps({"error": f"Sem dados recentes para {symbol} ({interval})."})
    except Exception as e:
        return json.dumps({"error": f"Erro ao obter preço: {e}"})

@tool
def get_indicators(symbol: str, interval: str = "1m") -> str:
    """Retorna indicadores de mercado (OHLCV, variação, volume) dos últimos 100 candles via gRPC.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    import time
    start = time.time()
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    
    try:
        client = _get_grpc_client()
        response = client.get_ohlcv(
            symbol=symbol,
            interval=interval,
            from_time=from_ts,
            to_time=to_ts,
            limit=100
        )
        elapsed_ms = (time.time() - start) * 1000
        
        if not response or not response.candles:
            return json.dumps({"error": f"Sem candles para {symbol} ({interval})."})
        
        candles = response.candles
        closes  = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        highs   = [c.high for c in candles]
        lows    = [c.low for c in candles]
        price_now   = closes[-1]
        price_open  = closes[0]
        pct_change  = (price_now - price_open) / price_open * 100

        candles_24h = {
            "1m": 1440,
            "3m": 480,
            "5m": 288,
            "15m": 96,
            "30m": 48,
            "1h": 24,
            "2h": 12,
            "4h": 6,
            "6h": 4,
            "12h": 2,
            "1d": 1,
            "1w": 1,
            "1M": 1,
        }
        n_candles_24h = candles_24h.get(interval, 24)
        vol_24h = sum(volumes[-n_candles_24h:]) if len(volumes) >= n_candles_24h else sum(volumes)

        # Data quality validation for extreme price changes
        anomalies = []
        # For 24h change (daily interval or equivalent)
        if interval in ["1d", "1D"]:
            if abs(pct_change) > 15:  # >15% daily change is extremely unusual for BTC in normal regime
                anomalies.append(f"Variação 24h extrema: {pct_change:.1f}% - possível erro de cálculo ou evento extremo")
        elif interval == "1h":
            # For hourly, check if the cumulative 24h change is extreme
            if abs(pct_change) > 15:
                anomalies.append(f"Variação no período extrema: {pct_change:.1f}% - verificar dados")

        result = {
            "symbol": symbol,
            "interval": interval,
            "candle_count": len(candles),
            "current_price": price_now,
            "pct_change": pct_change,
            "total_volume": vol_24h,
            "high": max(highs),
            "low": min(lows),
            "first_timestamp": candles[0].close_time,
            "last_timestamp": candles[-1].close_time,
            "anomalies": anomalies if anomalies else None
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Erro ao obter indicadores: {e}"})

@tool
def calculate_risk(symbol: str, interval: str = "1m") -> str:
    """Calcula métricas de risco (VaR, CVaR, Sharpe, drawdown) para um símbolo usando tools de indicadores.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    results = {}
    
    # Use _get_indicator for risk indicators
    data, st, ms, _ = _get_indicator(symbol, "cvar_95", interval)
    if st == 200 and data and data[-1].get("indicator_value") is not None:
        results["cvar_95"] = data[-1].get("indicator_value")
    
    data, st, ms, _ = _get_indicator(symbol, "sharpe", interval)
    if st == 200 and data and data[-1].get("indicator_value") is not None:
        results["sharpe"] = data[-1].get("indicator_value")
    
    data, st, ms, _ = _get_indicator(symbol, "max_drawdown", interval)
    if st == 200 and data and data[-1].get("indicator_value") is not None:
        results["max_drawdown"] = data[-1].get("indicator_value")
    
    data, st, ms, _ = _get_feature(symbol, "volatility_21", interval)
    if st == 200 and data and data[-1].get("feature_value") is not None:
        results["volatility_21"] = data[-1].get("feature_value")
    
    if not results:
        return json.dumps({"error": f"Métricas de risco não disponíveis para {symbol} ({interval})."})
    
    result = {
        "symbol": symbol,
        "interval": interval,
        "cvar_95": results.get('cvar_95'),
        "sharpe": results.get('sharpe'),
        "max_drawdown": results.get('max_drawdown'),
        "volatility_21": results.get('volatility_21')
    }
    return json.dumps(result)

@tool
def search_market_news(query: str) -> str:
    """Busca notícias recentes do mercado cripto. Placeholder até RAG ser implementado."""
    return (
        f"Notícias relevantes para '{query}':\n"
        f"  [1] Mercado cripto mostra recuperação após correção recente.\n"
        f"  [2] Analistas apontam resistência técnica em níveis-chave.\n"
        f"  [3] Volume de negociação aumentou nas últimas 24h.\n"
        f"  (Nota: RAG com ChromaDB será implementado na Fase 5)"
    )

@tool
def get_redis_history(symbol: str, interval: str = "1m", from_ts: int = None, to_ts: int = None) -> str:
    """Obtém dados históricos de candles via gRPC.
    
    Args:
        symbol: Símbolo do par de trading (ex: BTCUSDT)
        interval: Intervalo de tempo (1m, 5m, 15m, 1h, 1d). Padrão: 1m
        from_ts: Timestamp inicial em milissegundos (opcional)
        to_ts: Timestamp final em milissegundos (opcional)
    """
    import time
    start = time.time()
    from_ts, to_ts = _default_timestamps(interval, from_ts, to_ts)
    symbol = _normalize_symbol(symbol)
    
    try:
        client = _get_grpc_client()
        response = client.get_ohlcv(
            symbol=symbol,
            interval=interval,
            from_time=from_ts,
            to_time=to_ts,
            limit=100
        )
        elapsed_ms = (time.time() - start) * 1000
        
        if not response or not response.candles:
            return _r("gRPC", f"Nenhum dado encontrado para {symbol} no período especificado.", elapsed_ms)
        
        candles = response.candles
        result = f"Dados históricos de {symbol} ({interval}):\n"
        for candle in candles[:5]:
            result += f"  {candle.close_time}: O=${candle.open}, H=${candle.high}, L=${candle.low}, C=${candle.close}, V={candle.volume}\n"
        if len(candles) > 5:
            result += f"  ... e mais {len(candles) - 5} candles\n"
        return _r("gRPC", result, elapsed_ms)
    except Exception as e:
        return _r("gRPC", f"Erro ao obter histórico: {str(e)}")

@tool
def get_metrics() -> str:
    """Obtém métricas do sistema via gRPC."""
    import time
    start = time.time()
    
    elapsed_ms = (time.time() - start) * 1000
    result = "Métricas do sistema (gRPC):\n"
    result += f"  Status: Serviço gRPC conectado em {GRPC_HOST}:{GRPC_PORT}\n"
    result += f"  Nota: Endpoint de métricas detalhadas não disponível via gRPC\n"
    return _r("gRPC", result, elapsed_ms)

@tool
def get_feature_rsi(symbol: str, interval: str = "1m") -> str:
    """Obtém o valor atual do RSI (14 períodos) para um símbolo.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    data, status, ms, anomalies = _get_indicator(symbol, "rsi_14", interval)
    if status == 200 and data and data[-1].get("indicator_value") is not None:
        val = data[-1].get("indicator_value")
        # RSI em escala 0-100 (padrão da indústria)
        if val > 70:
            regime = "overbought"
        elif val < 30:
            regime = "oversold"
        else:
            regime = "neutral"
        result = {
            "symbol": symbol,
            "interval": interval,
            "rsi_14": val,
            "regime": regime
        }
        return json.dumps(result)
    if anomalies:
        return json.dumps({"error": f"RSI não disponível — violação de integridade temporal", "timestamp_anomalies": anomalies})
    return json.dumps({"error": f"RSI não disponível para {symbol} ({interval})."})

@tool
def get_feature_macd(symbol: str, interval: str = "1m") -> str:
    """Obtém os valores MACD (line, signal, histogram) para um símbolo.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    results = {}
    anomalies = []
    for ind in ["macd", "macd_signal"]:
        data, st, ms, ind_anomalies = _get_indicator(symbol, ind, interval)
        anomalies.extend(ind_anomalies)
        if st == 200 and data and data[-1].get("indicator_value") is not None:
            val = data[-1].get("indicator_value")
            results[ind] = val
            # Check for suspicious exact zero (indicates possible rounding error or calculation bug)
            if val == 0.0:
                anomalies.append(f"MACD {ind} é exatamente 0.0 - possível erro de arredondamento ou bug de cálculo")
    if not results:
        if anomalies:
            return json.dumps({"error": "MACD não disponível — violação de integridade temporal", "timestamp_anomalies": anomalies})
        return json.dumps({"error": f"MACD não disponível para {symbol} ({interval})."})
    macd_line = results.get('macd', 0)
    signal = results.get('macd_signal', 0)
    histogram = macd_line - signal
    if histogram > 0 and macd_line > signal:
        crossover = "bullish"
    elif histogram < 0 and macd_line < signal:
        crossover = "bearish"
    else:
        crossover = "none"
    result = {
        "symbol": symbol,
        "interval": interval,
        "macd_line": macd_line,
        "signal": signal,
        "histogram": histogram,
        "crossover": crossover,
        "anomalies": anomalies if anomalies else None
    }
    return json.dumps(result)

@tool
def get_feature_bollinger(symbol: str, interval: str = "1m") -> str:
    """Obtém as Bollinger Bands (upper, lower) para um símbolo.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    results = {}
    anomalies = []
    for ind in ["bb_upper", "bb_lower"]:
        data, st, ms, ind_anomalies = _get_indicator(symbol, ind, interval)
        anomalies.extend(ind_anomalies)
        if st == 200 and data and data[-1].get("indicator_value") is not None:
            results[ind] = data[-1].get("indicator_value")
    if not results:
        if anomalies:
            return json.dumps({"error": "Bollinger Bands não disponível — violação de integridade temporal", "timestamp_anomalies": anomalies})
        return json.dumps({"error": f"Bollinger Bands não disponível para {symbol} ({interval})."})
    upper = results.get('bb_upper', 0)
    lower = results.get('bb_lower', 0)
    
    # Consistency checks
    if upper > 0 and lower > 0:
        if upper <= lower:
            anomalies.append(f"Bollinger Bands inconsistent: upper ({upper}) <= lower ({lower})")
        width = (upper - lower) / upper
    else:
        width = 0
        if upper == 0 or lower == 0:
            anomalies.append(f"Bollinger Bands contain zero value: upper={upper}, lower={lower}")
    
    # Compute middle and %B (requires current price; BB values are in absolute price)
    middle = (upper + lower) / 2 if (upper > 0 and lower > 0) else 0.0
    current_price = _get_current_price(symbol)
    if upper > lower and upper > 0:
        pct_b = (current_price - lower) / (upper - lower)
        pct_b = max(0.0, min(1.0, pct_b))
        raw_pct_b = (current_price - lower) / (upper - lower)  # unclipped for breakout detection
    else:
        pct_b = 0.5
        raw_pct_b = 0.5
    
    result = {
        "symbol": symbol,
        "interval": interval,
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "pct_b": round(pct_b, 4),
        "width": width,
        "breakout": raw_pct_b > 1.0 or raw_pct_b < 0.0,
        "available": bool(results),
        "anomalies": anomalies if anomalies else None
    }
    return json.dumps(result)

@tool
def get_feature_volatility(symbol: str, interval: str = "1m") -> str:
    """Obtém a volatilidade (7 períodos) para um símbolo.
    Returns structured JSON output using log returns for crypto markets.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    data, status, ms, anomalies = _get_feature(symbol, "volatility_7", interval)
    if status == 200 and data and data[-1].get("feature_value") is not None:
        val = data[-1].get("feature_value")
        import math
        if interval == "1m":
            annualized = val * math.sqrt(252 * 1440)
        elif interval == "1h":
            annualized = val * math.sqrt(252 * 24)
        elif interval == "1d":
            annualized = val * math.sqrt(252)
        elif interval in ["1D", "1w", "1W"]:
            annualized = val * math.sqrt(52)
        else:
            annualized = val
        result = {
            "symbol": symbol,
            "interval": interval,
            "volatility_raw": val,
            "volatility_annualized": annualized,
            "data_interval": interval,
            "method": "log_returns"
        }
        return json.dumps(result)
    if anomalies:
        return json.dumps({"error": "Volatilidade não disponível — violação de integridade temporal", "timestamp_anomalies": anomalies})
    return json.dumps({"error": f"Volatilidade não disponível para {symbol} ({interval})."})

@tool
def get_feature_sharpe(symbol: str, interval: str = "1m") -> str:
    """Obtém o Sharpe ratio para um símbolo.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    data, status, ms, anomalies = _get_indicator(symbol, "sharpe", interval)
    if status == 200 and data and data[-1].get("indicator_value") is not None:
        val = data[-1].get("indicator_value")
        result = {
            "symbol": symbol,
            "interval": interval,
            "sharpe": val
        }
        return json.dumps(result)
    if anomalies:
        return json.dumps({"error": "Sharpe não disponível — violação de integridade temporal", "timestamp_anomalies": anomalies})
    return json.dumps({"error": f"Sharpe ratio não disponível para {symbol} ({interval})."})

@tool
def get_feature_cvar(symbol: str, interval: str = "1m") -> str:
    """Obtém o CVaR 95% para um símbolo.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    data, status, ms, anomalies = _get_indicator(symbol, "cvar_95", interval)
    if status == 200 and data and data[-1].get("indicator_value") is not None:
        val = data[-1].get("indicator_value")
        result = {
            "symbol": symbol,
            "interval": interval,
            "cvar_95": val
        }
        return json.dumps(result)
    if anomalies:
        return json.dumps({"error": "CVaR não disponível — violação de integridade temporal", "timestamp_anomalies": anomalies})
    return json.dumps({"error": f"CVaR 95% não disponível para {symbol} ({interval})."})

@tool
def get_feature_max_drawdown(symbol: str, interval: str = "1m") -> str:
    """Obtém o máximo drawdown para um símbolo.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    data, status, ms, anomalies = _get_indicator(symbol, "max_drawdown", interval)
    if status == 200 and data and data[-1].get("indicator_value") is not None:
        val = data[-1].get("indicator_value")
        result = {
            "symbol": symbol,
            "interval": interval,
            "max_drawdown": val
        }
        return json.dumps(result)
    if anomalies:
        return json.dumps({"error": "Max drawdown não disponível — violação de integridade temporal", "timestamp_anomalies": anomalies})
    return json.dumps({"error": f"Max drawdown não disponível para {symbol} ({interval})."})

@tool
def get_feature_sma(symbol: str, period: int = 20, interval: str = "1m") -> str:
    """Obtém o SMA para um símbolo.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        period: Período do SMA (7, 21, ou 50). Padrão: 20 (mapeia para 21)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    
    # Map period to available MA features
    period_map = {7: "ma_7", 20: "ma_21", 21: "ma_21", 50: "ma_50"}
    feat = period_map.get(period, "ma_21")
    
    data, status, ms, anomalies = _get_feature(symbol, feat, interval)
    if status == 200 and data and data[-1].get("feature_value") is not None:
        val = data[-1].get("feature_value")
        result = {
            "symbol": symbol,
            "interval": interval,
            "period": period,
            "sma": val
        }
        return json.dumps(result)
    if anomalies:
        return json.dumps({"error": f"SMA({period}) não disponível — violação de integridade temporal", "timestamp_anomalies": anomalies})
    return json.dumps({"error": f"SMA({period}) não disponível para {symbol} ({interval})."})

@tool
def get_feature_ema_return(symbol: str, interval: str = "1m") -> str:
    """Obtém o EMA de retornos (60 períodos) para um símbolo.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    data, status, ms, anomalies = _get_feature(symbol, "ema_return_60", interval)
    if status == 200 and data and data[-1].get("feature_value") is not None:
        val = data[-1].get("feature_value")
        result = {
            "symbol": symbol,
            "interval": interval,
            "ema_return_60": val
        }
        return json.dumps(result)
    if anomalies:
        return json.dumps({"error": "EMA Return não disponível — violação de integridade temporal", "timestamp_anomalies": anomalies})
    return json.dumps({"error": f"EMA Return (60) não disponível para {symbol} ({interval})."})

all_tools = [
    get_live_price,
    get_indicators,
    calculate_risk,
    search_market_news,
    get_redis_history,
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