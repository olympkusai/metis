from langchain_core.tools import tool
import httpx
import time as _time
import os
import json
from typing import Any, Dict

API_BASE_URL = os.getenv("CRYPTO_DATA_API_URL", "http://localhost:8080")
API_KEY = "dev_api_key_12345"
HEADERS = {"X-API-Key": API_KEY}

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
_MIN_CANDLES = 100  # margem acima dos 61 necessários para features como sharpe

def _r(method: str, url: str, result: str, status: int = None, elapsed_ms: float = None) -> str:
    """Prefixa o resultado com rota, status e tempo de resposta."""
    meta = f"[{method} {url}]"
    if status is not None:
        meta += f" → {status}"
    if elapsed_ms is not None:
        meta += f" ({elapsed_ms:.0f}ms)"
    return f"{meta}\n{result}"

def _http_get(url: str, params: dict = None, timeout: int = 10):
    """GET com captura de status e tempo. Retorna (response, status_code, elapsed_ms)."""
    r = httpx.get(url, headers=HEADERS, params=params or {}, timeout=timeout)
    return r, r.status_code, r.elapsed.total_seconds() * 1000

def _http_post(url: str, params: dict = None, timeout: int = 30):
    """POST com captura de status e tempo. Retorna (response, status_code, elapsed_ms)."""
    r = httpx.post(url, headers=HEADERS, params=params or {}, timeout=timeout)
    return r, r.status_code, r.elapsed.total_seconds() * 1000

def _get_feature(symbol: str, feature: str, interval: str = "1m") -> tuple:
    """Busca feature específica com auto-recálculo via API. Retorna (data, status, elapsed_ms)."""
    from_ts, to_ts = _default_timestamps(interval)
    url = f"{API_BASE_URL}/api/v1/features/{symbol}/{feature}"
    params = {"interval": interval, "limit": 1, "from": from_ts, "to": to_ts}
    r, status, ms = _http_get(url, params=params)
    data = r.json().get("data") or []
    return data, status, ms

def _default_timestamps(interval: str, from_ts=None, to_ts=None):
    """Retorna (from_ts, to_ts) garantindo pelo menos _MIN_CANDLES candles."""
    if not to_ts:
        to_ts = int(_time.time() * 1000)
    if not from_ts:
        candle_ms = _INTERVAL_MS.get(interval, _INTERVAL_MS["1m"])
        from_ts = to_ts - (_MIN_CANDLES * candle_ms)
    return from_ts, to_ts

def _normalize_symbol(symbol: str) -> str:
    """Normaliza o símbolo para o formato BTCUSDT.
    Aceita: BTC, BTCUSD, BTCUSDT → retorna BTCUSDT
    """
    s = symbol.upper().strip()
    if s.endswith("USDT"):
        return s
    if s.endswith("USD"):
        return s[:-3] + "USDT"
    return s + "USDT"

@tool
def get_live_price(symbol: str, interval: str = "1m") -> str:
    """Retorna o preço atual de uma criptomoeda a partir do candle mais recente da API local.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal do candle (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    to_ts = int(_time.time() * 1000)
    from_ts = to_ts - 5 * _INTERVAL_MS.get(interval, _INTERVAL_MS["1m"])
    route = f"{API_BASE_URL}/api/v1/history/{symbol}?interval={interval}&from={from_ts}&to={to_ts}"
    try:
        r, status, ms = _http_get(
            f"{API_BASE_URL}/api/v1/history/{symbol}",
            params={"interval": interval, "from": from_ts, "to": to_ts},
        )
        resp = r.json()
        if not resp:
            return json.dumps({"error": f"Sem dados recentes para {symbol} ({interval})."})
        # Sort candles by timestamp to ensure chronological order
        resp = sorted(resp, key=lambda x: x["t"])
        last = resp[-1]
        result = {
            "symbol": symbol,
            "interval": interval,
            "close": float(last['c']),
            "high": float(last['h']),
            "low": float(last['l']),
            "volume": float(last['v']),
            "timestamp": last['t']
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Erro ao obter preço: {e}"})

@tool
def get_indicators(symbol: str, interval: str = "1m") -> str:
    """Retorna indicadores de mercado (OHLCV, variação, volume) dos últimos 100 candles via API local.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    try:
        r, status, ms = _http_get(
            f"{API_BASE_URL}/api/v1/history/{symbol}",
            params={"interval": interval, "from": from_ts, "to": to_ts},
        )
        candles = r.json()
        if not candles:
            return json.dumps({"error": f"Sem candles para {symbol} ({interval})."})
        # Sort candles by timestamp to ensure chronological order
        candles = sorted(candles, key=lambda x: x["t"])
        closes  = [float(c["c"]) for c in candles]
        volumes = [float(c["v"]) for c in candles]
        highs   = [float(c["h"]) for c in candles]
        lows    = [float(c["l"]) for c in candles]
        price_now   = closes[-1]
        price_open  = closes[0]
        pct_change  = (price_now - price_open) / price_open * 100

        # Calcula volume 24h correto baseado no intervalo
        candles_24h = {
            "1m": 1440,    # 24h * 60min
            "3m": 480,     # 24h * 20
            "5m": 288,     # 24h * 12
            "15m": 96,     # 24h * 4
            "30m": 48,     # 24h * 2
            "1h": 24,      # 24h
            "2h": 12,      # 24h / 2
            "4h": 6,       # 24h / 4
            "6h": 4,       # 24h / 6
            "12h": 2,      # 24h / 12
            "1d": 1,       # 24h
            "1w": 1,       # usa último candle disponível
            "1M": 1,       # usa último candle disponível
        }
        n_candles_24h = candles_24h.get(interval, 24)
        vol_24h = sum(volumes[-n_candles_24h:]) if len(volumes) >= n_candles_24h else sum(volumes)

        result = {
            "symbol": symbol,
            "interval": interval,
            "candle_count": len(candles),
            "current_price": price_now,
            "pct_change": pct_change,
            "total_volume": vol_24h,  # Volume das últimas 24h
            "high": max(highs),
            "low": min(lows),
            "first_timestamp": candles[0]['t'],
            "last_timestamp": candles[-1]['t']
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Erro ao obter indicadores: {e}"})

@tool
def calculate_risk(symbol: str, interval: str = "1m") -> str:
    """Calcula métricas de risco (VaR, CVaR, Sharpe, drawdown) para um símbolo usando tools de features.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    results = {}
    for feat in ["cvar_95", "sharpe", "max_drawdown", "volatility_20d"]:
        data, st, ms = _get_feature(symbol, feat, interval)
        if data and data[-1].get("feature_value") is not None:
            results[feat] = data[-1].get("feature_value")
    if not results:
        return json.dumps({"error": f"Métricas de risco não disponíveis para {symbol} ({interval})."})
    result = {
        "symbol": symbol,
        "interval": interval,
        "cvar_95": results.get('cvar_95'),
        "sharpe": results.get('sharpe'),
        "max_drawdown": results.get('max_drawdown'),
        "volatility_20d": results.get('volatility_20d')
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

# Tools da API interna (API.md)
@tool
def get_redis_history(symbol: str, interval: str = "1m", from_ts: int = None, to_ts: int = None) -> str:
    """Obtém dados históricos de candles do Redis cache. Use para análise histórica de preços.
    
    Args:
        symbol: Símbolo do par de trading (ex: BTCUSDT)
        interval: Intervalo de tempo (1m, 5m, 15m, 1h, 1d). Padrão: 1m
        from_ts: Timestamp inicial em milissegundos (opcional, padrão: 100 candles atrás)
        to_ts: Timestamp final em milissegundos (opcional, padrão: agora)
    """
    from_ts, to_ts = _default_timestamps(interval, from_ts, to_ts)
    symbol = _normalize_symbol(symbol)
    url = f"{API_BASE_URL}/api/v1/history/{symbol}"
    route = f"{url}?interval={interval}&from={from_ts}&to={to_ts}"
    params = {"interval": interval, "from": from_ts, "to": to_ts}
    try:
        r, status, ms = _http_get(url, params=params)
        resp = r.json()
        if not resp:
            return _r("GET", route, f"Nenhum dado encontrado para {symbol} no período especificado.", status, ms)
        # Sort candles by timestamp to ensure chronological order
        resp = sorted(resp, key=lambda x: x["t"])
        result = f"Dados históricos de {symbol} ({interval}):\n"
        for candle in resp[:5]:
            result += f"  {candle['t']}: O=${candle['o']}, H=${candle['h']}, L=${candle['l']}, C=${candle['c']}, V={candle['v']}\n"
        if len(resp) > 5:
            result += f"  ... e mais {len(resp) - 5} candles\n"
        return _r("GET", route, result, status, ms)
    except Exception as e:
        return _r("GET", route, f"Erro ao obter histórico: {str(e)}")

@tool
def get_metrics() -> str:
    """Obtém métricas do sistema incluindo contagens de ingestão, erros de parse, status de shards, profundidade de fila e mensagens DLQ."""
    route = f"{API_BASE_URL}/api/v1/metrics"
    try:
        r, status, ms = _http_get(route)
        resp = r.json()
        result = "Métricas do sistema:\n"
        result += f"  Contagens de ingestão: {resp.get('ingestion_counts', {})}\n"
        result += f"  Erros de parse: {resp.get('parse_errors', {})}\n"
        result += f"  Status de shards: {resp.get('shard_status', {})}\n"
        result += f"  Profundidade de fila: {resp.get('queue_depth', {})}\n"
        result += f"  Mensagens DLQ: {resp.get('dlq_messages', 0)}\n"
        return _r("GET", route, result, status, ms)
    except Exception as e:
        return _r("GET", route, f"Erro ao obter métricas: {str(e)}")

@tool
def get_feature_rsi(symbol: str, interval: str = "1m") -> str:
    """Obtém o valor atual do RSI (14 períodos) para um símbolo. Recalcula automaticamente se necessário.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    data, status, ms = _get_feature(symbol, "rsi_14", interval)
    if data and data[-1].get("feature_value") is not None:
        val = data[-1].get("feature_value")
        # Determine regime
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
    return json.dumps({"error": f"RSI não disponível para {symbol} ({interval})."})

@tool
def get_feature_macd(symbol: str, interval: str = "1m") -> str:
    """Obtém os valores MACD (line, signal, histogram) para um símbolo. Recalcula automaticamente se necessário.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    results = {}
    for feat in ["macd_line", "macd_signal", "macd_histogram"]:
        data, st, ms = _get_feature(symbol, feat, interval)
        if data and data[-1].get("feature_value") is not None:
            results[feat] = data[-1].get("feature_value")
    if not results:
        return json.dumps({"error": f"MACD não disponível para {symbol} ({interval})."})
    # Determine crossover
    macd_line = results.get('macd_line', 0)
    signal = results.get('macd_signal', 0)
    histogram = results.get('macd_histogram', 0)
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
        "crossover": crossover
    }
    return json.dumps(result)

@tool
def get_feature_bollinger(symbol: str, interval: str = "1m") -> str:
    """Obtém as Bollinger Bands (upper, middle, lower) para um símbolo. Recalcula automaticamente se necessário.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    results = {}
    for feat in ["bb_upper", "bb_middle", "bb_lower"]:
        data, st, ms = _get_feature(symbol, feat, interval)
        if data and data[-1].get("feature_value") is not None:
            results[feat] = data[-1].get("feature_value")
    if not results:
        return json.dumps({"error": f"Bollinger Bands não disponível para {symbol} ({interval})."})
    upper = results.get('bb_upper', 0)
    lower = results.get('bb_lower', 0)
    middle = results.get('bb_middle', 0)
    # Calculate %B (position within bands): %B = (price - lower) / (upper - lower)
    # Using middle as proxy for current price since we don't have it directly
    if upper != lower and upper > lower:
        pct_b = (middle - lower) / (upper - lower)
    else:
        pct_b = 0.5
    result = {
        "symbol": symbol,
        "interval": interval,
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "pct_b": pct_b
    }
    return json.dumps(result)

@tool
def get_feature_volatility(symbol: str, interval: str = "1m") -> str:
    """Obtém a volatilidade (20 períodos) para um símbolo. Recalcula automaticamente se necessário.
    Returns structured JSON output using log returns for crypto markets.
    Uses standard sqrt scaling without magic factors (institutional approach).

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    data, status, ms = _get_feature(symbol, "volatility_20d", interval)
    if data and data[-1].get("feature_value") is not None:
        val = data[-1].get("feature_value")
        # Annualize using standard sqrt scaling (no magic factors)
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
            "method": "log_returns"  # Indica que usa log returns
        }
        return json.dumps(result)
    return json.dumps({"error": f"Volatilidade não disponível para {symbol} ({interval})."})

@tool
def get_feature_sharpe(symbol: str, interval: str = "1m") -> str:
    """Obtém o Sharpe ratio para um símbolo. Recalcula automaticamente se necessário.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    data, status, ms = _get_feature(symbol, "sharpe", interval)
    if data and data[-1].get("feature_value") is not None:
        val = data[-1].get("feature_value")
        result = {
            "symbol": symbol,
            "interval": interval,
            "sharpe": val
        }
        return json.dumps(result)
    return json.dumps({"error": f"Sharpe ratio não disponível para {symbol} ({interval})."})

@tool
def get_feature_cvar(symbol: str, interval: str = "1m") -> str:
    """Obtém o CVaR 95% para um símbolo. Recalcula automaticamente se necessário.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    data, status, ms = _get_feature(symbol, "cvar_95", interval)
    if data and data[-1].get("feature_value") is not None:
        val = data[-1].get("feature_value")
        result = {
            "symbol": symbol,
            "interval": interval,
            "cvar_95": val
        }
        return json.dumps(result)
    return json.dumps({"error": f"CVaR 95% não disponível para {symbol} ({interval})."})

@tool
def get_feature_max_drawdown(symbol: str, interval: str = "1m") -> str:
    """Obtém o máximo drawdown para um símbolo. Recalcula automaticamente se necessário.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    data, status, ms = _get_feature(symbol, "max_drawdown", interval)
    if data and data[-1].get("feature_value") is not None:
        val = data[-1].get("feature_value")
        result = {
            "symbol": symbol,
            "interval": interval,
            "max_drawdown": val
        }
        return json.dumps(result)
    return json.dumps({"error": f"Max Drawdown não disponível para {symbol} ({interval})."})

@tool
def get_feature_sma(symbol: str, period: int = 20, interval: str = "1m") -> str:
    """Obtém o SMA para um símbolo. Recalcula automaticamente se necessário.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        period: Período do SMA (5 ou 20). Padrão: 20
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    feat = f"sma_{period}"
    data, status, ms = _get_feature(symbol, feat, interval)
    if data and data[-1].get("feature_value") is not None:
        val = data[-1].get("feature_value")
        result = {
            "symbol": symbol,
            "interval": interval,
            "period": period,
            "sma": val
        }
        return json.dumps(result)
    return json.dumps({"error": f"SMA({period}) não disponível para {symbol} ({interval})."})

@tool
def get_feature_ema_return(symbol: str, interval: str = "1m") -> str:
    """Obtém o EMA de retornos (60 períodos) para um símbolo. Recalcula automaticamente se necessário.
    Returns structured JSON output.

    Args:
        symbol: Símbolo do par (ex: BTC, BTCUSDT)
        interval: Janela temporal (1m, 5m, 15m, 1h, 1d). Padrão: 1m
    """
    symbol = _normalize_symbol(symbol)
    from_ts, to_ts = _default_timestamps(interval)
    data, status, ms = _get_feature(symbol, "ema_return_60", interval)
    if data and data[-1].get("feature_value") is not None:
        val = data[-1].get("feature_value")
        result = {
            "symbol": symbol,
            "interval": interval,
            "ema_return_60": val
        }
        return json.dumps(result)
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
