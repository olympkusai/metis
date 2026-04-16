from langchain_core.tools import tool
import httpx

API_BASE_URL = "http://host.docker.internal:8080"
API_KEY = "dev_api_key_12345"
HEADERS = {"X-API-Key": API_KEY}

# Tools Binance (existentes)
@tool
def get_live_price(symbol: str) -> str:
    """Retorna o preço atual de uma criptomoeda em USD. Use para obter cotações em tempo real."""
    # Se symbol já termina com USDT, não adicionar novamente
    binance_symbol = symbol.upper() if symbol.upper().endswith("USDT") else f"{symbol.upper()}USDT"
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
    resp = httpx.get(url, timeout=10).json()
    if "price" not in resp:
        return f"Símbolo {symbol.upper()} não encontrado."
    return f"Preço atual de {symbol.upper()}: ${float(resp['price']):.2f} USDT"

@tool
def get_indicators(symbol: str) -> str:
    """Retorna indicadores técnicos (RSI, variação 24h, volume) de uma criptomoeda."""
    binance_symbol = symbol.upper() if symbol.upper().endswith("USDT") else f"{symbol.upper()}USDT"
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={binance_symbol}"
    resp = httpx.get(url, timeout=10).json()
    if "lastPrice" not in resp:
        return f"Indicadores para {symbol.upper()} não disponíveis."
    return (
        f"Indicadores de {symbol.upper()} (24h):\n"
        f"  Preço atual: ${float(resp['lastPrice']):.2f}\n"
        f"  Variação 24h: {float(resp['priceChangePercent']):.2f}%\n"
        f"  Volume 24h: {float(resp['volume']):.2f} {symbol.upper()}\n"
        f"  Máxima 24h: ${float(resp['highPrice']):.2f}\n"
        f"  Mínima 24h: ${float(resp['lowPrice']):.2f}"
    )

@tool
def calculate_risk(symbol: str, amount_usd: float) -> str:
    """Calcula métricas de risco básicas para um valor investido em uma criptomoeda."""
    binance_symbol = symbol.upper() if symbol.upper().endswith("USDT") else f"{symbol.upper()}USDT"
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={binance_symbol}"
    resp = httpx.get(url, timeout=10).json()
    if "priceChangePercent" not in resp:
        return "Dados insuficientes para cálculo de risco."
    volatility = abs(float(resp["priceChangePercent"]))
    var_95 = amount_usd * (volatility / 100) * 1.65
    return (
        f"Análise de risco para ${amount_usd:.2f} em {symbol.upper()}:\n"
        f"  Volatilidade 24h: {volatility:.2f}%\n"
        f"  VaR (95%): -${var_95:.2f}\n"
        f"  Risco estimado: {'Alto' if volatility > 5 else 'Moderado' if volatility > 2 else 'Baixo'}"
    )

@tool
def search_market_news(query: str) -> str:
    """Busca notícias recentes do mercado cripto relacionadas a uma query. Placeholder até RAG ser implementado."""
    return (
        f"Notícias relevantes para '{query}':\n"
        f"  [1] Mercado cripto mostra recuperação após correção recente.\n"
        f"  [2] Analistas apontam resistência técnica em níveis-chave.\n"
        f"  [3] Volume de negociação aumentou nas últimas 24h.\n"
        f"  (Nota: RAG com ChromaDB será implementado na Fase 5)"
    )

@tool
def get_top_cryptos(limit: int = 5) -> str:
    """Retorna o ranking das principais criptomoedas por volume de negociação na Binance."""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    resp = httpx.get(url, timeout=10).json()
    # Filtrar apenas pares USDT e ordenar por volume
    usdt_pairs = [r for r in resp if r["symbol"].endswith("USDT")]
    top = sorted(usdt_pairs, key=lambda x: float(x["quoteVolume"]), reverse=True)[:limit]
    result = f"Top {limit} criptomoedas por volume (24h):\n"
    for i, coin in enumerate(top, 1):
        symbol = coin["symbol"].replace("USDT", "")
        result += f"  {i}. {symbol}: ${float(coin['lastPrice']):.2f} ({float(coin['priceChangePercent']):.2f}%)\n"
    return result

# Tools da API interna (API.md)
@tool
def get_redis_history(symbol: str, interval: str = "1m", from_ts: int = None, to_ts: int = None) -> str:
    """Obtém dados históricos de candles do Redis cache. Use para análise histórica de preços.
    
    Args:
        symbol: Símbolo do par de trading (ex: BTCUSDT)
        interval: Intervalo de tempo (1m, 5m, 15m, 1h, 1d). Padrão: 1m
        from_ts: Timestamp inicial em milissegundos (opcional, padrão: 24h atrás)
        to_ts: Timestamp final em milissegundos (opcional, padrão: agora)
    """
    import time
    # Se timestamps não fornecidos, usar últimas 24 horas
    if not to_ts:
        to_ts = int(time.time() * 1000)
    if not from_ts:
        from_ts = to_ts - (24 * 60 * 60 * 1000)  # 24h atrás
    
    url = f"{API_BASE_URL}/api/v1/history/{symbol}"
    params = {"interval": interval, "from": from_ts, "to": to_ts}
    try:
        resp = httpx.get(url, headers=HEADERS, params=params, timeout=10).json()
        if not resp:
            return f"Nenhum dado encontrado para {symbol} no período especificado."
        result = f"Dados históricos de {symbol} ({interval}):\n"
        for candle in resp[:5]:  # Mostrar apenas os 5 primeiros
            result += f"  {candle['t']}: O=${candle['o']}, H=${candle['h']}, L=${candle['l']}, C=${candle['c']}, V={candle['v']}\n"
        if len(resp) > 5:
            result += f"  ... e mais {len(resp) - 5} candles\n"
        return result
    except Exception as e:
        return f"Erro ao obter histórico: {str(e)}"

@tool
def get_metrics() -> str:
    """Obtém métricas do sistema incluindo contagens de ingestão, erros de parse, status de shards, profundidade de fila e mensagens DLQ."""
    try:
        resp = httpx.get(f"{API_BASE_URL}/api/v1/metrics", headers=HEADERS, timeout=10).json()
        result = "Métricas do sistema:\n"
        result += f"  Contagens de ingestão: {resp.get('ingestion_counts', {})}\n"
        result += f"  Erros de parse: {resp.get('parse_errors', {})}\n"
        result += f"  Status de shards: {resp.get('shard_status', {})}\n"
        result += f"  Profundidade de fila: {resp.get('queue_depth', {})}\n"
        result += f"  Mensagens DLQ: {resp.get('dlq_messages', 0)}\n"
        return result
    except Exception as e:
        return f"Erro ao obter métricas: {str(e)}"

@tool
def get_all_features(symbol: str, interval: str = "1m", limit: int = 100, from_ts: int = None, to_ts: int = None) -> str:
    """Obtém valores de todas as features para um símbolo. Retorna dados de features em vez de apenas nomes.
    
    Args:
        symbol: Símbolo do par de trading (ex: BTCUSDT)
        interval: Intervalo de tempo (1m, 5m, 15m, 1h, 1d). Padrão: 1m
        limit: Número de registros para retornar. Padrão: 100
        from_ts: Timestamp inicial em milissegundos (opcional)
        to_ts: Timestamp final em milissegundos (opcional)
    """
    url = f"{API_BASE_URL}/api/v1/features/{symbol}"
    params = {"interval": interval, "limit": limit}
    if from_ts:
        params["from"] = from_ts
    if to_ts:
        params["to"] = to_ts
    try:
        resp = httpx.get(url, headers=HEADERS, params=params, timeout=10)
        if resp.status_code != 200:
            return f"Erro HTTP {resp.status_code}: {resp.text}"
        resp_json = resp.json()
        if resp_json is None:
            return "Resposta da API é None"
        data = resp_json.get("data")
        count = resp_json.get("count", 0)
        
        if data is None or count == 0:
            return f"Nenhuma feature encontrada para {symbol} ({interval}). Use recalculate_features para gerar dados."
        
        result = f"Features para {symbol} ({interval}): {count} registros\n"
        for item in data[:5]:  # Mostrar apenas os 5 primeiros
            result += f"  {item.get('timestamp', 'N/A')}: {item.get('feature_name', 'N/A')} = {item.get('feature_value', 'N/A')}\n"
        if len(data) > 5:
            result += f"  ... e mais {len(data) - 5} registros\n"
        return result
    except Exception as e:
        return f"Erro ao obter features: {str(e)}"

@tool
def recalculate_features(symbol: str, interval: str = "1m", from_ts: int = None, to_ts: int = None, param: str = None) -> str:
    """Recalcula e persiste features para um símbolo em um intervalo de tempo específico.
    
    Args:
        symbol: Símbolo do par de trading (ex: BTCUSDT)
        interval: Intervalo de tempo (1m, 5m, 15m, 1h, 1d). Padrão: 1m
        from_ts: Timestamp inicial em milissegundos (opcional, padrão: 24h atrás)
        to_ts: Timestamp final em milissegundos (opcional, padrão: agora)
        param: Lista separada por vírgula de features para recalcular (ex: sharpe,ema_return_60). Se não fornecido, recalcula todas.
    """
    import time
    # Se timestamps não fornecidos, usar últimas 24 horas
    if not to_ts:
        to_ts = int(time.time() * 1000)
    if not from_ts:
        from_ts = to_ts - (24 * 60 * 60 * 1000)  # 24h atrás
    
    url = f"{API_BASE_URL}/api/v1/features/{symbol}/recalculate"
    params = {"interval": interval, "from": from_ts, "to": to_ts}
    if param:
        params["param"] = param
    try:
        resp = httpx.post(url, headers=HEADERS, params=params, timeout=30).json()
        return (
            f"Recálculo de features para {symbol}:\n"
            f"  Candles processados: {resp.get('candles_processed', 0)}\n"
            f"  Features calculadas: {resp.get('features_count', 0)}\n"
            f"  Intervalo: {interval}\n"
            f"  Período: {from_ts} a {to_ts}\n"
        )
    except Exception as e:
        return f"Erro ao recalcular features: {str(e)}"

@tool
def get_feature_values(symbol: str, feature: str, interval: str = "1m", limit: int = 100, from_ts: int = None, to_ts: int = None) -> str:
    """Obtém valores históricos de uma feature específica.
    
    IMPORTANTE: O parâmetro 'feature' é OBRIGATÓRIO. Você deve especificar qual feature deseja buscar.
    
    Args:
        symbol: Símbolo do par de trading (ex: BTCUSDT) - OBRIGATÓRIO
        feature: Nome da feature (ex: sharpe, ema_return_60, rsi_14) - OBRIGATÓRIO
        interval: Intervalo de tempo (1m, 5m, 15m, 1h, 1d). Padrão: 1m
        limit: Número de registros para retornar. Padrão: 100
        from_ts: Timestamp inicial em milissegundos (opcional)
        to_ts: Timestamp final em milissegundos (opcional)
    """
    if not feature:
        return "Erro: parâmetro 'feature' é obrigatório. Especifique a feature desejada (ex: sharpe, rsi_14, etc.)"
    
    url = f"{API_BASE_URL}/api/v1/features/{symbol}/{feature}"
    params = {"interval": interval, "limit": limit}
    if from_ts:
        params["from"] = from_ts
    if to_ts:
        params["to"] = to_ts
    try:
        resp = httpx.get(url, headers=HEADERS, params=params, timeout=10)
        if resp.status_code != 200:
            return f"Erro HTTP {resp.status_code}: {resp.text}"
        resp_json = resp.json()
        data = resp_json.get("data")
        
        if data is None or len(data) == 0:
            return f"Nenhum valor encontrado para feature '{feature}' de {symbol} ({interval}). Use recalculate_features para gerar dados."
        
        result = f"Valores da feature '{feature}' para {symbol} ({interval}):\n"
        for item in data[:10]:  # Mostrar apenas os 10 primeiros
            result += f"  {item.get('timestamp', 'N/A')}: {item.get('feature_value', 'N/A')}\n"
        if len(data) > 10:
            result += f"  ... e mais {len(data) - 10} valores\n"
        return result
    except Exception as e:
        return f"Erro ao obter valores da feature: {str(e)}"

all_tools = [
    get_live_price,
    get_indicators,
    calculate_risk,
    search_market_news,
    get_top_cryptos,
    get_redis_history,
    get_metrics,
    recalculate_features,
    get_feature_values
]
