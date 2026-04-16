from langchain.tools import tool
import httpx

@tool
def get_live_price(symbol: str) -> str:
    """Retorna o preço atual de uma criptomoeda em USD. Use para obter cotações em tempo real."""
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}USDT"
    resp = httpx.get(url, timeout=10).json()
    if "price" not in resp:
        return f"Símbolo {symbol.upper()} não encontrado."
    return f"Preço atual de {symbol.upper()}: ${float(resp['price']):.2f} USDT"

@tool
def get_indicators(symbol: str) -> str:
    """Retorna indicadores técnicos (RSI, variação 24h, volume) de uma criptomoeda."""
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol.upper()}USDT"
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
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol.upper()}USDT"
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

all_tools = [get_live_price, get_indicators, calculate_risk, search_market_news, get_top_cryptos]
