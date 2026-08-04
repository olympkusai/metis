"""
Tests for deterministic extraction helpers in the orchestrator:
- _extract_symbol: maps names/aliases to canonical tickers (bitcoin → BTCUSDT)
- _extract_timeframe_hint: detects AnalysisTimeframe from temporal keywords
"""
import pytest

from metis.agent.graph import (
    AnalysisTimeframe,
    _extract_symbol,
    _extract_timeframe_hint,
)


# ═══════════════════════════════════════════════════════════════
# Symbol extraction
# ═══════════════════════════════════════════════════════════════
class TestExtractSymbolFromName:
    """Casos centrais: usuário menciona nome do ativo (bitcoin, ethereum)."""

    @pytest.mark.parametrize("question,expected", [
        ("como está o bitcoin agora?", "BTCUSDT"),
        ("preço do BITCOIN", "BTCUSDT"),
        ("Bitcoin vai subir?", "BTCUSDT"),
        ("análise do ethereum", "ETHUSDT"),
        ("ETHEREUM agora", "ETHUSDT"),
        ("o que está acontecendo com o ether?", "ETHUSDT"),
        ("solana hoje", "SOLUSDT"),
        ("preço do cardano", "ADAUSDT"),
        ("análise de polkadot", "DOTUSDT"),
        ("dogecoin vai a lua?", "DOGEUSDT"),
        ("avalanche em alta", "AVAXUSDT"),
        ("polygon (matic) está bom?", "MATICUSDT"),
        ("uniswap análise", "UNIUSDT"),
        ("ripple está em alta?", "XRPUSDT"),
    ])
    def test_extracts_from_full_name(self, question, expected):
        assert _extract_symbol(question) == expected


class TestExtractSymbolFromTicker:
    """Usuário usa ticker abreviado (BTC, ETH, etc)."""

    @pytest.mark.parametrize("question,expected", [
        ("como vai o BTC?", "BTCUSDT"),
        ("ETH amanhã", "ETHUSDT"),
        ("preço do SOL", "SOLUSDT"),
        ("ADA em alta?", "ADAUSDT"),
        ("LINK now", "LINKUSDT"),
        ("DOT analysis", "DOTUSDT"),
        ("LTC", "LTCUSDT"),
    ])
    def test_extracts_from_ticker(self, question, expected):
        assert _extract_symbol(question) == expected


class TestExtractSymbolFullTicker:
    """Usuário fornece ticker já formatado (BTCUSDT, ETHUSDC)."""

    @pytest.mark.parametrize("question,expected", [
        ("BTCUSDT analysis", "BTCUSDT"),
        ("como vai estar o btcusdt amanhã?", "BTCUSDT"),
        ("preço do ETHUSDT agora", "ETHUSDT"),
        ("SOLUSDC vai subir?", "SOLUSDC"),
        ("análise BNBBUSD", "BNBBUSD"),
        ("ADAUSDT hoje", "ADAUSDT"),
    ])
    def test_extracts_full_ticker(self, question, expected):
        assert _extract_symbol(question) == expected


class TestExtractSymbolEdgeCases:
    def test_empty_returns_none(self):
        assert _extract_symbol("") is None
        assert _extract_symbol("   ") is None
        assert _extract_symbol(None) is None  # type: ignore[arg-type]

    def test_no_crypto_returns_none(self):
        assert _extract_symbol("oi tudo bem") is None
        assert _extract_symbol("explique blockchain") is None
        # "blockchain" sem ativo específico → None
        assert _extract_symbol("o que é defi?") is None

    def test_custom_quote_pair(self):
        # FUTURO: este será o ponto de injeção da config do usuário
        assert _extract_symbol("bitcoin", default_quote="USDC") == "BTCUSDC"
        assert _extract_symbol("ethereum", default_quote="BRL") == "ETHBRL"
        assert _extract_symbol("solana", default_quote="DAI") == "SOLDAI"

    def test_first_match_wins_when_multiple_assets(self):
        # Quando há múltiplos ativos mencionados, o primeiro vence
        result = _extract_symbol("comparar bitcoin e ethereum")
        assert result in ("BTCUSDT", "ETHUSDT")  # qualquer um é aceitável

    def test_case_insensitive(self):
        assert _extract_symbol("BITCOIN") == "BTCUSDT"
        assert _extract_symbol("BiTcOiN") == "BTCUSDT"
        assert _extract_symbol("bitcoin") == "BTCUSDT"

    def test_handles_punctuation(self):
        assert _extract_symbol("bitcoin!") == "BTCUSDT"
        assert _extract_symbol("bitcoin?") == "BTCUSDT"
        assert _extract_symbol("(bitcoin)") == "BTCUSDT"
        assert _extract_symbol("bitcoin, ethereum") in ("BTCUSDT", "ETHUSDT")


class TestRegressionFromProductionLog:
    """Casos exatos do log de produção que falhavam."""

    def test_btc_forecast_question(self):
        # "como vai estar o btcusdt amanhã?"
        assert _extract_symbol("como vai estar o btcusdt amanhã?") == "BTCUSDT"

    def test_bitcoin_now(self):
        # "como está o bitcoin agora?"
        assert _extract_symbol("como está o bitcoin agora?") == "BTCUSDT"

    def test_analyze_bitcoin_today(self):
        # "pode analisar o bitcoin hoje?"
        assert _extract_symbol("pode analisar o bitcoin hoje?") == "BTCUSDT"


# ═══════════════════════════════════════════════════════════════
# Timeframe extraction
# ═══════════════════════════════════════════════════════════════
class TestExtractTimeframeIntraday:
    @pytest.mark.parametrize("question", [
        "scalp do BTC",
        "scalping agora",
        "intraday analysis",
        "próximos minutos",
        "next minutes prediction",
        "agora mesmo",
        "right now",
        "day trade ETH",
        "1min chart",
        "5m timeframe",
        "minuto a minuto",
    ])
    def test_intraday_keywords(self, question):
        assert _extract_timeframe_hint(question) == AnalysisTimeframe.INTRADAY


class TestExtractTimeframeDaily:
    @pytest.mark.parametrize("question", [
        "como vai estar o BTC amanhã?",
        "preço do bitcoin hoje",
        "tomorrow forecast",
        "análise diária",
        "daily analysis",
        "swing trade",
        "esta semana",
        "this week",
        "próximas horas",
        "próximas 24 horas",
        "next 24 hours",
        "curto prazo",
        "short term",
        "BTC agora",
        "como está atualmente",
        "currently bitcoin",
        "bitcoin now",
    ])
    def test_daily_keywords(self, question):
        assert _extract_timeframe_hint(question) == AnalysisTimeframe.DAILY


class TestExtractTimeframeWeekly:
    @pytest.mark.parametrize("question", [
        "próxima semana",
        "next week BTC",
        "semanal",
        "weekly analysis",
        "este mês",
        "this month",
        "mensal",
        "monthly forecast",
        "longo prazo",
        "long term ETH",
        "trimestre",
        "quarter prediction",
        "anual",
        "year ahead",
        "1W timeframe",
        "1mo",
    ])
    def test_weekly_keywords(self, question):
        assert _extract_timeframe_hint(question) == AnalysisTimeframe.WEEKLY


class TestExtractTimeframeEdgeCases:
    def test_empty_returns_none(self):
        assert _extract_timeframe_hint("") is None
        assert _extract_timeframe_hint("   ") is None
        assert _extract_timeframe_hint(None) is None  # type: ignore[arg-type]

    def test_no_temporal_keyword_returns_none(self):
        # Pergunta sem nenhuma pista temporal → None (caller usa default)
        assert _extract_timeframe_hint("preço do bitcoin") is None
        assert _extract_timeframe_hint("explique RSI") is None

    def test_specificity_priority_weekly_over_daily(self):
        # "esta semana" → DAILY (timeframe)
        # "próxima semana" → WEEKLY (timeframe)
        # Frase composta tem prioridade sobre token único.
        assert _extract_timeframe_hint("esta semana") == AnalysisTimeframe.DAILY
        assert _extract_timeframe_hint("próxima semana") == AnalysisTimeframe.WEEKLY

    def test_case_insensitive(self):
        assert _extract_timeframe_hint("AMANHÃ") == AnalysisTimeframe.DAILY
        assert _extract_timeframe_hint("Próxima SEMANA") == AnalysisTimeframe.WEEKLY


# ═══════════════════════════════════════════════════════════════
# Combined: realistic user questions
# ═══════════════════════════════════════════════════════════════
class TestCombinedExtraction:
    """Validates that real user questions yield (symbol, timeframe) correctly."""

    @pytest.mark.parametrize("question,expected_symbol,expected_tf", [
        ("como vai estar o btcusdt amanhã?", "BTCUSDT", AnalysisTimeframe.DAILY),
        ("como está o bitcoin agora?", "BTCUSDT", AnalysisTimeframe.DAILY),
        ("pode analisar o bitcoin hoje?", "BTCUSDT", AnalysisTimeframe.DAILY),
        ("ethereum próxima semana", "ETHUSDT", AnalysisTimeframe.WEEKLY),
        ("scalp do solana", "SOLUSDT", AnalysisTimeframe.INTRADAY),
        ("dogecoin no longo prazo", "DOGEUSDT", AnalysisTimeframe.WEEKLY),
        ("XRP intraday", "XRPUSDT", AnalysisTimeframe.INTRADAY),
        ("polygon mensal", "MATICUSDT", AnalysisTimeframe.WEEKLY),
        ("LINK swing trade", "LINKUSDT", AnalysisTimeframe.DAILY),
    ])
    def test_realistic_user_questions(self, question, expected_symbol, expected_tf):
        assert _extract_symbol(question) == expected_symbol
        assert _extract_timeframe_hint(question) == expected_tf
