"""
Tests for deterministic intent classification in the orchestrator.

Estes testes garantem que a heurística de detecção de saudação NÃO produz
falsos positivos para perguntas de análise/forecast — bug corrigido em
abril/2026 onde substrings como "oi" em "bitcoin" disparavam saudação.
"""
import pytest

from app.agent.graph import _classify_intent, _normalize_text


# ───────── Normalization ─────────
class TestNormalizeText:
    def test_strips_accents(self):
        assert _normalize_text("amanhã") == "amanha"
        assert _normalize_text("previsão") == "previsao"
        assert _normalize_text("análise técnica") == "analise tecnica"

    def test_lowercases(self):
        assert _normalize_text("BITCOIN") == "bitcoin"
        assert _normalize_text("BtCuSdT") == "btcusdt"

    def test_removes_punctuation(self):
        assert _normalize_text("oi, tudo bem?") == "oi tudo bem"
        assert _normalize_text("BTC!!!") == "btc"

    def test_collapses_whitespace(self):
        assert _normalize_text("  hello   world  ") == "hello world"

    def test_empty_input(self):
        assert _normalize_text("") == ""
        assert _normalize_text("   ") == ""


# ───────── Bugs reproduzidos do log de produção ─────────
class TestRegressionFromProductionLog:
    """Casos exatos que estavam quebrando em produção."""

    def test_btc_forecast_question_is_task(self):
        # Log: "como vai estar o btcusdt amanhã?" → classificado como greeting (BUG)
        assert _classify_intent("como vai estar o btcusdt amanhã?") == "task"

    def test_bitcoin_now_question_is_task(self):
        # Log: "como está o bitcoin agora?" → classificado como greeting (BUG)
        assert _classify_intent("como está o bitcoin agora?") == "task"

    def test_analyze_bitcoin_today_is_task(self):
        # Log: "pode analisar o bitcoin hoje?" → classificado como greeting porque
        # "oi" estava em "bitcoin" via substring matching (BUG)
        assert _classify_intent("pode analisar o bitcoin hoje?") == "task"

    def test_pure_greeting_eae_tudo_bem(self):
        # Log: "eae tudo bem?" → corretamente greeting (deve continuar funcionando)
        assert _classify_intent("eae tudo bem?") == "greeting"


# ───────── Saudações puras ─────────
class TestPureGreetings:
    @pytest.mark.parametrize("phrase", [
        "oi",
        "olá",
        "ola",
        "eae",
        "eai",
        "opa",
        "hey",
        "hello",
        "hi",
        "tudo bem",
        "tudo bom",
        "como vai",
        "como está",
        "beleza",
        "bom dia",
        "boa tarde",
        "boa noite",
        "good morning",
        "how are you",
        "eae tudo bem",
        "oi tudo bem",
        "fala mano",
    ])
    def test_pure_greetings_classified_as_greeting(self, phrase):
        assert _classify_intent(phrase) == "greeting"
        assert _classify_intent(phrase + "?") == "greeting"
        assert _classify_intent(phrase + "!") == "greeting"
        assert _classify_intent(phrase.upper()) == "greeting"


# ───────── Perguntas técnicas ─────────
class TestTechnicalQuestions:
    @pytest.mark.parametrize("question", [
        "qual é o preço do bitcoin?",
        "qual a cotação do BTC?",
        "como está o ethereum?",
        "análise técnica do solana",
        "previsão do BTC para amanhã",
        "forecast for ETH",
        "RSI do bitcoin",
        "MACD bullish?",
        "devo comprar bitcoin?",
        "vai subir o ETH?",
        "what is the price of bitcoin",
        "bitcoin price now",
        "preço atual btc",
        "tendência do mercado",
        "explique blockchain",
        "o que é staking?",
        "como funciona DeFi?",
        "halving do bitcoin",
        "wallet recomendada para ETH",
    ])
    def test_technical_questions_classified_as_task(self, question):
        assert _classify_intent(question) == "task"


# ───────── Casos ambíguos / edge cases ─────────
class TestEdgeCases:
    def test_empty_input_is_greeting(self):
        # Default seguro: vazio → greeting (não tem nada para analisar)
        assert _classify_intent("") == "greeting"
        assert _classify_intent("   ") == "greeting"

    def test_none_input_is_greeting(self):
        # Robustez: None não deve lançar exceção, e é tratado como entrada vazia → greeting
        assert _classify_intent(None) == "greeting"

    def test_greeting_followed_by_task_is_task(self):
        # "oi, qual o preço do btc?" → tem "oi" mas tem "preço" e "btc"
        assert _classify_intent("oi, qual o preço do btc?") == "task"

    def test_greeting_followed_by_short_task_is_task(self):
        # "eae btc?" → starter de saudação MAS tem token técnico
        assert _classify_intent("eae btc?") == "task"

    def test_long_non_technical_text_is_task(self):
        # Texto longo sem palavras técnicas e que não começa com saudação
        # → task (default seguro)
        assert _classify_intent("preciso de ajuda urgente sobre algo importante") == "task"

    def test_oi_inside_word_does_not_match(self):
        # Bug original: "oi" em "bitcoin" disparava greeting via substring
        # Agora deve ser tokenizado e não casar
        assert _classify_intent("bitcoin") == "task"
        assert _classify_intent("loira") == "task"  # "oi" em "loira" não dispara
        assert _classify_intent("foi") == "task"  # "oi" em "foi" não dispara

    def test_como_vai_alone_is_greeting(self):
        # "como vai" sem complemento técnico = saudação
        assert _classify_intent("como vai") == "greeting"
        assert _classify_intent("como vai?") == "greeting"

    def test_como_vai_with_technical_is_task(self):
        # "como vai estar" + token técnico = task (forecast)
        assert _classify_intent("como vai estar o bitcoin") == "task"

    def test_como_esta_alone_is_greeting(self):
        assert _classify_intent("como está") == "greeting"
        assert _classify_intent("como está?") == "greeting"

    def test_como_esta_with_asset_is_task(self):
        assert _classify_intent("como está o bitcoin?") == "task"
        assert _classify_intent("como está o btc?") == "task"


# ───────── Variações de capitalização e pontuação ─────────
class TestCaseAndPunctuation:
    def test_uppercase_btc_question(self):
        assert _classify_intent("QUAL O PREÇO DO BTC?") == "task"

    def test_mixed_case_greeting(self):
        assert _classify_intent("EaE TuDo BeM?") == "greeting"

    def test_excessive_punctuation(self):
        assert _classify_intent("oi!!!") == "greeting"
        assert _classify_intent("BTC???") == "task"

    def test_trailing_whitespace(self):
        assert _classify_intent("  oi  ") == "greeting"
        assert _classify_intent("  bitcoin  ") == "task"
