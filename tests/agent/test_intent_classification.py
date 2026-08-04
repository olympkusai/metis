"""
Tests for deterministic intent classification in the orchestrator.

Estes testes garantem que a heurística de detecção de saudação NÃO produz
falsos positivos para perguntas reais.
"""
import pytest

from metis.agent.graph import _classify_intent, _normalize_text


# ───────── Normalization ─────────
class TestNormalizeText:
    def test_strips_accents(self):
        assert _normalize_text("amanhã") == "amanha"
        assert _normalize_text("previsão") == "previsao"
        assert _normalize_text("análise técnica") == "analise tecnica"

    def test_lowercases(self):
        assert _normalize_text("Bom Dia") == "bom dia"
        assert _normalize_text("OLÁ") == "ola"

    def test_removes_punctuation(self):
        assert _normalize_text("oi, tudo bem?") == "oi tudo bem"
        assert _normalize_text("hello!!!") == "hello"

    def test_collapses_whitespace(self):
        assert _normalize_text("  oi   tudo   bem  ") == "oi tudo bem"

    def test_empty_input(self):
        assert _normalize_text("") == ""
        assert _normalize_text(None) == ""


# ───────── Casos de regressão ─────────
class TestRegressionFromProductionLog:
    def test_analyze_finance_today_is_task(self):
        assert _classify_intent("pode analisar minhas finanças hoje?") == "task"

    def test_pure_greeting_eae_tudo_bem(self):
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


# ───────── Casos ambíguos / edge cases ─────────
class TestEdgeCases:
    def test_empty_input_is_greeting(self):
        assert _classify_intent("") == "greeting"
        assert _classify_intent("   ") == "greeting"

    def test_none_input_is_greeting(self):
        assert _classify_intent(None) == "greeting"

    def test_long_non_greeting_text_is_task(self):
        assert _classify_intent("preciso de ajuda urgente sobre algo importante") == "task"

    def test_finance_question_is_task(self):
        assert _classify_intent("quanto gastei com comida esse mês?") == "task"
        assert _classify_intent("qual o progresso das minhas metas?") == "task"
        assert _classify_intent("quais contas estão vencendo?") == "task"

    def test_como_vai_alone_is_greeting(self):
        assert _classify_intent("como vai") == "greeting"
        assert _classify_intent("como vai?") == "greeting"

    def test_como_esta_alone_is_greeting(self):
        assert _classify_intent("como está") == "greeting"
        assert _classify_intent("como está?") == "greeting"


# ───────── Variações de capitalização e pontuação ─────────
class TestCaseAndPunctuation:
    def test_mixed_case_greeting(self):
        assert _classify_intent("EaE TuDo BeM?") == "greeting"

    def test_excessive_punctuation(self):
        assert _classify_intent("oi!!!") == "greeting"

    def test_trailing_whitespace(self):
        assert _classify_intent("  oi  ") == "greeting"
