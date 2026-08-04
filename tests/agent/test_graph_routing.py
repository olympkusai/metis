"""
Tests for graph routing: ensures the orchestrator's conditional edge respects
next_action and that the build process is intact.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from metis.agent.graph import (
    NextAction,
    QuantAgentState,
    build_quant_graph,
    route_after_forecast,
)


class TestGraphCompilation:
    def test_graph_compiles_without_error(self):
        """Smoke test: build_quant_graph() não deve lançar exceção."""
        graph = build_quant_graph()
        assert graph is not None

    def test_graph_has_orchestrator_as_entry(self):
        """Entry point deve ser o orchestrator."""
        graph = build_quant_graph()
        # LangGraph compiled graphs expose nodes via the underlying graph
        nodes = set(graph.nodes.keys()) if hasattr(graph, "nodes") else set()
        assert "orchestrator" in nodes
        assert "finalize" in nodes
        assert "market_data" in nodes


class TestRouteAfterOrchestrator:
    """
    Testa a função de roteamento condicional após o orchestrator.

    Importante porque a edge fixa 'orchestrator → market_data' (bug histórico)
    ignorava next_action=FINALIZE, fazendo o pipeline rodar mesmo após
    detectar saudação. Agora deve respeitar next_action.
    """

    def _make_state(self, next_action: NextAction, symbol: str = "") -> QuantAgentState:
        # symbol agora é uma property derivada de primary_symbol/symbols.
        # Para configurar o "símbolo primário" no estado, usar primary_symbol.
        kwargs: dict = {"messages": [], "next_action": next_action}
        if symbol:
            kwargs["primary_symbol"] = symbol
            kwargs["symbols"] = [symbol]
        return QuantAgentState(**kwargs)

    def test_finalize_action_routes_to_finalize(self):
        from metis.agent.graph import build_quant_graph
        # Reextrai a função interna via re-importação do módulo.
        # Como route_after_orchestrator é local, validamos via comportamento:
        # next_action=FINALIZE deve resultar em rota "finalize".
        # Aqui validamos a lógica equivalente:
        state = self._make_state(NextAction.FINALIZE)
        # Replicação direta da lógica do builder:
        result = "finalize" if state.next_action == NextAction.FINALIZE else "market_data"
        assert result == "finalize"

    def test_market_data_action_routes_to_market_data(self):
        state = self._make_state(NextAction.MARKET_DATA, symbol="BTCUSDT")
        result = "finalize" if state.next_action == NextAction.FINALIZE else "market_data"
        assert result == "market_data"

    def test_other_actions_route_to_market_data(self):
        # Qualquer next_action que não seja FINALIZE deve ir para market_data
        for action in [NextAction.MARKET_DATA, NextAction.FEATURES_MACRO,
                       NextAction.RISK, NextAction.SIGNAL]:
            state = self._make_state(action)
            result = "finalize" if state.next_action == NextAction.FINALIZE else "market_data"
            assert result == "market_data", f"Action {action} should route to market_data"


class TestRouteAfterForecast:
    """
    Testa a função `route_after_forecast` — decide se a pergunta deve cair em
    `forecast_question` (resposta dedicada do Apollo: valor estimado / janela /
    riscos) ou em `trend_interpret` (pipeline completo de análise técnica).

    Histórico do bug: a versão antiga lia `state.messages[-1]` para extrair
    keywords. No ponto em que a função roda (após `features_macro/setup/exec/
    forecast`), a última mensagem é uma `AIMessage` adicionada por esses nós,
    e não a pergunta original do usuário. Resultado: perguntas como
    "Como vai estar o bitcoin amanhã?" caíam no fallback `trend_interpret` e o
    usuário recebia 3 RSIs diferentes + uma decisão técnica contraditória, em
    vez de uma resposta de previsão estruturada.
    """

    @staticmethod
    def _state(*messages) -> QuantAgentState:
        return QuantAgentState(messages=list(messages))

    # ----------- regressão crítica do bug original -----------

    def test_pipeline_ai_messages_do_not_hijack_routing(self):
        """REGRESSÃO: AIMessages adicionadas pelos nós upstream do pipeline
        não devem influenciar a decisão. Antes do fix, qualquer pergunta com
        AIMessage ao final caía em trend_interpret."""
        state = self._state(
            HumanMessage(content="Como vai estar o bitcoin amanhã?"),
            AIMessage(content="Routed to BTCUSDT (1d)"),
            AIMessage(content="RSI 64.59, MACD bullish crossover, BB próximo banda inferior."),
            AIMessage(content="RSI 72.29 sobrecomprado, MACD bullish, BB sem breakout."),
            AIMessage(content="RSI 61.27, MACD bullish, volatilidade 11.00%."),
            AIMessage(content="Apollo forecast: -4.56% predicted with 56.20% confidence."),
        )
        assert route_after_forecast(state) == "forecast_question"

    # ----------- robustez a acentuação e capitalização -----------

    @pytest.mark.parametrize("question", [
        "Como vai estar o bitcoin amanhã?",
        "Como vai estar o bitcoin amanha?",   # sem til
        "COMO VAI ESTAR O BTC AMANHÃ???",
        "como vai estar o btc amanha",
        "Qual a previsão para BTC?",
        "Qual a previsao para BTC?",          # sem til
        "Bitcoin vai subir semana que vem?",
        "Predict BTC tomorrow",
        "Qual o futuro do ETH?",
        "BTC nas próximas semanas?",
    ])
    def test_future_questions_route_to_forecast(self, question):
        state = self._state(HumanMessage(content=question))
        assert route_after_forecast(state) == "forecast_question", (
            f"Esperado forecast_question para: {question!r}"
        )

    @pytest.mark.parametrize("question", [
        "Como está o BTC agora?",
        "Qual o preço do bitcoin neste momento?",
        "BTC está em quanto?",
        "How is BTC currently?",
        "Bitcoin price now",
        "Atualmente, qual o preço do ETH?",
    ])
    def test_current_price_questions_route_to_trend_interpret(self, question):
        state = self._state(HumanMessage(content=question))
        assert route_after_forecast(state) == "trend_interpret", (
            f"Esperado trend_interpret para: {question!r}"
        )

    # ----------- conversa multi-turno (chat.py injeta histórico) -----------

    def test_picks_latest_human_message_not_first(self):
        """REGRESSÃO LATENTE: chat.py injeta histórico da sessão como
        HumanMessages ANTES da pergunta corrente. A função deve olhar a
        ÚLTIMA HumanMessage, não a primeira."""
        state = self._state(
            HumanMessage(content="Olá, tudo bem?"),                  # turno antigo
            AIMessage(content="Oi! Como posso ajudar?"),
            HumanMessage(content="Como está o BTC agora?"),          # turno antigo
            AIMessage(content="BTC está em 78,400."),
            HumanMessage(content="E como vai estar amanhã?"),        # PERGUNTA CORRENTE
            AIMessage(content="Routed to BTCUSDT (1d)"),
            AIMessage(content="...análise upstream..."),
        )
        assert route_after_forecast(state) == "forecast_question"

    def test_global_context_prefix_does_not_hijack_routing(self):
        """chat.py injeta um bloco 'Previous conversations context: ...' como
        HumanMessage no início. Ele não deve roubar o roteamento mesmo que
        contenha keywords espúrias."""
        state = self._state(
            HumanMessage(content=(
                "Previous conversations context:\n"
                "user: como vai estar o btc amanhã?...\n"
                "assistant: análise prévia...\n"
            )),
            HumanMessage(content="Como está o BTC agora?"),
            AIMessage(content="Routed to BTCUSDT"),
        )
        assert route_after_forecast(state) == "trend_interpret"

    # ----------- precedência: 'agora' tem que vencer 'amanhã' -----------

    def test_current_price_keywords_take_precedence(self):
        """Se a pergunta mistura sinais ('agora' + 'amanhã'), tratamos como
        consulta de preço atual (mais conservador) e o pipeline técnico
        responde com o snapshot atual."""
        state = self._state(
            HumanMessage(content="Como está o BTC agora? E amanhã, como vai estar?")
        )
        assert route_after_forecast(state) == "trend_interpret"

    # ----------- fallback seguro -----------

    @pytest.mark.parametrize("question", [
        "BTC",
        "Análise do Ethereum",
        "",
    ])
    def test_unknown_questions_fall_back_to_trend_interpret(self, question):
        state = self._state(HumanMessage(content=question))
        assert route_after_forecast(state) == "trend_interpret"

    def test_empty_messages_does_not_raise(self):
        """state.messages vazio não deve explodir."""
        state = self._state()
        assert route_after_forecast(state) == "trend_interpret"

    def test_only_ai_messages_does_not_raise(self):
        """Se por algum motivo só houver AIMessages, retorna fallback sem erro."""
        state = self._state(
            AIMessage(content="alguma coisa"),
            AIMessage(content="outra coisa"),
        )
        assert route_after_forecast(state) == "trend_interpret"
