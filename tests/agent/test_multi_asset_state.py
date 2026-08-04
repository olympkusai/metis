"""
Tests for the multi-asset refactor of QuantAgentState.

Cobre:
- AssetState e helpers (`_patch_asset`, `state.primary`, `state.get_asset`)
- Heurística `_extract_all_symbols` (incluindo "X é melhor que Y?")
- Backward-compat: properties top-level continuam delegando ao primary
- Renderização do bloco de ativo (`_render_asset_block`)
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from metis.agent.graph import (
    AnalysisTimeframe,
    AssetState,
    QuantAgentState,
    RiskLevel,
    _extract_all_symbols,
    _extract_symbol,
    _latest_user_message,
    _patch_asset,
    _render_asset_block,
    _resolve_symbols,
)


class TestExtractAllSymbols:
    """Heurística determinística que extrai N símbolos da pergunta original.
    A ordem importa: o primeiro é o ATIVO PRIMÁRIO."""

    def test_single_alias(self):
        assert _extract_all_symbols("Como está o bitcoin?") == ["BTCUSDT"]

    def test_single_ticker(self):
        assert _extract_all_symbols("BTCUSDT está caindo?") == ["BTCUSDT"]

    def test_comparative_two_aliases(self):
        # Regressão chave: a pergunta que motivou o refator.
        assert _extract_all_symbols("XRP é melhor que Chainlink?") == ["XRPUSDT", "LINKUSDT"]

    def test_comparative_three_assets(self):
        result = _extract_all_symbols("Compare bitcoin com ethereum e SOL")
        assert result == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def test_ticker_and_alias_mixed(self):
        result = _extract_all_symbols("Análise de DOGEUSDT vs PEPE")
        assert result == ["DOGEUSDT", "PEPEUSDT"]

    def test_dedup_same_asset_repeated(self):
        # "btc" e "bitcoin" são o mesmo ativo — deve aparecer só uma vez.
        result = _extract_all_symbols("Bitcoin vai subir? Acho que BTC já subiu muito")
        assert result == ["BTCUSDT"]

    def test_empty_text(self):
        assert _extract_all_symbols("") == []

    def test_no_symbol_in_text(self):
        assert _extract_all_symbols("Explica o que é DeFi") == []

    def test_max_symbols_caps_extraction(self):
        # Pergunta muito longa não deve gerar pipeline gigante.
        result = _extract_all_symbols(
            "btc eth sol ada xrp doge bnb avax dot link",
            max_symbols=3,
        )
        assert len(result) == 3
        assert result[0] == "BTCUSDT"

    def test_extract_symbol_uses_first(self):
        # _extract_symbol é só um alias para _extract_all_symbols[0].
        assert _extract_symbol("XRP é melhor que Chainlink?") == "XRPUSDT"


class TestAssetStateAndPatching:
    """`_patch_asset` retorna um NOVO dict (não muta o original) e atualiza
    apenas o ativo alvo. É a base do merge per-asset usado pelos nodes."""

    def test_patch_creates_asset_when_missing(self):
        assets: dict[str, AssetState] = {}
        new_assets = _patch_asset(assets, "BTCUSDT", live_price=42000.0)
        assert new_assets["BTCUSDT"].symbol == "BTCUSDT"
        assert new_assets["BTCUSDT"].live_price == 42000.0
        # Original não foi mutado.
        assert assets == {}

    def test_patch_updates_existing_asset(self):
        assets = {"BTCUSDT": AssetState(symbol="BTCUSDT", live_price=10.0, rsi_14=50.0)}
        new_assets = _patch_asset(assets, "BTCUSDT", live_price=20.0)
        assert new_assets["BTCUSDT"].live_price == 20.0
        # rsi_14 preservado (não foi parte do patch).
        assert new_assets["BTCUSDT"].rsi_14 == 50.0
        # Original não mutou.
        assert assets["BTCUSDT"].live_price == 10.0

    def test_patch_does_not_touch_other_assets(self):
        assets = {
            "BTCUSDT": AssetState(symbol="BTCUSDT", live_price=10.0),
            "ETHUSDT": AssetState(symbol="ETHUSDT", live_price=2.0),
        }
        new_assets = _patch_asset(assets, "BTCUSDT", live_price=99.0)
        assert new_assets["ETHUSDT"].live_price == 2.0
        assert new_assets["BTCUSDT"].live_price == 99.0


class TestStateBackwardCompat:
    """Properties read-only de QuantAgentState delegam ao primary asset.
    Isso permite que código legado (`state.live_price`, `state.rsi_14`) continue
    funcionando sem refactor invasivo."""

    def test_empty_state_returns_safe_defaults(self):
        state = QuantAgentState()
        assert state.symbol == ""
        assert state.live_price is None
        assert state.rsi_14 is None
        assert state.primary == AssetState()

    def test_symbol_alias_resolves_to_primary_symbol(self):
        state = QuantAgentState(primary_symbol="BTCUSDT", symbols=["BTCUSDT"])
        assert state.symbol == "BTCUSDT"

    def test_top_level_reads_delegate_to_primary_asset(self):
        primary = AssetState(symbol="BTCUSDT", live_price=42000.0, rsi_14=65.0,
                             risk_level=RiskLevel.MODERATE)
        state = QuantAgentState(
            primary_symbol="BTCUSDT",
            symbols=["BTCUSDT"],
            assets={"BTCUSDT": primary},
        )
        assert state.live_price == 42000.0
        assert state.rsi_14 == 65.0
        assert state.risk_level == RiskLevel.MODERATE

    def test_get_asset_returns_empty_for_unknown(self):
        state = QuantAgentState()
        a = state.get_asset("BTCUSDT")
        assert a.symbol == "BTCUSDT"
        assert a.live_price is None

    def test_multi_asset_state_isolates_per_symbol(self):
        # Garante que dados de XRP não vazam para a leitura de LINK.
        state = QuantAgentState(
            primary_symbol="XRPUSDT",
            symbols=["XRPUSDT", "LINKUSDT"],
            assets={
                "XRPUSDT": AssetState(symbol="XRPUSDT", live_price=1.42),
                "LINKUSDT": AssetState(symbol="LINKUSDT", live_price=9.45),
            },
        )
        # state.live_price aponta para o PRIMÁRIO (XRP).
        assert state.live_price == 1.42
        # get_asset acessa cada um isoladamente.
        assert state.get_asset("XRPUSDT").live_price == 1.42
        assert state.get_asset("LINKUSDT").live_price == 9.45


class TestResolveSymbols:
    def test_uses_symbols_when_present(self):
        state = QuantAgentState(symbols=["BTCUSDT", "ETHUSDT"], primary_symbol="BTCUSDT")
        assert _resolve_symbols(state) == ["BTCUSDT", "ETHUSDT"]

    def test_falls_back_to_primary_symbol(self):
        # Caminho legado: alguém criou state só com primary_symbol.
        state = QuantAgentState(primary_symbol="BTCUSDT")
        assert _resolve_symbols(state) == ["BTCUSDT"]

    def test_returns_empty_when_nothing(self):
        assert _resolve_symbols(QuantAgentState()) == []


class TestRenderAssetBlock:
    """O bloco renderizado entra no contexto do execution_node. Precisa:
    - Incluir o symbol e o label
    - Renderizar None como '(não coletado)' (sem alucinar)
    - Cobrir todas as seções (market, multi-TF, risco, sinal, forecast)"""

    def test_renders_label_and_symbol(self):
        asset = AssetState(symbol="BTCUSDT", live_price=42000.0)
        block = _render_asset_block(asset, label="BTCUSDT (PRIMÁRIO)")
        assert "BTCUSDT (PRIMÁRIO)" in block
        assert "Symbol:          BTCUSDT" in block

    def test_renders_none_safely(self):
        # Sem dados, não deve quebrar nem inventar números.
        block = _render_asset_block(AssetState(symbol="UNKNOWN"))
        assert "(não coletado)" in block
        # Nenhum número fake: o forecast_status default é "not_requested".
        assert "not_requested" in block

    def test_renders_all_sections(self):
        asset = AssetState(
            symbol="BTCUSDT",
            live_price=42000.0,
            rsi_14=65.0,
            macro_rsi_14=70.0,
            sharpe=1.5,
            risk_level=RiskLevel.LOW,
            forecast_direction="up",
            forecast_return_pct=2.5,
        )
        block = _render_asset_block(asset)
        # Cada seção esperada pelo prompt do execution_node:
        assert "DADOS DE MERCADO" in block
        assert "INDICADORES — MACRO (1D)" in block
        assert "INDICADORES — SETUP (4H)" in block
        assert "INDICADORES — EXEC (1H)" in block
        assert "MÉTRICAS DE RISCO" in block
        assert "SINAL & DECISÃO" in block
        assert "FORECAST APOLLO" in block
        assert "QUALIDADE DOS DADOS" in block
        # Valores presentes:
        assert "42000" in block
        assert "1.500" in block  # sharpe rendered with prec=3


class TestSymbolHeuristicCoverage:
    """A heurística determinística cobre canônicos e aliases conhecidos.
    Typos arbitrários (ex.: 'etherium') são tratados pelo LLM normalizador
    no orchestrator — vide TestComparativeIntent e o fluxo do orchestrator_node."""

    def test_canonical_names_recognized(self):
        # Os aliases canônicos devem continuar funcionando perfeitamente.
        assert _extract_all_symbols("compare ethereum vs bitcoin") == ["ETHUSDT", "BTCUSDT"]

    def test_typo_falls_through_to_llm(self):
        # Typo "etherium" NÃO está no alias map por design — não queremos
        # listar typos manualmente. A heurística devolve só BTC; é o
        # orchestrator que aciona o LLM normalizador para resgatar ETH.
        assert _extract_all_symbols(
            "compare bitcoin com etherium hoje?"
        ) == ["BTCUSDT"]


class TestComparativeIntent:
    """Detecção heurística de pergunta comparativa. É o gatilho que decide
    se vale a pena chamar o LLM normalizador para resgatar typos no segundo
    ativo (ex.: 'bitcoin vs etherium' → heurística pega só BTC, LLM resgata ETH)."""

    def test_versus_token(self):
        from metis.agent.graph import _has_comparative_intent
        assert _has_comparative_intent("bitcoin vs ethereum")
        assert _has_comparative_intent("BTC versus ETH")

    def test_compare_verb(self):
        from metis.agent.graph import _has_comparative_intent
        assert _has_comparative_intent("compare bitcoin com etherium")
        assert _has_comparative_intent("comparar BTC e SOL")

    def test_better_than_phrase(self):
        from metis.agent.graph import _has_comparative_intent
        assert _has_comparative_intent("XRP é melhor que Chainlink?")
        assert _has_comparative_intent("ETH is better than BTC")

    def test_or_token(self):
        from metis.agent.graph import _has_comparative_intent
        assert _has_comparative_intent("compro BTC ou ETH?")

    def test_non_comparative_returns_false(self):
        from metis.agent.graph import _has_comparative_intent
        assert not _has_comparative_intent("como está o bitcoin?")
        assert not _has_comparative_intent("qual o preço do ETH?")
        assert not _has_comparative_intent("explica o que é DeFi")

    def test_empty_text_returns_false(self):
        from metis.agent.graph import _has_comparative_intent
        assert not _has_comparative_intent("")
        assert not _has_comparative_intent(None)  # type: ignore[arg-type]


class TestLatestUserMessage:
    """Regressão crítica: `_run_agent_loop` antes pegava a PRIMEIRA HumanMessage,
    que era o summary de contexto histórico injetado pelo `chat.py`. Resultado:
    todos os agentes downstream recebiam a "pergunta" errada e ignoravam o
    pedido atual do usuário."""

    def test_returns_none_when_no_messages(self):
        assert _latest_user_message([]) is None

    def test_returns_none_when_only_ai_messages(self):
        assert _latest_user_message([AIMessage(content="oi"), AIMessage(content="tchau")]) is None

    def test_returns_only_human_message(self):
        msg = HumanMessage(content="oi")
        assert _latest_user_message([msg]) is msg

    def test_picks_last_human_when_multiple(self):
        first = HumanMessage(content="pergunta antiga")
        second = HumanMessage(content="pergunta NOVA")
        result = _latest_user_message([first, AIMessage(content="resposta"), second])
        assert result is second
        assert result.content == "pergunta NOVA"

    def test_ignores_ai_messages_after_last_human(self):
        # Esse é o cenário real: o pipeline empilha AIMessages depois da
        # pergunta atual. messages[-1] seria uma AIMessage do orchestrator/etc.
        current = HumanMessage(content="pergunta atual")
        messages = [
            HumanMessage(content="Previous conversations context: ..."),  # summary
            HumanMessage(content="oi"),                                    # turno antigo
            AIMessage(content="Olá, como posso ajudar?"),
            current,                                                       # pergunta atual
            AIMessage(content="Routed to BTCUSDT"),                        # orchestrator
            AIMessage(content="MarketData output"),                        # market_data
        ]
        assert _latest_user_message(messages) is current

    def test_chat_py_scenario_with_history_and_current_question(self):
        """Simula EXATAMENTE o que `chat.py` constrói antes de invocar o graph.

        Antes do fix, o pipeline pegava a primeira HumanMessage (summary de
        contexto), e por isso não conseguia ver a pergunta real do usuário.
        """
        messages = [
            # 1) summary de global_context
            HumanMessage(content="Previous conversations context:\nuser: olá...\nassistant: oi...\n"),
            # 2) histórico de sessão (turno N-1)
            HumanMessage(content="oi"),
            AIMessage(content="Olá! Como posso ajudar?"),
            # 3) pergunta corrente
            HumanMessage(content="pode comparar o desempenho do bitcoin com etherium hoje?"),
        ]
        latest = _latest_user_message(messages)
        assert latest is not None
        assert "etherium" in latest.content
        # E NÃO o summary nem a pergunta antiga.
        assert "Previous conversations" not in latest.content
        assert latest.content != "oi"

    def test_handles_system_messages_interleaved(self):
        # SystemMessage não é HumanMessage; deve ser ignorada.
        sys = SystemMessage(content="você é um agente")
        h = HumanMessage(content="qual o preço do BTC?")
        assert _latest_user_message([sys, h]) is h
        assert _latest_user_message([h, sys]) is h
