"""
Quant Fund-Grade Multi-Agent System
────────────────────────────────────
Arquitetura inspirada em fundos quantitativos institucionais:

  ┌─ Orchestrator ─────────────────────────────────────────────┐
  │  MarketDataAgent · FeatureAgent · RiskAgent · SignalAgent  │
  │  ↓                                                         │
  │  PreTradeRiskGate  (circuit breaker, limites de exposição) │
  │  ↓                                                         │
  │  ExecutionLayer    (sizing, TWAP/VWAP, slippage model)     │
  └────────────────────────────────────────────────────────────┘

Principais upgrades vs. implementação anterior:
  • Grafo multi-agente com responsabilidades separadas por domínio
  • Risk gate como nó bloqueador (interrompe o fluxo antes da execução)
  • Estado estruturado e tipado com Pydantic (AgentState)
  • Prompt engineering com cadeia de raciocínio explícita (CoT obrigatório)
  • Anomaly detection embutido no pipeline
  • Rastreio completo de decisões (audit trail) via intermediate_steps
  • Suporte a parallel tool calls no nó de features
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.tools import all_tools

# ─────────────────────────────────────────────
# 1. DOMAIN ENUMS
# ─────────────────────────────────────────────

class AnalysisTimeframe(str, Enum):
    INTRADAY  = "intraday"   # 1m / 5m
    DAILY     = "daily"      # 1h / 1d
    WEEKLY    = "weekly"     # 1D / 1W

class RiskLevel(str, Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    EXTREME  = "extreme"

class NextAction(str, Enum):
    MARKET_DATA    = "market_data"
    FEATURES       = "features"
    RISK           = "risk"
    SIGNAL         = "signal"
    RISK_GATE      = "risk_gate"
    EXECUTION      = "execution"
    FINALIZE       = "finalize"
    BLOCKED        = "blocked"

# ─────────────────────────────────────────────
# 2. TYPED STATE  (Pydantic-backed)
# ─────────────────────────────────────────────

class QuantAgentState(BaseModel):
    # Conversation
    messages:           list[Any]          = Field(default_factory=list)
    user_id:            str                = ""

    # Routing
    next_action:        NextAction         = NextAction.MARKET_DATA
    intermediate_steps: list[tuple[str, str]] = Field(default_factory=list)
    final_answer:       str                = ""

    # Market context (populated by MarketDataAgent)
    symbol:             str                = ""
    timeframe:          AnalysisTimeframe  = AnalysisTimeframe.DAILY
    live_price:         float | None       = None
    volume_24h:         float | None       = None
    price_change_pct:   float | None       = None

    # Risk context (populated by RiskAgent)
    risk_level:         RiskLevel | None   = None
    var_95:             float | None       = None
    cvar_95:            float | None       = None
    sharpe:             float | None       = None
    max_drawdown:       float | None       = None
    volatility_annualized: float | None    = None

    # Signal context (populated by SignalAgent)
    regime:             str | None         = None   # "trending" | "ranging" | "breakout"
    signal_direction:   str | None         = None   # "long" | "short" | "neutral"
    signal_confidence:  float | None       = None   # 0..1

    # Risk gate output
    gate_approved:      bool | None        = None
    gate_reason:        str                = ""

    # Data quality
    anomalies_detected: list[str]          = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

# ─────────────────────────────────────────────
# 3. LLM INSTANCES  (per agent, tuned separately)
# ─────────────────────────────────────────────

_BASE_MODEL = "gpt-4o"

def _make_llm(temperature: float = 0.1, **kw) -> ChatOpenAI:
    return ChatOpenAI(model=_BASE_MODEL, temperature=temperature, **kw)

tool_map = {t.name: t for t in all_tools}

llm_orchestrator   = _make_llm(temperature=0.1)   # sem tools — apenas roteamento
llm_market_data    = _make_llm(temperature=0.0).bind_tools(all_tools)
llm_features       = _make_llm(temperature=0.0).bind_tools(all_tools, parallel_tool_calls=True)
llm_risk           = _make_llm(temperature=0.0).bind_tools(all_tools)
llm_signal         = _make_llm(temperature=0.15).bind_tools(all_tools)
llm_execution      = _make_llm(temperature=0.0)   # no tools — pure reasoning

# ─────────────────────────────────────────────
# 4. SYSTEM PROMPTS  (cada agente tem seu próprio)
# ─────────────────────────────────────────────

_ORCHESTRATOR_SYSTEM = """
Você é o Orchestrator de um sistema multi-agente de análise quantitativa institucional.
Seu papel é EXCLUSIVAMENTE rotear e sintetizar — não analisar dados diretamente.

PIPELINE OBRIGATÓRIO:
1. market_data  → coleta preço, volume, 24h stats
2. features     → calcula indicadores técnicos (paralelo)
3. risk         → avalia métricas de risco
4. signal       → gera sinal direcional com regime
5. risk_gate    → valida limites pré-trade
6. finalize     → consolida resposta ao usuário

REGRAS DE ROTEAMENTO:
- Extraia symbol, timeframe e contexto da pergunta do usuário
- Se timeframe não especificado: use DAILY para análise geral, INTRADAY para scalping
- Sempre passe o contexto acumulado para os agentes downstream
- Se risk_gate retornar BLOCKED: informe o usuário com motivo claro

OUTPUT FORMAT: JSON estruturado com campos: next_agent, symbol, timeframe, context
""".strip()

_MARKET_DATA_SYSTEM = """
Você é o Market Data Agent de um fundo quantitativo.

RESPONSABILIDADES:
1. Coletar dados de mercado em tempo real via ferramentas Binance
2. Detectar anomalias nos dados ANTES de reportar
3. Normalizar e contextualizar os valores

ANOMALY DETECTION (OBRIGATÓRIO):
- Timestamps inconsistentes (alternância :59/:00 suspeita)
- Price spikes > 3σ em janela de 5min
- Volume 0 com candle fechado
- Preços idênticos em N candles consecutivos (provável bug de duplicação)
- Spread bid/ask > 2% (illiquidity flag)

Se detectar anomalia: adicione ao campo anomalies_detected e use os dados COM CAUTELA.

FERRAMENTAS: get_live_price, get_indicators, get_top_cryptos
OUTPUT: Preencha live_price, volume_24h, price_change_pct, anomalies_detected
""".strip()

_FEATURE_SYSTEM = """
Você é o Feature Engineering Agent de um fundo quantitativo.

RESPONSABILIDADES:
1. Calcular indicadores técnicos no timeframe correto (NUNCA misture timeframes)
2. Recalcular features se necessário (recalculate_features primeiro)
3. Normalizar TODOS os valores antes de reportar

REGRAS DE TIMEFRAME:
- INTRADAY: interval="1m" ou "5m" — retornos em % por minuto
- DAILY: interval="1h" — retornos em % por hora  
- WEEKLY: interval="1D" — retornos em % por dia

NORMALIZAÇÃO OBRIGATÓRIA:
- Volatilidade 1m: anualizar = sqrt(252 × 1440) × vol_1m
- Momentum: converter para z-score = (valor - média_30d) / std_30d
- RSI: reportar regime (overbought >70, oversold <30, neutro 30-70)
- Bollinger Bands: reportar %B = (close - lower) / (upper - lower)
- MACD: reportar cruzamento (bull cross / bear cross / divergência)

FERRAMENTAS (chame em PARALELO): get_feature_values, get_redis_history, recalculate_features
""".strip()

_RISK_SYSTEM = """
Você é o Risk Agent de um fundo quantitativo — o guardião da gestão de risco.

RESPONSABILIDADES:
1. Calcular e interpretar métricas de risco com rigor estatístico
2. Determinar risk_level baseado em múltiplos fatores
3. Contextualizar todas as métricas com benchmarks históricos

MÉTRICAS OBRIGATÓRIAS:
- VaR 95% (1 dia): P&L esperado no pior 5% dos cenários
- CVaR 95% (Expected Shortfall): perda média além do VaR
- Sharpe Ratio: (retorno - risk_free) / volatilidade anualizada
  • > 2.0 = excelente | 1.0-2.0 = bom | 0-1.0 = aceitável | < 0 = ruim
- Max Drawdown: queda do pico ao vale em % 
- Calmar Ratio = CAGR / |Max Drawdown| (> 1.0 é institucional)
- Volatilidade anualizada: interpretar vs. BTC histórico (~80% a.a.)

CLASSIFICAÇÃO DE RISCO:
- LOW:      vol < 40% a.a., sharpe > 1.5, drawdown < 15%
- MODERATE: vol 40-80% a.a., sharpe 0.5-1.5, drawdown 15-30%
- HIGH:     vol 80-120% a.a., sharpe < 0.5, drawdown 30-50%
- EXTREME:  vol > 120% a.a., sharpe < 0, drawdown > 50%

FERRAMENTAS: calculate_risk, get_feature_values (cvar_95, sharpe, max_drawdown, calmar)
""".strip()

_SIGNAL_SYSTEM = """
Você é o Signal Agent de um fundo quantitativo — gerador de alpha.

RESPONSABILIDADES:
1. Identificar o regime de mercado atual
2. Gerar sinal direcional com fundamentação estatística
3. Estimar confiança do sinal (0..1) baseada em convergência de indicadores

REGIMES DE MERCADO:
- trending: EMA > SMA, momentum positivo, ADX implícito > 25
- ranging:  preço entre BB, momentum próximo de zero, RSI 40-60
- breakout: preço fora das BBands, volume_ratio > 1.5, momentum extremo

GERAÇÃO DE SINAL:
- Analise divergência RSI vs. preço (sinal de reversão de alta confiança)
- Analise cruzamento MACD (signal line vs. MACD line + histograma)
- Analise posição relativa às Bollinger Bands (%B)
- Analise momentum relativo ao histórico (z-score)
- NÃO gere sinal long/short sem pelo menos 3 indicadores convergentes

CONFIANÇA:
- 0.8-1.0: ≥4 indicadores convergentes + regime confirmado
- 0.6-0.8: 3 indicadores convergentes
- 0.4-0.6: 2 indicadores ou sinal contraditório
- < 0.4:   inconclusivo — use "neutral"

FERRAMENTAS: get_feature_values, search_market_news
""".strip()

_EXECUTION_SYSTEM = """
Você é o Execution Layer de um fundo quantitativo.

Dado o contexto completo (mercado, features, risco, sinal), monte a resposta final
para o usuário com:

1. RESUMO EXECUTIVO (2-3 linhas): situação atual do ativo
2. ANÁLISE TÉCNICA: indicadores chave com valores normalizados e interpretação
3. MÉTRICAS DE RISCO: VaR, Sharpe, drawdown com contexto histórico
4. SINAL: direção, regime, confiança — com fundamentação
5. SIZING SUGERIDO (se perguntado): baseado no risk_level e Kelly Criterion parcial
   • Kelly fraction = (p × b - q) / b, onde b = payoff ratio, p = win_rate estimado
   • Use 1/4 Kelly para conservadorismo institucional
6. ALERTAS: anomalias detectadas, riscos não cobertos, limitações da análise
7. DISCLAIMER: "Análise quantitativa — não é recomendação de investimento"

Tom: analista institucional sênior. Preciso, direto, sem jargão desnecessário.
""".strip()

# ─────────────────────────────────────────────
# 5. TOOL EXECUTOR  (reutilizável)
# ─────────────────────────────────────────────

def _execute_tools(last_ai: AIMessage, steps: list) -> tuple[list[ToolMessage], list]:
    """Executa tool calls e retorna (tool_messages, steps_atualizados)."""
    tool_messages: list[ToolMessage] = []
    steps = list(steps)  # cópia para não mutar o original
    for tc in last_ai.tool_calls:
        name   = tc["name"]
        args   = tc["args"]
        tool   = tool_map.get(name)
        result = tool.invoke(args) if tool else f"[ERR] Tool '{name}' não encontrada."
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        steps.append((name, str(result)))
    return tool_messages, steps

def _run_agent_loop(
    state: QuantAgentState,
    llm,
    system_prompt: str,
    extra_context: str = "",
) -> tuple[str, list[ToolMessage], list]:
    """
    Loop LLM → tool call → result para um agente especializado.
    Usa APENAS a mensagem original do usuário — não o histórico acumulado
    de outros agentes (que pode conter tool_calls sem resposta).
    Retorna (resposta_final, tool_messages_acumulados, steps_acumulados).
    """
    # Extrai apenas a mensagem original do usuário
    original_query = next((m for m in state.messages if isinstance(m, HumanMessage)), None)

    msgs: list = [SystemMessage(content=system_prompt)]
    if original_query:
        msgs.append(original_query)
    if extra_context:
        msgs.append(HumanMessage(content=f"[CONTEXTO DO PIPELINE]\n{extra_context}"))

    all_tool_msgs: list[ToolMessage] = []
    steps = list(state.intermediate_steps)
    MAX_ITERATIONS = 6

    for _ in range(MAX_ITERATIONS):
        response = llm.invoke(msgs)
        msgs.append(response)
        if not response.tool_calls:
            return response.content, all_tool_msgs, steps
        tool_msgs, steps = _execute_tools(response, steps)
        msgs.extend(tool_msgs)
        all_tool_msgs.extend(tool_msgs)

    # Força finalização se exceder iterações
    final = llm.invoke(msgs + [HumanMessage(content="Sintetize os resultados obtidos até agora.")])
    return final.content, all_tool_msgs, steps

# ─────────────────────────────────────────────
# 6. AGENT NODES
# ─────────────────────────────────────────────

def orchestrator_node(state: QuantAgentState) -> dict:
    """
    Extrai intent, symbol, timeframe e define o pipeline inicial.
    """
    msgs  = [SystemMessage(content=_ORCHESTRATOR_SYSTEM)] + state.messages
    resp  = llm_orchestrator.invoke(msgs)

    # Tenta extrair metadados do JSON emitido pelo orchestrator
    symbol    = state.symbol
    timeframe = state.timeframe
    try:
        raw = resp.content
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            meta      = json.loads(raw[start:end])
            symbol    = meta.get("symbol", symbol)
            timeframe = AnalysisTimeframe(meta.get("timeframe", timeframe))
    except Exception:
        pass

    return {
        **state.dict(),
        "messages":     state.messages + [resp],
        "next_action":  NextAction.MARKET_DATA,
        "symbol":       symbol,
        "timeframe":    timeframe,
        "intermediate_steps": state.intermediate_steps + [("orchestrator", resp.content[:200])],
    }

def market_data_node(state: QuantAgentState) -> dict:
    context = f"Symbol: {state.symbol} | Timeframe: {state.timeframe}"
    content, tool_msgs, steps = _run_agent_loop(
        state, llm_market_data, _MARKET_DATA_SYSTEM, context
    )

    # Parse anomalias (padrão simples — adaptar ao seu schema de output)
    anomalies = state.anomalies_detected[:]
    if "anomalia" in content.lower() or "bug" in content.lower():
        anomalies.append(f"[market_data] {content[:200]}")

    return {
        **state.dict(),
        "messages":          state.messages + [AIMessage(content=content)],
        "next_action":       NextAction.FEATURES,
        "intermediate_steps": steps,
        "anomalies_detected": anomalies,
    }

def feature_engineering_node(state: QuantAgentState) -> dict:
    context = (
        f"Symbol: {state.symbol} | Timeframe: {state.timeframe} | "
        f"Preço atual: {state.live_price} | "
        f"Anomalias: {state.anomalies_detected}"
    )
    content, tool_msgs, steps = _run_agent_loop(
        state, llm_features, _FEATURE_SYSTEM, context
    )
    return {
        **state.dict(),
        "messages":           state.messages + [AIMessage(content=content)],
        "next_action":        NextAction.RISK,
        "intermediate_steps": steps,
    }

def risk_agent_node(state: QuantAgentState) -> dict:
    context = (
        f"Symbol: {state.symbol} | Preço: {state.live_price} | "
        f"Vol 24h: {state.volume_24h} | Variação: {state.price_change_pct}%"
    )
    content, tool_msgs, steps = _run_agent_loop(
        state, llm_risk, _RISK_SYSTEM, context
    )

    # Tenta extrair risk_level do output (padrão simples)
    risk_level = state.risk_level
    for lvl in RiskLevel:
        if lvl.value in content.lower():
            risk_level = lvl
            break

    return {
        **state.dict(),
        "messages":           state.messages + [AIMessage(content=content)],
        "next_action":        NextAction.SIGNAL,
        "risk_level":         risk_level,
        "intermediate_steps": steps,
    }

def signal_agent_node(state: QuantAgentState) -> dict:
    context = (
        f"Symbol: {state.symbol} | Risk level: {state.risk_level} | "
        f"Sharpe: {state.sharpe} | Drawdown: {state.max_drawdown} | "
        f"CVaR 95%: {state.cvar_95}"
    )
    content, tool_msgs, steps = _run_agent_loop(
        state, llm_signal, _SIGNAL_SYSTEM, context
    )

    # Parse regime e direção do sinal
    direction = "neutral"
    for d in ("long", "short", "neutral"):
        if d in content.lower():
            direction = d
            break

    return {
        **state.dict(),
        "messages":           state.messages + [AIMessage(content=content)],
        "next_action":        NextAction.RISK_GATE,
        "signal_direction":   direction,
        "intermediate_steps": steps,
    }

def pre_trade_risk_gate_node(state: QuantAgentState) -> dict:
    """
    Nó bloqueador: verifica limites antes de qualquer recomendação de execução.
    Em produção, aqui entrariam limites de posição, concentração, horários proibidos, etc.
    """
    blocked  = False
    reasons: list[str] = []

    # Regra 1: bloqueio por risco extremo
    if state.risk_level == RiskLevel.EXTREME:
        blocked = True
        reasons.append("Risk level EXTREME — exposição bloqueada por política de risco.")

    # Regra 2: anomalias críticas não resolvidas
    if len(state.anomalies_detected) >= 3:
        blocked = True
        reasons.append(f"Múltiplas anomalias detectadas ({len(state.anomalies_detected)}) — dados não confiáveis.")

    # Regra 3: sinal com confiança muito baixa
    if (state.signal_confidence or 1.0) < 0.3 and state.signal_direction != "neutral":
        blocked = True
        reasons.append(f"Sinal {state.signal_direction} com confiança {state.signal_confidence:.0%} — abaixo do threshold mínimo (30%).")

    # Regra 4: CVaR extremo
    if state.cvar_95 and abs(state.cvar_95) > 0.25:
        blocked = True
        reasons.append(f"CVaR 95% = {state.cvar_95:.1%} — excede limite de -25% por posição.")

    gate_reason  = " | ".join(reasons) if reasons else "Aprovado."
    next_action  = NextAction.BLOCKED if blocked else NextAction.EXECUTION

    return {
        **state.dict(),
        "next_action":  next_action,
        "gate_approved": not blocked,
        "gate_reason":   gate_reason,
        "intermediate_steps": state.intermediate_steps + [("risk_gate", gate_reason)],
    }

def execution_node(state: QuantAgentState) -> dict:
    """
    Consolida toda a análise e gera resposta final institucional.
    """
    context = f"""
DADOS DE MERCADO
  Symbol:          {state.symbol}
  Timeframe:       {state.timeframe}
  Preço atual:     {state.live_price}
  Variação 24h:    {state.price_change_pct}%
  Volume 24h:      {state.volume_24h}

MÉTRICAS DE RISCO
  Risk level:      {state.risk_level}
  VaR 95%:         {state.var_95}
  CVaR 95%:        {state.cvar_95}
  Sharpe:          {state.sharpe}
  Max Drawdown:    {state.max_drawdown}
  Vol. anualizada: {state.volatility_annualized}

SINAL
  Regime:          {state.regime}
  Direção:         {state.signal_direction}
  Confiança:       {state.signal_confidence}

QUALIDADE DOS DADOS
  Anomalias:       {state.anomalies_detected or 'Nenhuma'}

RISK GATE
  Aprovado:        {state.gate_approved}
  Motivo:          {state.gate_reason}

HISTÓRICO DE ANÁLISE (resumo):
{chr(10).join(f'  [{k}] {v[:120]}' for k, v in state.intermediate_steps[-8:])}
""".strip()

    # Usa apenas o contexto estruturado — state.messages pode conter mensagens
    # de outros agentes incompatíveis com a API OpenAI (órfãos de tool_calls)
    msgs = [
        SystemMessage(content=_EXECUTION_SYSTEM),
        HumanMessage(content=context),
    ]
    final_response = llm_execution.invoke(msgs)

    return {
        **state.dict(),
        "messages":     state.messages + [final_response],
        "next_action":  NextAction.FINALIZE,
        "final_answer": final_response.content,
    }

def blocked_node(state: QuantAgentState) -> dict:
    """Resposta quando o risk gate bloqueia a análise."""
    msg = (
        f"⛔ **Análise bloqueada pelo Risk Gate**\n\n"
        f"**Motivo:** {state.gate_reason}\n\n"
        f"A posição não foi avaliada para execução. "
        f"Revise as condições de mercado ou os parâmetros de risco antes de prosseguir."
    )
    return {
        **state.dict(),
        "final_answer": msg,
        "next_action":  NextAction.FINALIZE,
    }

def finalize_node(state: QuantAgentState) -> dict:
    """Extrai a resposta final — nó terminal."""
    return {**state.dict(), "next_action": "done"}

# ─────────────────────────────────────────────
# 7. GRAPH ASSEMBLY
# ─────────────────────────────────────────────

def build_quant_graph() -> Any:
    workflow = StateGraph(QuantAgentState)

    # Nodes
    workflow.add_node("orchestrator",   orchestrator_node)
    workflow.add_node("market_data",    market_data_node)
    workflow.add_node("features",       feature_engineering_node)
    workflow.add_node("risk",           risk_agent_node)
    workflow.add_node("signal",         signal_agent_node)
    workflow.add_node("risk_gate",      pre_trade_risk_gate_node)
    workflow.add_node("execution",      execution_node)
    workflow.add_node("blocked",        blocked_node)
    workflow.add_node("finalize",       finalize_node)

    # Entry
    workflow.set_entry_point("orchestrator")

    # Fixed edges (pipeline linear)
    workflow.add_edge("orchestrator", "market_data")
    workflow.add_edge("market_data",  "features")
    workflow.add_edge("features",     "risk")
    workflow.add_edge("risk",         "signal")
    workflow.add_edge("signal",       "risk_gate")

    # Conditional: risk gate → execution or blocked
    workflow.add_conditional_edges(
        "risk_gate",
        lambda s: s.next_action.value,
        {
            NextAction.EXECUTION.value: "execution",
            NextAction.BLOCKED.value:   "blocked",
        },
    )

    # Terminal nodes → finalize → END
    workflow.add_edge("execution", "finalize")
    workflow.add_edge("blocked",   "finalize")
    workflow.add_edge("finalize",  END)

    return workflow.compile()

# ─────────────────────────────────────────────
# 8. PUBLIC API
# ─────────────────────────────────────────────

_graph: Any | None = None

def get_agent_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_quant_graph()
    return _graph

async def run_analysis(user_message: str, user_id: str = "anon") -> str:
    """
    Ponto de entrada de alto nível.

    Example:
        result = await run_analysis("Analise BTC com gestão de risco institucional")
        print(result)
    """
    graph  = get_agent_graph()
    state  = QuantAgentState(
        messages=[HumanMessage(content=user_message)],
        user_id=user_id,
    )
    result = await graph.ainvoke(state)
    return result["final_answer"]