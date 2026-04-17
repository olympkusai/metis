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
import math
import time
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ConfigDict

from app.tools import all_tools
from app.agent.schemas import (
    OrchestratorOutput,
    MarketDataOutput,
    FeatureEngineeringOutput,
    RiskAgentOutput,
    SignalAgentOutput,
    RiskGateOutput,
    LivePriceOutput,
    IndicatorsOutput,
    RSIOutput,
    MACDOutput,
    BollingerBandsOutput,
    VolatilityOutput,
    RiskMetricsOutput,
    SharpeOutput,
    CVaROutput,
    MaxDrawdownOutput,
)
from app.agent.quant_engine import (
    compute_signal_score,
    determine_risk_level,
    calculate_position_size,
)
from app.agent.portfolio import (
    PortfolioState,
    PortfolioConstraints,
    check_portfolio_constraints,
)
from app.agent.execution import (
    ExecutionStrategy,
    calculate_vwap_execution,
    calculate_twap_execution,
    estimate_slippage,
    recommend_execution_strategy,
)

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
    intermediate_steps_global: list[tuple[str, str]] = Field(default_factory=list)  # Global audit trail
    intermediate_steps_agent: list[tuple[str, str]] = Field(default_factory=list)  # Agent-specific steps
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

    # Technical indicators (populated by FeatureAgent)
    rsi_14:             float | None       = None
    macd_line:          float | None       = None
    macd_signal:        float | None       = None
    macd_histogram:     float | None       = None
    bb_upper:           float | None       = None
    bb_middle:          float | None       = None
    bb_lower:           float | None       = None

    # Signal context (populated by SignalAgent)
    regime:             str | None         = None   # "trending" | "ranging" | "breakout"
    signal_direction:   str | None         = None   # "long" | "short" | "neutral"
    signal_confidence:  float | None       = None   # 0..1

    # Risk gate output
    gate_approved:      bool | None        = None
    gate_reason:        str                = ""

    # Data quality
    anomalies_detected: list[str]          = Field(default_factory=list)

    # Portfolio context (for institutional portfolio management)
    portfolio_state: PortfolioState       = Field(default_factory=PortfolioState)
    proposed_position_size: float | None   = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

# ─────────────────────────────────────────────
# 3. LLM INSTANCES  (per agent, tuned separately)
# ─────────────────────────────────────────────

_BASE_MODEL = "gpt-4o"

def _make_llm(temperature: float = 0.1, **kw) -> ChatOpenAI:
    return ChatOpenAI(model=_BASE_MODEL, temperature=temperature, **kw)

tool_map = {t.name: t for t in all_tools}

llm_orchestrator   = _make_llm(temperature=0.1).with_structured_output(OrchestratorOutput)
llm_market_data    = _make_llm(temperature=0.0).bind_tools(all_tools)
llm_features       = _make_llm(temperature=0.0).bind_tools(all_tools, parallel_tool_calls=True)
llm_risk           = _make_llm(temperature=0.0).bind_tools(all_tools)
llm_signal         = _make_llm(temperature=0.15).bind_tools(all_tools)
llm_execution      = _make_llm(temperature=0.0)  # no tools — pure reasoning

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

INSTRUÇÃO IMPORTANTE: Você DEVE chamar as ferramentas get_live_price e get_indicators para obter os dados. Não responda sem chamar as ferramentas primeiro.

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

INSTRUÇÃO IMPORTANTE: Você DEVE chamar as ferramentas get_feature_rsi, get_feature_macd, get_feature_bollinger, get_feature_volatility para calcular os indicadores. Chame-as em PARALELO.

FERRAMENTAS (chame em PARALELO): get_feature_rsi, get_feature_macd, get_feature_bollinger, get_feature_volatility, get_redis_history
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
- Volatilidade anualizada: interpretar vs. histórico do ativo (~80% a.a.)

CLASSIFICAÇÃO DE RISCO:
- LOW:      vol < 40% a.a., sharpe > 1.5, drawdown < 15%
- MODERATE: vol 40-80% a.a., sharpe 0.5-1.5, drawdown 15-30%
- HIGH:     vol 80-120% a.a., sharpe < 0.5, drawdown 30-50%
- EXTREME:  vol > 120% a.a., sharpe < 0, drawdown > 50%

INSTRUÇÃO IMPORTANTE: Você DEVE chamar as ferramentas calculate_risk, get_feature_sharpe, get_feature_cvar, get_feature_max_drawdown, get_feature_volatility para calcular as métricas de risco.

FERRAMENTAS: calculate_risk, get_feature_sharpe, get_feature_cvar, get_feature_max_drawdown, get_feature_volatility
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

INSTRUÇÃO IMPORTANTE: Você DEVE chamar a ferramenta search_market_news para buscar contexto adicional sobre o mercado.

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
    clear_steps: bool = False,
) -> tuple[str, list[ToolMessage], list]:
    """
    Loop LLM → tool call → result para um agente especializado.
    Usa APENAS a mensagem original do usuário — não o histórico acumulado
    de outros agentes (que pode conter tool_calls sem resposta).
    Retorna (resposta_final, tool_messages_acumulados, steps_acumulados).
    Includes retry mechanism for LLM calls.
    """
    # Extrai apenas a mensagem original do usuário
    original_query = next((m for m in state.messages if isinstance(m, HumanMessage)), None)

    # Adiciona instrução explícita para chamar ferramentas
    tool_instruction = "\n\nIMPORTANTE: Você tem acesso a ferramentas. Você DEVE chamar as ferramentas apropriadas antes de fornecer sua análise final. Não responda sem chamar as ferramentas primeiro."
    enhanced_prompt = system_prompt + tool_instruction

    msgs: list = [SystemMessage(content=enhanced_prompt)]
    if original_query:
        msgs.append(original_query)
    if extra_context:
        msgs.append(HumanMessage(content=f"[CONTEXTO DO PIPELINE]\n{extra_context}"))

    all_tool_msgs: list[ToolMessage] = []
    steps = [] if clear_steps else list(state.intermediate_steps_agent)
    MAX_ITERATIONS = 6
    MAX_RETRIES = 3
    
    # Track previous state to detect no-progress loops
    previous_step_count = len(steps)
    no_progress_count = 0

    for iteration in range(MAX_ITERATIONS):
        # Retry mechanism for LLM calls with exponential backoff
        response = None
        for attempt in range(MAX_RETRIES):
            try:
                response = llm.invoke(msgs, timeout=30)  # 30s timeout
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    # Fallback: return error message
                    error_msg = f"[LLM ERROR after {MAX_RETRIES} retries]: {str(e)}"
                    return error_msg, all_tool_msgs, steps
                # Exponential backoff: 0.5s, 1s, 2s
                backoff_time = 0.5 * (2 ** attempt)
                time.sleep(backoff_time)
                continue
        
        if response is None:
            error_msg = "[LLM ERROR] Failed to get response after retries"
            return error_msg, all_tool_msgs, steps
            
        msgs.append(response)
        
        # Debug: log response type and tool_calls
        print(f"[DEBUG] Response type: {type(response).__name__}")
        if hasattr(response, 'tool_calls'):
            print(f"[DEBUG] Tool calls: {response.tool_calls}")
        print(f"[DEBUG] Response content preview: {str(response)[:200]}")
        
        # Handle both AIMessage (with tool_calls) and structured Pydantic output
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_msgs, steps = _execute_tools(response, steps)
            msgs.extend(tool_msgs)
            all_tool_msgs.extend(tool_msgs)
        elif isinstance(response, (str, dict)) or hasattr(response, 'model_dump'):
            # Structured output (Pydantic model) - return as JSON string
            if hasattr(response, 'model_dump'):
                content = json.dumps(response.model_dump(), ensure_ascii=False, indent=2)
            elif isinstance(response, dict):
                content = json.dumps(response, ensure_ascii=False, indent=2)
            else:
                content = str(response)
            return content, all_tool_msgs, steps
        else:
            # Regular AIMessage without tool calls
            print(f"[DEBUG] No tool calls found, returning content directly")
            return response.content, all_tool_msgs, steps
        
        # Detect no-progress loops
        if len(steps) == previous_step_count:
            no_progress_count += 1
            if no_progress_count >= 2:
                # No progress for 2 iterations - abort early
                error_msg = "[AGENT LOOP] No progress detected - aborting to prevent infinite loop"
                return error_msg, all_tool_msgs, steps
        else:
            no_progress_count = 0
            previous_step_count = len(steps)

    # Força finalização se exceder iterações
    final = llm.invoke(msgs + [HumanMessage(content="Sintetize os resultados obtidos até agora.")])
    return final.content, all_tool_msgs, steps

# ─────────────────────────────────────────────
# 6. AGENT NODES
# ─────────────────────────────────────────────

def orchestrator_node(state: QuantAgentState) -> dict:
    """
    Extrai intent, symbol, timeframe e define o pipeline inicial.
    Uses structured output from LLM.
    """
    print(f"[DEBUG] ===== ORCHESTRATOR NODE START =====")
    print(f"[DEBUG] Input state symbol: {state.symbol}")
    print(f"[DEBUG] Input state messages: {len(state.messages)} messages")
    msgs  = [SystemMessage(content=_ORCHESTRATOR_SYSTEM)] + state.messages
    try:
        resp = llm_orchestrator.invoke(msgs)
        print(f"[DEBUG] Orchestrator response type: {type(resp).__name__}")
        print(f"[DEBUG] Orchestrator response: {resp}")
        # resp is now a structured OrchestratorOutput object
        symbol = resp.symbol or state.symbol
        timeframe = AnalysisTimeframe(resp.timeframe) if resp.timeframe else state.timeframe
    except Exception as e:
        print(f"[DEBUG] Orchestrator exception: {e}")
        # Fallback to defaults if structured parsing fails
        symbol = state.symbol or "BTCUSDT"
        timeframe = state.timeframe or AnalysisTimeframe.DAILY
    print(f"[DEBUG] Orchestrator output - symbol: {symbol}, timeframe: {timeframe}")
    print(f"[DEBUG] ===== ORCHESTRATOR NODE END =====")

    return {
        **state.model_dump(),
        "messages":     state.messages + [AIMessage(content=f"Routed to {symbol} ({timeframe})")],
        "next_action":  NextAction.MARKET_DATA,
        "symbol":       symbol,
        "timeframe":    timeframe,
        "intermediate_steps_global": state.intermediate_steps_global + [("orchestrator", f"symbol={symbol}, timeframe={timeframe}")],
        "intermediate_steps_agent": [],
    }

def market_data_node(state: QuantAgentState) -> dict:
    print(f"[DEBUG] ===== MARKET DATA NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}, timeframe: {state.timeframe}")
    context = f"Symbol: {state.symbol} | Timeframe: {state.timeframe}"
    print(f"[DEBUG] Context: {context}")
    content, tool_msgs, steps = _run_agent_loop(
        state, llm_market_data, _MARKET_DATA_SYSTEM, context
    )
    print(f"[DEBUG] Market data - steps: {len(steps)}, tool_msgs: {len(tool_msgs)}")
    print(f"[DEBUG] Market data content preview: {content[:200] if content else 'None'}")

    print(f"[DEBUG] Parsing tool results...")
    # Parse valores dos tool results usando Pydantic schemas (STRICT - no regex fallback)
    live_price = state.live_price
    volume_24h = state.volume_24h
    price_change_pct = state.price_change_pct

    for tool_name, result in steps:
        print(f"[DEBUG] Processing tool: {tool_name}, result preview: {result[:100] if result else 'None'}")
        try:
            if "error" in result:
                continue
            if tool_name == "get_live_price":
                parsed = LivePriceOutput.model_validate_json(result)
                live_price = parsed.close
            elif tool_name == "get_indicators":
                parsed = IndicatorsOutput.model_validate_json(result)
                price_change_pct = parsed.pct_change
                volume_24h = parsed.total_volume
                if live_price is None:
                    live_price = parsed.current_price
        except (json.JSONDecodeError, KeyError, TypeError, Exception):
            # Strict parsing: ignore invalid data (institutional rule)
            continue

    # Try to parse structured output from LLM
    anomalies = state.anomalies_detected[:]
    try:
        # content is now a string representation of the structured output
        # We need to extract the JSON from it
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            structured_data = json.loads(json_match.group())
            anomalies.extend(structured_data.get("anomalies", []))
    except Exception:
        # If structured parsing fails, continue without anomalies
        pass

    print(f"[DEBUG] Market data output - live_price: {live_price}, volume_24h: {volume_24h}, price_change_pct: {price_change_pct}")
    print(f"[DEBUG] ===== MARKET DATA NODE END =====")
    return {
        **state.model_dump(),
        "messages":          state.messages + [AIMessage(content=content)],
        "next_action":       NextAction.FEATURES,
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "intermediate_steps_agent": steps,
        "anomalies_detected": anomalies,
        "live_price":        live_price,
        "volume_24h":        volume_24h,
        "price_change_pct":  price_change_pct,
    }

def feature_engineering_node(state: QuantAgentState) -> dict:
    print(f"[DEBUG] ===== FEATURE ENGINEERING NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}, live_price: {state.live_price}")
    context = (
        f"Symbol: {state.symbol} | Timeframe: {state.timeframe} | "
        f"Preço atual: {state.live_price} | "
        f"Anomalias: {state.anomalies_detected}"
    )
    print(f"[DEBUG] Context: {context}")
    content, tool_msgs, steps = _run_agent_loop(
        state, llm_features, _FEATURE_SYSTEM, context
    )
    print(f"[DEBUG] Feature engineering - steps: {len(steps)}, tool_msgs: {len(tool_msgs)}")

    print(f"[DEBUG] Parsing feature tool results...")
    # Parse technical indicator values from tool results using Pydantic schemas (STRICT)
    rsi_14 = state.rsi_14
    macd_line = state.macd_line
    macd_signal = state.macd_signal
    macd_histogram = state.macd_histogram
    bb_upper = state.bb_upper
    bb_middle = state.bb_middle
    bb_lower = state.bb_lower

    for tool_name, result in steps:
        print(f"[DEBUG] Processing feature tool: {tool_name}, result preview: {result[:100] if result else 'None'}")
        try:
            if "error" in result:
                continue
            if tool_name == "get_feature_rsi":
                parsed = RSIOutput.model_validate_json(result)
                rsi_14 = parsed.rsi_14
            elif tool_name == "get_feature_macd":
                parsed = MACDOutput.model_validate_json(result)
                macd_line = parsed.macd_line
                macd_signal = parsed.signal
                macd_histogram = parsed.histogram
            elif tool_name == "get_feature_bollinger":
                parsed = BollingerBandsOutput.model_validate_json(result)
                bb_upper = parsed.upper
                bb_middle = parsed.middle
                bb_lower = parsed.lower
        except (json.JSONDecodeError, KeyError, TypeError, Exception):
            # Strict parsing: ignore invalid data (institutional rule)
            continue

    print(f"[DEBUG] Feature output - rsi: {rsi_14}, macd: {macd_line}, bb_upper: {bb_upper}")
    print(f"[DEBUG] ===== FEATURE ENGINEERING NODE END =====")
    return {
        **state.model_dump(),
        "messages":           state.messages + [AIMessage(content=content)],
        "next_action":        NextAction.RISK,
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "intermediate_steps_agent": steps,
        "rsi_14":             rsi_14,
        "macd_line":          macd_line,
        "macd_signal":        macd_signal,
        "macd_histogram":     macd_histogram,
        "bb_upper":           bb_upper,
        "bb_middle":          bb_middle,
        "bb_lower":           bb_lower,
    }

def risk_agent_node(state: QuantAgentState) -> dict:
    print(f"[DEBUG] ===== RISK AGENT NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}, live_price: {state.live_price}")
    # Passa contexto dos dados já coletados, mas limpa histórico de tool calls
    context = (
        f"Symbol: {state.symbol} | Preço: {state.live_price} | "
        f"Vol 24h: {state.volume_24h} | Variação: {state.price_change_pct}%\n"
        f"DADOS JÁ COLETADOS: RSI={state.rsi_14}, MACD={state.macd_line}, BB={state.bb_upper}\n"
        f"SUA TAREFA: Calcular métricas de risco usando as ferramentas disponíveis."
    )
    print(f"[DEBUG] Context: {context}")
    content, tool_msgs, steps = _run_agent_loop(
        state, llm_risk, _RISK_SYSTEM, context, clear_steps=True
    )
    print(f"[DEBUG] Risk agent - steps: {len(steps)}, tool_msgs: {len(tool_msgs)}")

    print(f"[DEBUG] Parsing risk tool results...")
    # Parse valores dos tool results usando Pydantic schemas (STRICT - no regex fallback)
    var_95 = state.var_95
    cvar_95 = state.cvar_95
    sharpe = state.sharpe
    max_drawdown = state.max_drawdown
    volatility_annualized = state.volatility_annualized
    volatility_raw = None
    volatility_interval = None

    for tool_name, result in steps:
        print(f"[DEBUG] Processing risk tool: {tool_name}, result preview: {result[:100] if result else 'None'}")
        try:
            if "error" in result:
                print(f"[DEBUG] Tool returned error, skipping")
                continue
            if tool_name == "calculate_risk":
                parsed = RiskMetricsOutput.model_validate_json(result)
                cvar_95 = parsed.cvar_95
                sharpe = parsed.sharpe
                max_drawdown = parsed.max_drawdown
                volatility_raw = parsed.volatility_20d
                volatility_interval = "1d"
                print(f"[DEBUG] Parsed calculate_risk: cvar={cvar_95}, sharpe={sharpe}, drawdown={max_drawdown}")
            elif tool_name == "get_feature_cvar":
                parsed = CVaROutput.model_validate_json(result)
                cvar_95 = parsed.cvar_95
                print(f"[DEBUG] Parsed get_feature_cvar: cvar={cvar_95}")
            elif tool_name == "get_feature_sharpe":
                parsed = SharpeOutput.model_validate_json(result)
                sharpe = parsed.sharpe
                print(f"[DEBUG] Parsed get_feature_sharpe: sharpe={sharpe}")
            elif tool_name == "get_feature_max_drawdown":
                parsed = MaxDrawdownOutput.model_validate_json(result)
                max_drawdown = parsed.max_drawdown
                print(f"[DEBUG] Parsed get_feature_max_drawdown: drawdown={max_drawdown}")
            elif tool_name == "get_feature_volatility":
                parsed = VolatilityOutput.model_validate_json(result)
                volatility_raw = parsed.volatility_raw
                volatility_annualized = parsed.volatility_annualized
                volatility_interval = parsed.data_interval
                print(f"[DEBUG] Parsed get_feature_volatility: vol_raw={volatility_raw}, vol_ann={volatility_annualized}")
        except (json.JSONDecodeError, KeyError, TypeError, Exception) as e:
            # Strict parsing: ignore invalid data (institutional rule)
            print(f"[DEBUG] Exception parsing {tool_name}: {type(e).__name__}: {e}")
            continue

    print(f"[DEBUG] Risk metrics before calculation - cvar: {cvar_95}, sharpe: {sharpe}, drawdown: {max_drawdown}, vol: {volatility_annualized}")
    # QUANTITATIVE ENGINE: Deterministic risk level calculation (no LLM)
    risk_level_str = determine_risk_level(
        cvar_95=cvar_95,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        volatility_annualized=volatility_annualized
    )
    risk_level = RiskLevel(risk_level_str)
    print(f"[DEBUG] Calculated risk level: {risk_level}")

    # Annualizar volatilidade baseado no timeframe e interval dos dados
    # Usando log returns para crypto (mais robusto que simple returns)
    # Standard sqrt scaling sem fatores mágicos (institucional)
    if volatility_raw is not None and volatility_annualized is None:
        # Se o intervalo dos dados não foi detectado, usar o timeframe atual
        if volatility_interval is None:
            timeframe = state.timeframe
            if timeframe == AnalysisTimeframe.INTRADAY:
                volatility_interval = "1h"
            elif timeframe == AnalysisTimeframe.DAILY:
                volatility_interval = "1d"
            elif timeframe == AnalysisTimeframe.WEEKLY:
                volatility_interval = "1D"
            else:
                volatility_interval = "1h"

        # Annualizar baseado no intervala dos dados (standard sqrt scaling)
        if volatility_interval == "1m":
            volatility_annualized = volatility_raw * math.sqrt(252 * 1440)
        elif volatility_interval == "1h":
            volatility_annualized = volatility_raw * math.sqrt(252 * 24)
        elif volatility_interval == "1d":
            volatility_annualized = volatility_raw * math.sqrt(252)
        elif volatility_interval in ["1D", "1w", "1W"]:
            volatility_annualized = volatility_raw * math.sqrt(52)
        else:
            volatility_annualized = volatility_raw

    print(f"[DEBUG] ===== RISK AGENT NODE END =====")
    return {
        **state.model_dump(),
        "messages":           state.messages + [AIMessage(content=content)],
        "next_action":        NextAction.SIGNAL,
        "risk_level":         risk_level,
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "intermediate_steps_agent": steps,
        "var_95":             var_95,
        "cvar_95":            cvar_95,
        "sharpe":             sharpe,
        "max_drawdown":       max_drawdown,
        "volatility_annualized": volatility_annualized,
    }

def signal_agent_node(state: QuantAgentState) -> dict:
    print(f"[DEBUG] ===== SIGNAL AGENT NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}, risk_level: {state.risk_level}")
    context = (
        f"Symbol: {state.symbol} | Risk level: {state.risk_level} | "
        f"Sharpe: {state.sharpe} | Drawdown: {state.max_drawdown} | "
        f"CVaR 95%: {state.cvar_95}"
    )
    print(f"[DEBUG] Context: {context}")
    content, tool_msgs, steps = _run_agent_loop(
        state, llm_signal, _SIGNAL_SYSTEM, context
    )
    print(f"[DEBUG] Signal agent - steps: {len(steps)}, tool_msgs: {len(tool_msgs)}")

    print(f"[DEBUG] Parsing signal from content...")
    # Try to parse structured output from LLM for regime
    regime = state.regime
    try:
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            structured_data = json.loads(json_match.group())
            regime = structured_data.get("regime", regime)
    except Exception:
        # If structured parsing fails, keep default
        pass

    print(f"[DEBUG] Input for quant engine - rsi: {state.rsi_14}, macd: {state.macd_line}, bb: {state.bb_upper}, price: {state.live_price}")
    # QUANTITATIVE ENGINE: Deterministic signal calculation (no LLM)
    signal_output = compute_signal_score(
        rsi_14=state.rsi_14,
        macd_line=state.macd_line,
        macd_signal=state.macd_signal,
        bb_upper=state.bb_upper,
        bb_lower=state.bb_lower,
        live_price=state.live_price,
        price_change_pct=state.price_change_pct,
        volatility_annualized=state.volatility_annualized
    )

    # Override regime with quantitative engine result
    regime = signal_output.regime
    direction = signal_output.direction
    signal_confidence = signal_output.confidence
    print(f"[DEBUG] Quant engine output - regime: {regime}, direction: {direction}, confidence: {signal_confidence}")

    print(f"[DEBUG] ===== SIGNAL AGENT NODE END =====")
    return {
        **state.model_dump(),
        "messages":           state.messages + [AIMessage(content=content)],
        "next_action":        NextAction.RISK_GATE,
        "signal_direction":   direction,
        "regime":             regime,
        "signal_confidence":  signal_confidence,
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "intermediate_steps_agent": steps,
    }

def pre_trade_risk_gate_node(state: QuantAgentState) -> dict:
    """
    Nó bloqueador: verifica limites antes de qualquer recomendação de execução.
    Inclui position sizing, portfolio exposure, e liquidity constraints.
    """
    print(f"[DEBUG] ===== RISK GATE NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}, risk_level: {state.risk_level}, signal: {state.signal_direction}")
    print(f"[DEBUG] Volume 24h: {state.volume_24h}, CVaR: {state.cvar_95}, confidence: {state.signal_confidence}")
    blocked  = False
    reasons: list[str] = []
    warnings: list[str] = []

    # Configuration (em produção, viria de config/DB)
    MAX_POSITION_SIZE = 0.10  # 10% do capital por posição
    MAX_PORTFOLIO_EXPOSURE = 0.30  # 30% de exposição total
    MIN_LIQUIDITY_VOLUME = 1_000  # $1M volume mínimo 24h
    MAX_CVAR_PER_POSITION = 0.25  # 25% CVaR máximo por posição

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
    if state.cvar_95 and abs(state.cvar_95) > MAX_CVAR_PER_POSITION:
        blocked = True
        reasons.append(f"CVaR 95% = {state.cvar_95:.1%} — excede limite de {MAX_CVAR_PER_POSITION:.0%} por posição.")

    # Regra 5: Liquidity constraint (volume 24h)
    if state.volume_24h and state.volume_24h < MIN_LIQUIDITY_VOLUME:
        blocked = True
        reasons.append(f"Liquidity insuficiente: volume 24h = ${state.volume_24h:,.0f} < mínimo ${MIN_LIQUIDITY_VOLUME:,.0f}")

    # Regra 6: Position sizing (deterministic via quant_engine)
    if state.signal_confidence is not None:
        position_size = calculate_position_size(
            signal_confidence=state.signal_confidence,
            volatility_annualized=state.volatility_annualized,
            max_position_size=MAX_POSITION_SIZE
        )
        if position_size < 0.01:
            warnings.append(f"Posição reduzida para <1% devido a volatilidade/confiança")

    # Regra 7: Portfolio exposure (institutional portfolio constraints)
    constraints = PortfolioConstraints(
        max_position_size=MAX_POSITION_SIZE,
        max_portfolio_exposure=MAX_PORTFOLIO_EXPOSURE
    )
    
    # Calculate proposed position size
    if state.signal_confidence is not None:
        proposed_size = calculate_position_size(
            signal_confidence=state.signal_confidence,
            volatility_annualized=state.volatility_annualized,
            max_position_size=MAX_POSITION_SIZE
        )
    else:
        proposed_size = 0.0
    
    # Check portfolio constraints
    portfolio_approved, portfolio_violations = check_portfolio_constraints(
        portfolio=state.portfolio_state,
        proposed_symbol=state.symbol,
        proposed_size=proposed_size,
        constraints=constraints
    )
    
    if not portfolio_approved:
        blocked = True
        reasons.extend(portfolio_violations)

    # Regra 8: Volatilidade excessiva
    if state.volatility_annualized and state.volatility_annualized > 1.5:  # 150% a.a.
        warnings.append(f"Volatilidade anualizada muito alta ({state.volatility_annualized:.1%}) — considere reduzir exposição")

    gate_reason  = " | ".join(reasons) if reasons else "Aprovado."
    if warnings:
        gate_reason += f" | Avisos: {'; '.join(warnings)}"
    next_action  = NextAction.BLOCKED if blocked else NextAction.EXECUTION
    print(f"[DEBUG] Risk gate - blocked: {blocked}, reason: {gate_reason}, next_action: {next_action}")
    print(f"[DEBUG] ===== RISK GATE NODE END =====")
    return {
        **state.model_dump(),
        "next_action":  next_action,
        "gate_approved": not blocked,
        "gate_reason":   gate_reason,
        "intermediate_steps_global": state.intermediate_steps_global + [("risk_gate", gate_reason)],
        "intermediate_steps_agent": [],
    }

def execution_node(state: QuantAgentState) -> dict:
    """
    Consolida toda a análise e gera resposta final institucional.
    Inclui validação final antes de responder.
    """
    # VALIDAÇÃO FINAL OBRIGATÓRIA
    if not state.gate_approved:
        raise Exception(f"Execution node chamado sem aprovação do risk gate. Motivo: {state.gate_reason}")

    # Validar dados críticos estão presentes
    if not state.symbol:
        raise Exception("Symbol não definido no estado.")
    if state.signal_direction is None:
        raise Exception("Signal direction não definido no estado.")
    if state.risk_level is None:
        raise Exception("Risk level não definido no estado.")

    context = f"""
DADOS DE MERCADO
  Symbol:          {state.symbol}
  Timeframe:       {state.timeframe}
  Preço atual:     {state.live_price}
  Variação 24h:    {state.price_change_pct}%
  Volume 24h:      {state.volume_24h}

INDICADORES TÉCNICOS
  RSI (14):        {state.rsi_14}
  MACD Line:       {state.macd_line}
  MACD Signal:     {state.macd_signal}
  MACD Histogram:  {state.macd_histogram}
  Bollinger Upper: {state.bb_upper}
  Bollinger Middle:{state.bb_middle}
  Bollinger Lower: {state.bb_lower}

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
{chr(10).join(f'  [{k}] {v[:120]}' for k, v in state.intermediate_steps_global[-8:])}
""".strip()

    # Usa apenas o contexto estruturado — state.messages pode conter mensagens
    # de outros agentes incompatíveis com a API OpenAI (órfãos de tool_calls)
    msgs = [
        SystemMessage(content=_EXECUTION_SYSTEM),
        HumanMessage(content=context),
    ]
    final_response = llm_execution.invoke(msgs)

    return {
        **state.model_dump(),
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
        **state.model_dump(),
        "final_answer": msg,
        "next_action":  NextAction.FINALIZE,
    }

def finalize_node(state: QuantAgentState) -> dict:
    """Extrai a resposta final — nó terminal."""
    return {**state.model_dump(), "next_action": "done"}

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
        result = await run_analysis("Analise de criptomoeda com gestão de risco institucional")
        print(result)
    """
    graph  = get_agent_graph()
    state  = QuantAgentState(
        messages=[HumanMessage(content=user_message)],
        user_id=user_id,
    )
    result = await graph.ainvoke(state)
    return result["final_answer"]