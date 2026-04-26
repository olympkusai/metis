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

import asyncio
import json
import math
from datetime import UTC, datetime
from enum import Enum
from functools import lru_cache
from app.utils.timing import timed_async, timed
from app.utils.cost_tracker import get_cost_tracker
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ConfigDict

from app.tools import all_tools
from app.apollo_client import ApolloApiError, get_apollo_client
from app.config import get_settings
from app.agent.forecasting import (
    assess_forecast_quality,
    build_prediction_window,
    calculate_error_pct,
    calculate_return_pct,
    overlay_forecast_on_signal,
)
from app.agent.schemas import (
    ApolloBacktestOutput,
    ApolloPredictionOutput,
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
from app.agent.moe import MoESignalLayer
from app.agent.quant_engine import (
    compute_signal_score,
    determine_risk_level,
    calculate_position_size,
)
from app.agent.trend_state import (
    TrendStateMachine,
    MultiTimeframeInterpreter,
    TrendState,
    TrendDirection,
)
from app.agent.decision_engine import (
    DecisionEngine,
    DecisionOutput,
    MacroTrend,
    ExecutionState,
    FinalSignal,
    SignalType,
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
    FEATURES_MACRO  = "features_macro"   # Daily features
    FEATURES_SETUP  = "features_setup"   # 4H features
    FEATURES_EXEC   = "features_exec"    # 1H features
    FORECAST       = "forecast"
    TREND_INTERPRET = "trend_interpret"   # Multi-timeframe interpretation
    DECISION_ENGINE = "decision_engine"   # Deterministic decision layer (FINAL authority)
    RISK           = "risk"
    SIGNAL         = "signal"
    MOE            = "moe"
    RISK_GATE      = "risk_gate"
    EXECUTION      = "execution"
    FINALIZE       = "finalize"
    BLOCKED        = "blocked"

# Multi-timeframe structure
class TimeframeLayer(str, Enum):
    MACRO = "daily"    # 1D - regime detection
    SETUP = "4h"       # 4H - signal generation
    EXECUTION = "1h"   # 1H - execution timing

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
    recent_high:        float | None       = None  # from get_indicators (last 100 candles)
    recent_low:         float | None       = None

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

    # Multi-timeframe indicators (separated by layer)
    # Macro layer (1D) - regime detection
    macro_rsi_14:       float | None       = None
    macro_macd_line:    float | None       = None
    macro_macd_signal:  float | None       = None
    macro_bb_upper:     float | None       = None
    macro_bb_lower:     float | None       = None
    macro_bb_pct_b:     float | None       = None
    macro_regime:       str | None         = None   # "trending" | "ranging" | "breakout" | "volatile" | "reversal"
    macro_bias:         str | None         = None   # "bullish" | "bearish" | "neutral"
    
    # Trend state context (new hierarchical interpretation)
    trend_state:        str | None         = None   # "trending" | "overextended" | "pullback" | "reversal" | "neutral"
    trend_direction:    str | None         = None   # "bullish" | "bearish" | "neutral"
    signal_type:        str | None         = None   # "trend_follow" | "pullback_entry" | "breakout" | "reversal" | "no_edge"
    execution_timing:   str | None         = None   # "immediate" | "wait_for_pullback" | "wait_for_confirmation" | "wait"
    
    # Final decision (from Decision Engine - FINAL authority)
    final_signal:       str | None         = None   # "long" | "short" | "conditional_long" | "conditional_short" | "wait" | "neutral"
    final_signal_type:  str | None         = None   # "trend_follow" | "pullback_entry" | "breakout" | "reversal" | "no_edge"
    final_confidence:   float | None       = None   # 0 to 1
    final_reasoning:    str | None         = None   # Explanation of final decision

    # Setup layer (4H) - signal generation
    setup_rsi_14:       float | None       = None
    setup_macd_line:    float | None       = None
    setup_macd_signal:  float | None       = None
    setup_bb_upper:     float | None       = None
    setup_bb_lower:     float | None       = None
    setup_bb_pct_b:     float | None       = None
    setup_signal:       float | None       = None   # -1 to 1
    setup_confidence:   float | None       = None   # 0..1
    setup_direction:    str | None         = None   # "long" | "short" | "weak_long" | "weak_short"

    # Execution layer (1H) - timing + risk
    exec_rsi_14:        float | None       = None
    exec_macd_line:     float | None       = None
    exec_macd_signal:   float | None       = None
    exec_sharpe:        float | None       = None
    exec_volatility:    float | None       = None
    exec_timing:        str | None         = None   # "entry" | "wait" | "exit"

    # Apollo ML forecast context
    forecast_period_start: str | None      = None
    forecast_period_end:   str | None      = None
    forecast_data_points:  int | None      = None
    forecast_current_price: float | None   = None
    forecast_predicted_price: float | None = None
    forecast_direction:    str | None      = None
    forecast_confidence:   float | None    = None
    forecast_model_mape:   float | None    = None
    forecast_period_volatility: float | None = None
    forecast_data_quality: str | None      = None
    forecast_return_pct:   float | None    = None
    forecast_backtest_error_pct: float | None = None
    forecast_training_attempts: int        = 0
    forecast_actionable:  bool             = False
    forecast_status:      str              = "not_requested"
    forecast_warnings:    list[str]        = Field(default_factory=list)

    # Signal context (legacy, will be phased out)
    regime:             str | None         = None   # "trending" | "ranging" | "breakout"
    signal_direction:   str | None         = None   # "long" | "short" | "neutral"
    signal_confidence:  float | None       = None   # 0..1

    # MoE output (populated by MoE node)
    moe_final_signal:       float | None   = None   # -1 to 1
    moe_final_confidence:   float | None   = None   # 0 to 1
    moe_selected_experts:   list[str]     = Field(default_factory=list)
    moe_expert_weights:     dict[str, float] = Field(default_factory=dict)
    moe_gating_reason:      str            = ""
    moe_position_size:      float | None   = None   # 0 to 1 (fraction of capital)
    moe_risk_adjusted_signal: float | None = None   # signal * position_size

    # Risk gate output
    gate_approved:      bool | None        = None
    gate_reason:        str                = ""

    # Data quality
    anomalies_detected: list[str]          = Field(default_factory=list)

    # Portfolio context (for institutional portfolio management)
    portfolio_state: PortfolioState       = Field(default_factory=PortfolioState)
    proposed_position_size: float | None   = None

    # Tool call deduplication cache (per run) — keyed by '<name>|<sorted_args>'
    tool_cache:         dict[str, str]     = Field(default_factory=dict)

    # Chain-of-thought (per node, for streaming)
    cot:                str                = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

# ─────────────────────────────────────────────
# 3. NODE CONFIGURATION  (model + CoT per node)
# ─────────────────────────────────────────────

NODE_CONFIG = {
    "orchestrator": {"model": "gpt-4o-mini", "cot": True},
    "market_data": {"model": "gpt-4o-mini", "cot": True},
    "feature_engineering": {"model": "gpt-4o-mini", "cot": True},
    "risk_agent": {"model": "gpt-4o-mini", "cot": True},
    "signal_agent": {"model": "gpt-4o", "cot": True},
    "moe": {"model": "gpt-4o-mini", "cot": False},  # Deterministic, no LLM
    "forecast": {"model": "none", "cot": False},
    "forecast_question": {"model": "gpt-4o-mini", "cot": False},
    "risk_gate": {"model": "gpt-4o", "cot": True},
    "execution": {"model": "gpt-4o", "cot": True},
}

# ─────────────────────────────────────────────
# 4. LLM INSTANCES  (per agent, tuned separately)
# ─────────────────────────────────────────────

_BASE_MODEL = "gpt-4o"

def _make_llm(model: str = _BASE_MODEL, temperature: float = 0.1, **kw) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.openai_api_key,
        **kw,
    )

tool_map = {t.name: t for t in all_tools}
risk_tools = [t for t in all_tools if t.name == "calculate_risk"]

def _extract_cot_and_answer(content: str) -> tuple[str, str]:
    """
    Extrai CoT e answer de uma resposta formatada como:
    <thought>...</thought><answer>...</answer>
    
    Retorna (cot, answer). Se não encontrar formato, retorna ("", content).
    """
    import re
    # Try to extract <thought>...</thought><answer>...</answer>
    thought_match = re.search(r'<thought>(.*?)</thought>', content, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    
    if thought_match and answer_match:
        cot = thought_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return cot, answer
    elif thought_match:
        # Only thought found, assume rest is answer
        cot = thought_match.group(1).strip()
        answer = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL).strip()
        return cot, answer
    else:
        # No CoT format found
        return "", content

def _get_llm_for_node(node_name: str):
    """Retorna LLM configurado para o nó específico baseado no NODE_CONFIG."""
    config = NODE_CONFIG.get(node_name, {"model": _BASE_MODEL, "cot": False})
    model = config["model"]
    
    if node_name == "orchestrator":
        return _make_llm(model=model, temperature=0.1).with_structured_output(OrchestratorOutput)
    elif node_name == "market_data":
        return _make_llm(model=model, temperature=0.0).bind_tools(all_tools)
    elif node_name == "feature_engineering":
        return _make_llm(model=model, temperature=0.0).bind_tools(all_tools, parallel_tool_calls=True)
    elif node_name == "risk_agent":
        return _make_llm(model=model, temperature=0.0).bind_tools(risk_tools)
    elif node_name == "signal_agent":
        return _make_llm(model=model, temperature=0.15).bind_tools(all_tools)
    elif node_name == "risk_gate":
        return _make_llm(model=model, temperature=0.0)  # no tools
    elif node_name == "execution":
        return _make_llm(model=model, temperature=0.0)  # no tools — pure reasoning
    else:
        return _make_llm(model=model, temperature=0.1)

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
1. Coletar dados de mercado do banco de dados local (market_candles table)
2. Detectar anomalias nos dados ANTES de reportar
3. Normalizar e contextualizar os valores

ANOMALY DETECTION (OBRIGATÓRIO):
- Timestamps inconsistentes (alternância :59/:00 suspeita)
- Price spikes > 3σ em janela de 5min
- Volume 0 com candle fechado
- Preços idênticos em N candles consecutivos (provável bug de duplicação)
- Spread bid/ask > 2% (illiquidity flag)

Se detectar anomalia: adicione ao campo anomalies_detected e use os dados COM CAUTELA.

INSTRUÇÃO IMPORTANTE: Você DEVE chamar as ferramentas get_live_price e get_indicators para obter os dados.
TODAS as ferramentas usam cálculo local - NENHUMA chamada a API externa.

FERRAMENTAS: get_live_price, get_indicators
OUTPUT: Preencha live_price, volume_24h, price_change_pct, anomalies_detected
""".strip()

_FEATURE_SYSTEM = """
Você é o Feature Engineering Agent de um fundo quantitativo.

RESPONSABILIDADES:
1. Calcular indicadores técnicos localmente usando CalculationEngine
2. Usar dados do banco de dados (market_candles table)
3. Normalizar TODOS os valores antes de reportar

REGRAS DE TIMEFRAME:
- INTRADAY: interval="1m" ou "5m" — retornos em % por minuto
- DAILY: interval="1h" — retornos em % por hora  
- WEEKLY: interval="1D" — retornos em % por dia

NORMALIZAÇÃO OBRIGATÓRIA:
- Volatilidade 1m: anualizar = sqrt(252 × 1440) × vol_1m
- Momentum: converter para z-score = (valor - média_30d) / std_30d
- RSI: escala 0-100 (padrão da indústria) — overbought >70, oversold <30, neutro 30-70
- Bollinger Bands: reportar %B = (close - lower) / (upper - lower); breakout se %B > 1 ou < 0
- MACD: reportar cruzamento (bull cross / bear cross / divergência)

INSTRUÇÃO IMPORTANTE: Você DEVE chamar as ferramentas get_feature_rsi, get_feature_macd, get_feature_bollinger, get_feature_volatility para calcular os indicadores.
TODAS as ferramentas usam cálculo local com CalculationEngine - NENHUMA chamada a API externa.
Chame-as em PARALELO.

FERRAMENTAS (chame em PARALELO): get_feature_rsi, get_feature_macd, get_feature_bollinger, get_feature_volatility
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

INSTRUÇÃO IMPORTANTE: Você DEVE chamar a ferramenta calculate_risk para calcular as métricas de risco.
Ela usa CalculationEngine local - NENHUMA chamada a API externa.
Ela já consolida CVaR, Sharpe, Max Drawdown e volatilidade em uma única consulta.

FERRAMENTAS: calculate_risk
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

INSTRUÇÃO IMPORTANTE: Use apenas os dados já coletados pelos agentes upstream (market_data, features, risk).
NÃO chame ferramentas de API externa - todos os dados já estão no contexto.

FERRAMENTAS: Nenhuma - use dados do contexto
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

MODELO PREDITIVO (AUXILIAR):
• Trate o forecast de ML como sinal secundário, nunca como fonte única.
• Se confiança, MAPE ou qualidade de dados estiverem ruins, diga explicitamente que o forecast não é acionável.
• Se houver conflito entre sinais técnicos e forecast, reporte o conflito em ALERTAS.

REGRAS CRÍTICAS ANTI-ALUCINAÇÃO:
• Se um campo aparece como "(não coletado)" no contexto, reporte-o como indisponível.
• Se um campo tem valor numérico, USE-O. NUNCA afirme que um dado está
  indisponível se ele está preenchido no contexto.
• Se há anomalias listadas (incluindo ajustes do sanity audit), mencione-as em ALERTAS.
• NÃO invente valores. NÃO extrapole. Sintetize apenas o que está no contexto.

Tom: analista institucional sênior. Preciso, direto, sem jargão desnecessário.
""".strip()

_FORECAST_RESPONSE_SYSTEM = """
Você é um assistente especializado em análise de previsões de machine learning para ativos cripto.

CONTEXTO:
O modelo Apollo foi treinado com dados históricos usando técnicas de Time Series Forecasting (TFT + XGBoost).
Você tem acesso a:
- Previsão de preço para um período específico
- Nível de confiança do modelo (0-100%)
- Erro médio do modelo (MAPE - Mean Absolute Percentage Error)
- Qualidade dos dados de treinamento
- Volatilidade histórica do período

DIRETRIZES PARA RESPOSTA:

1. AVALIAÇÃO DE QUALIDADE
   • Confiança >= 60% E MAPE <= 2.5% E Data Quality = "good" → Forecast é ACIONÁVEL
   • Qualquer condição fora disso → Forecast é AUXILIAR apenas, nunca use como decisão única

2. INTERPRETAÇÃO DO VIÉS DIRECIONAL
   • DOWN: preço previsto < preço atual (retorno negativo esperado)
   • UP: preço previsto > preço atual (retorno positivo esperado)
   • FLAT: variação < 0.5% em ambas as direções

3. ESTRUTURA DE RESPOSTA
   a) STATUS DO FORECAST: Está acionável? Qualidade dos dados?
   b) PREVISÃO DIRECIONAL: Qual é a direção prevista e o retorno esperado?
   c) CONFIANÇA: Nível de confiança com contexto (se baixa, explique)
   d) LIMITAÇÕES: Cite explicitamente qualquer fraqueza (MAPE alto, confiança baixa, dados limitados)
   e) DISCLAIMER: "Esta é uma previsão de ML — não substitui análise técnica completa nem é recomendação de investimento"

4. LINGUAGEM
   • Use tom profissional mas acessível
   • Sempre cite os números (confiança, MAPE, preço previsto)
   • Seja honesto sobre limitações — nunca "venda" a previsão como certa
   • Se pergunta for sobre período muito curto (< 30 dias), avise que modelo é mais confiável para períodos maiores

5. EXEMPLO DE BOA RESPOSTA:
   "O modelo Apollo prevê Bitcoin DOWN com retorno estimado de -4.3% para o período 2026-01-01 até 2026-04-25.
   - Confiança: 60.9% (marginal, acima do mínimo)
   - Erro histórico (MAPE): 1.86% (excelente, bem abaixo do limite)
   - Qualidade dados: Good
   - Volatilidade esperada: 2.91%

   ⚠️ IMPORTANTE: Este é um sinal auxiliar baseado em ML. Não use como única base para decisão."

FERRAMENTAS: Nenhuma — use apenas o contexto de forecast fornecido
""".strip()

# ─────────────────────────────────────────────
# 5. TOOL EXECUTOR  (reutilizável)
# ─────────────────────────────────────────────

# Tools cujo argumento `interval` é canônico (devem respeitar o timeframe do pipeline)
_INTERVAL_AWARE_TOOLS = {
    "get_live_price", "get_indicators", "calculate_risk",
    "get_feature_rsi", "get_feature_macd", "get_feature_bollinger",
    "get_feature_volatility", "get_feature_sharpe", "get_feature_cvar",
    "get_feature_max_drawdown", "get_feature_sma", "get_feature_ema_return",
    "get_ohlcv_history",
}


def _cache_key(name: str, args: dict) -> str:
    try:
        return f"{name}|{json.dumps(args, sort_keys=True, default=str)}"
    except Exception:
        return f"{name}|{repr(sorted(args.items()))}"


async def _execute_tools(
    last_ai: AIMessage,
    steps: list,
    cache: dict[str, str] | None = None,
    force_interval: str | None = None,
) -> tuple[list[ToolMessage], list]:
    """Executa tool calls em paralelo, com dedup cache (por-run) e enforcement
    do `interval` canônico derivado do timeframe do orchestrator.
    """
    tool_messages: list[ToolMessage] = []
    steps = list(steps)
    cache = cache if cache is not None else {}

    async def _run_one(tc):
        name = tc["name"]
        args = dict(tc.get("args") or {})
        # Enforce timeframe canônico
        if force_interval and name in _INTERVAL_AWARE_TOOLS:
            # Always enforce interval, even if LLM didn't provide it
            if "interval" not in args or args.get("interval") != force_interval:
                args["interval"] = force_interval
        # Dedup cache
        key = _cache_key(name, args)
        if key in cache:
            return name, tc["id"], cache[key], True
        tool = tool_map.get(name)
        if tool is None:
            return name, tc["id"], f"[ERR] Tool '{name}' não encontrada.", False
        result = await tool.ainvoke(args)
        cache[key] = str(result)
        return name, tc["id"], result, False

    results = await asyncio.gather(
        *[_run_one(tc) for tc in last_ai.tool_calls],
        return_exceptions=True,
    )

    for item, tc in zip(results, last_ai.tool_calls):
        if isinstance(item, Exception):
            name = tc["name"]
            result = f"[ERR] Tool '{name}' exceção: {item}"
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            steps.append((name, str(result)))
        else:
            name, call_id, result, was_cached = item
            content = str(result)
            tool_messages.append(ToolMessage(content=content, tool_call_id=call_id))
            label = f"{name} [cached]" if was_cached else name
            steps.append((label, content))
    return tool_messages, steps

async def _run_agent_loop(
    state: QuantAgentState,
    llm,
    system_prompt: str,
    extra_context: str = "",
    clear_steps: bool = False,
    force_interval: str | None = None,
    node_name: str = "unknown",
    enable_cot: bool = False,
) -> tuple[str, str, list[ToolMessage], list]:
    """
    Loop LLM → tool call → result para um agente especializado.
    Usa APENAS a mensagem original do usuário — não o histórico acumulado
    de outros agentes (que pode conter tool_calls sem resposta).
    Retorna (resposta_final, cot, tool_messages_acumulados, steps_acumulados).
    Includes retry mechanism for LLM calls.
    """
    # Extrai apenas a mensagem original do usuário
    original_query = next((m for m in state.messages if isinstance(m, HumanMessage)), None)

    # Adiciona instrução explícita para chamar ferramentas
    tool_instruction = "\n\nIMPORTANTE: Você tem acesso a ferramentas. Você DEVE chamar as ferramentas apropriadas antes de fornecer sua análise final. Não responda sem chamar as ferramentas primeiro."
    
    # Adiciona instrução de CoT se habilitado
    cot_instruction = ""
    if enable_cot:
        cot_instruction = "\n\nFORMATO DE RESPOSTA OBRIGATÓRIO:\n<thought>Resumo CONCISO do seu raciocínio (max 2-3 frases). Ex: 'Obtive o indicador X que está em Y, indicando Z.'</thought>\n<answer>Sua resposta final aqui</answer>"
    
    enhanced_prompt = system_prompt + tool_instruction + cot_instruction

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
                response = await llm.ainvoke(msgs, timeout=30)  # 30s timeout
                
                # Track token usage
                cost_tracker = get_cost_tracker()
                if hasattr(response, 'response_metadata'):
                    metadata = response.response_metadata
                    input_tokens = metadata.get('token_usage', {}).get('prompt_tokens', 0)
                    output_tokens = metadata.get('token_usage', {}).get('completion_tokens', 0)
                    model_name = getattr(llm, 'model_name', 'unknown') or getattr(llm, 'model', 'unknown')
                    if input_tokens > 0 or output_tokens > 0:
                        cost_tracker.add_call(model_name, input_tokens, output_tokens, node_name)
                
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    # Fallback: return error message
                    error_msg = f"[LLM ERROR after {MAX_RETRIES} retries]: {str(e)}"
                    return error_msg, "", all_tool_msgs, steps
                # Exponential backoff: 0.5s, 1s, 2s (não-bloqueante)
                backoff_time = 0.5 * (2 ** attempt)
                await asyncio.sleep(backoff_time)
                continue
        
        if response is None:
            error_msg = "[LLM ERROR] Failed to get response after retries"
            return error_msg, "", all_tool_msgs, steps
            
        msgs.append(response)
        
        # Debug: log response type and tool_calls
        print(f"[DEBUG] Response type: {type(response).__name__}")
        if hasattr(response, 'tool_calls'):
            print(f"[DEBUG] Tool calls: {response.tool_calls}")
        print(f"[DEBUG] Response content preview: {str(response)[:200]}")
        
        # Handle both AIMessage (with tool_calls) and structured Pydantic output
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_msgs, steps = await _execute_tools(
                response, steps, cache=state.tool_cache, force_interval=force_interval
            )
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
            cot, answer = _extract_cot_and_answer(content) if enable_cot else ("", content)
            return answer, cot, all_tool_msgs, steps
        else:
            # Regular AIMessage without tool calls
            print(f"[DEBUG] No tool calls found, returning content directly")
            cot, answer = _extract_cot_and_answer(response.content) if enable_cot else ("", response.content)
            return answer, cot, all_tool_msgs, steps
        
        # Detect no-progress loops
        if len(steps) == previous_step_count:
            no_progress_count += 1
            if no_progress_count >= 2:
                # No progress for 2 iterations - abort early
                error_msg = "[AGENT LOOP] No progress detected - aborting to prevent infinite loop"
                return error_msg, "", all_tool_msgs, steps
        else:
            no_progress_count = 0
            previous_step_count = len(steps)

    # Força finalização se exceder iterações
    final = await llm.ainvoke(msgs + [HumanMessage(content="Sintetize os resultados obtidos até agora.")])
    cot, answer = _extract_cot_and_answer(final.content) if enable_cot else ("", final.content)
    return answer, cot, all_tool_msgs, steps

# ─────────────────────────────────────────────
# 6. AGENT NODES
# ─────────────────────────────────────────────

@timed_async("Node: Orchestrator")
async def orchestrator_node(state: QuantAgentState) -> dict:
    """
    Extrai intent, symbol, timeframe e define o pipeline inicial.
    Uses structured output from LLM.
    """
    print(f"[DEBUG] ===== ORCHESTRATOR NODE START =====")
    print(f"[DEBUG] Input state symbol: {state.symbol}")
    print(f"[DEBUG] Input state messages: {len(state.messages)} messages")
    msgs  = [SystemMessage(content=_ORCHESTRATOR_SYSTEM)] + state.messages
    try:
        llm = _get_llm_for_node("orchestrator")
        resp = await llm.ainvoke(msgs)
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
        "cot":          "",
    }

# Derive canonical interval from analysis timeframe (single source of truth)
_TIMEFRAME_INTERVAL = {
    AnalysisTimeframe.INTRADAY: "1m",
    AnalysisTimeframe.DAILY: "1h",
    AnalysisTimeframe.WEEKLY: "1D",
}

# Multi-timeframe mapping for swing trade (1D + 4H + 1H)
_MULTI_TF_INTERVALS = {
    TimeframeLayer.MACRO: "1D",      # Daily for regime detection
    TimeframeLayer.SETUP: "4h",     # 4H for signal generation
    TimeframeLayer.EXECUTION: "1h",  # 1H for execution timing
}

def _get_analysis_interval(timeframe: AnalysisTimeframe) -> str:
    """Returns the canonical data interval for a given analysis timeframe."""
    return _TIMEFRAME_INTERVAL.get(timeframe, "1h")

def _get_multi_tf_interval(layer: TimeframeLayer) -> str:
    """Returns the interval for a given multi-timeframe layer."""
    return _MULTI_TF_INTERVALS.get(layer, "1h")

@timed_async("Node: MarketData")
async def market_data_node(state: QuantAgentState) -> dict:
    print(f"[DEBUG] ===== MARKET DATA NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}, timeframe: {state.timeframe}")
    analysis_interval = _get_analysis_interval(state.timeframe)
    context = (
        f"Symbol: {state.symbol} | Timeframe: {state.timeframe} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{analysis_interval}\" — use ESTE intervalo em TODAS as ferramentas"
    )
    print(f"[DEBUG] Context: {context}")
    node_config = NODE_CONFIG["market_data"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("market_data"), _MARKET_DATA_SYSTEM, context,
        force_interval=analysis_interval, node_name="MarketData", enable_cot=node_config["cot"]
    )
    print(f"[DEBUG] Market data - steps: {len(steps)}, tool_msgs: {len(tool_msgs)}")
    print(f"[DEBUG] Market data content preview: {content[:200] if content else 'None'}")

    print(f"[DEBUG] Parsing tool results...")
    # Parse valores dos tool results usando Pydantic schemas (STRICT - no regex fallback)
    live_price = state.live_price
    volume_24h = state.volume_24h
    price_change_pct = state.price_change_pct
    recent_high = state.recent_high
    recent_low = state.recent_low

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
                recent_high = parsed.high
                recent_low = parsed.low
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

    # Data consistency check: detect conflicting values from cache vs fresh calculation
    # This prevents the RSI inconsistency issue (22.49 vs 48.91)
    data_anomalies = []
    if state.rsi_14 is not None:
        # Check if RSI is in valid range
        if state.rsi_14 < 0 or state.rsi_14 > 100:
            data_anomalies.append(f"RSI value {state.rsi_14} out of valid range [0, 100]")
            live_price = None  # Force recalculation on next run
            price_change_pct = None

    if data_anomalies:
        anomalies.extend(data_anomalies)
        print(f"[DEBUG] Data consistency anomalies detected: {data_anomalies}")

    print(f"[DEBUG] Market data output - live_price: {live_price}, volume_24h: {volume_24h}, price_change_pct: {price_change_pct}")
    print(f"[DEBUG] ===== MARKET DATA NODE END =====")
    return {
        **state.model_dump(),
        "messages":          state.messages + [AIMessage(content=content)],
        "next_action":       NextAction.FEATURES_MACRO,  # Route to multi-timeframe structure
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "intermediate_steps_agent": steps,
        "anomalies_detected": anomalies,
        "live_price":        live_price,
        "volume_24h":        volume_24h,
        "price_change_pct":  price_change_pct,
        "recent_high":       recent_high,
        "recent_low":        recent_low,
        "cot":               cot,
    }

@timed_async("Node: FeatureEngineering-Macro")
async def features_macro_node(state: QuantAgentState) -> dict:
    """
    Multi-timeframe: Macro layer (1D) - regime detection.
    Defines market regime and directional bias.
    """
    print(f"[DEBUG] ===== FEATURES MACRO NODE START (1D) =====")
    print(f"[DEBUG] Input symbol: {state.symbol}")
    
    # Use daily interval for regime detection
    macro_interval = _get_multi_tf_interval(TimeframeLayer.MACRO)
    context = (
        f"Symbol: {state.symbol} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{macro_interval}\" — Daily timeframe for regime detection"
    )
    print(f"[DEBUG] Context: {context}")
    
    node_config = NODE_CONFIG["feature_engineering"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("feature_engineering"), _FEATURE_SYSTEM, context,
        force_interval=macro_interval, node_name="FeatureEngineering-Macro", enable_cot=node_config["cot"]
    )
    
    # Parse macro features
    macro_rsi_14 = state.macro_rsi_14
    macro_macd_line = state.macro_macd_line
    macro_macd_signal = state.macro_macd_signal
    macro_bb_upper = state.macro_bb_upper
    macro_bb_lower = state.macro_bb_lower
    macro_bb_pct_b = state.macro_bb_pct_b
    macro_regime = state.macro_regime
    
    for tool_name, result in steps:
        try:
            if "error" in result:
                continue
            if tool_name == "get_feature_rsi":
                parsed = RSIOutput.model_validate_json(result)
                macro_rsi_14 = parsed.rsi_14
                macro_regime = parsed.regime
            elif tool_name == "get_feature_macd":
                parsed = MACDOutput.model_validate_json(result)
                macro_macd_line = parsed.macd_line
                macro_macd_signal = parsed.signal
            elif tool_name == "get_feature_bollinger":
                parsed = BollingerBandsOutput.model_validate_json(result)
                macro_bb_upper = parsed.upper
                macro_bb_lower = parsed.lower
                # Calculate %B
                if parsed.upper and parsed.lower and state.live_price:
                    bandwidth = parsed.upper - parsed.lower
                    if abs(bandwidth) > 1e-8:
                        macro_bb_pct_b = (state.live_price - parsed.lower) / bandwidth
                        macro_bb_pct_b = max(0.0, min(1.0, macro_bb_pct_b))
        except Exception:
            continue
    
    # Calculate macro bias based on regime
    if macro_regime == "overbought":
        macro_bias = "bullish_stretched"
    elif macro_regime == "oversold":
        macro_bias = "bearish_stretched"
    elif macro_regime == "neutral":
        macro_bias = "neutral"
    else:
        macro_bias = None
    
    print(f"[DEBUG] Macro features - rsi: {macro_rsi_14}, macd: {macro_macd_line}, regime: {macro_regime}, bias: {macro_bias}")
    print(f"[DEBUG] ===== FEATURES MACRO NODE END =====")
    
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=content)],
        "next_action": NextAction.FEATURES_SETUP,
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "intermediate_steps_agent": steps,
        "macro_rsi_14": macro_rsi_14,
        "macro_macd_line": macro_macd_line,
        "macro_macd_signal": macro_macd_signal,
        "macro_bb_upper": macro_bb_upper,
        "macro_bb_lower": macro_bb_lower,
        "macro_bb_pct_b": macro_bb_pct_b,
        "macro_regime": macro_regime,
        "macro_bias": macro_bias,
        "cot": cot,
    }


@timed_async("Node: FeatureEngineering-Setup")
async def features_setup_node(state: QuantAgentState) -> dict:
    """
    Multi-timeframe: Setup layer (4H) - signal generation.
    Identifies concrete trading opportunities.
    """
    print(f"[DEBUG] ===== FEATURES SETUP NODE START (4H) =====")
    print(f"[DEBUG] Input symbol: {state.symbol}")
    
    # Use 4H interval for signal generation
    setup_interval = _get_multi_tf_interval(TimeframeLayer.SETUP)
    context = (
        f"Symbol: {state.symbol} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{setup_interval}\" — 4H timeframe for signal generation"
    )
    print(f"[DEBUG] Context: {context}")
    
    node_config = NODE_CONFIG["feature_engineering"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("feature_engineering"), _FEATURE_SYSTEM, context,
        force_interval=setup_interval, node_name="FeatureEngineering-Setup", enable_cot=node_config["cot"]
    )
    
    # Parse setup features
    setup_rsi_14 = state.setup_rsi_14
    setup_macd_line = state.setup_macd_line
    setup_macd_signal = state.setup_macd_signal
    setup_bb_upper = state.setup_bb_upper
    setup_bb_lower = state.setup_bb_lower
    setup_bb_pct_b = state.setup_bb_pct_b
    
    for tool_name, result in steps:
        try:
            if "error" in result:
                continue
            if tool_name == "get_feature_rsi":
                parsed = RSIOutput.model_validate_json(result)
                setup_rsi_14 = parsed.rsi_14
            elif tool_name == "get_feature_macd":
                parsed = MACDOutput.model_validate_json(result)
                setup_macd_line = parsed.macd_line
                setup_macd_signal = parsed.signal
            elif tool_name == "get_feature_bollinger":
                parsed = BollingerBandsOutput.model_validate_json(result)
                setup_bb_upper = parsed.upper
                setup_bb_lower = parsed.lower
                # Calculate %B
                if parsed.upper and parsed.lower and state.live_price:
                    bandwidth = parsed.upper - parsed.lower
                    if abs(bandwidth) > 1e-8:
                        setup_bb_pct_b = (state.live_price - parsed.lower) / bandwidth
                        setup_bb_pct_b = max(0.0, min(1.0, setup_bb_pct_b))
        except Exception:
            continue
    
    print(f"[DEBUG] Setup features - rsi: {setup_rsi_14}, macd: {setup_macd_line}")
    print(f"[DEBUG] ===== FEATURES SETUP NODE END =====")
    
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=content)],
        "next_action": NextAction.FEATURES_EXEC,
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "intermediate_steps_agent": steps,
        "setup_rsi_14": setup_rsi_14,
        "setup_macd_line": setup_macd_line,
        "setup_macd_signal": setup_macd_signal,
        "setup_bb_upper": setup_bb_upper,
        "setup_bb_lower": setup_bb_lower,
        "setup_bb_pct_b": setup_bb_pct_b,
        "cot": cot,
    }


@timed_async("Node: FeatureEngineering-Exec")
async def features_exec_node(state: QuantAgentState) -> dict:
    """
    Multi-timeframe: Execution layer (1H) - timing + risk.
    Provides execution timing and risk metrics.
    """
    print(f"[DEBUG] ===== FEATURES EXEC NODE START (1H) =====")
    print(f"[DEBUG] Input symbol: {state.symbol}")
    
    # Use 1H interval for execution timing
    exec_interval = _get_multi_tf_interval(TimeframeLayer.EXECUTION)
    context = (
        f"Symbol: {state.symbol} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{exec_interval}\" — 1H timeframe for execution timing"
    )
    print(f"[DEBUG] Context: {context}")
    
    node_config = NODE_CONFIG["feature_engineering"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("feature_engineering"), _FEATURE_SYSTEM, context,
        force_interval=exec_interval, node_name="FeatureEngineering-Exec", enable_cot=node_config["cot"]
    )
    
    # Parse execution features
    exec_rsi_14 = state.exec_rsi_14
    exec_macd_line = state.exec_macd_line
    exec_macd_signal = state.exec_macd_signal
    
    for tool_name, result in steps:
        try:
            if "error" in result:
                continue
            if tool_name == "get_feature_rsi":
                parsed = RSIOutput.model_validate_json(result)
                exec_rsi_14 = parsed.rsi_14
            elif tool_name == "get_feature_macd":
                parsed = MACDOutput.model_validate_json(result)
                exec_macd_line = parsed.macd_line
                exec_macd_signal = parsed.signal
        except Exception:
            continue
    
    print(f"[DEBUG] Exec features - rsi: {exec_rsi_14}")
    print(f"[DEBUG] ===== FEATURES EXEC NODE END =====")
    
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=content)],
        "next_action": NextAction.FORECAST,  # Route to forecast before interpretation
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "intermediate_steps_agent": steps,
        "exec_rsi_14": exec_rsi_14,
        "exec_macd_line": exec_macd_line,
        "exec_macd_signal": exec_macd_signal,
        "cot": cot,
    }


async def _run_apollo_backtest_with_polling(symbol: str) -> tuple[ApolloBacktestOutput, int]:
    """Run or poll Apollo backtest until it is available or timeout expires."""
    settings = get_settings()
    client = get_apollo_client()
    deadline = datetime.now(UTC).timestamp() + settings.apollo_train_timeout_seconds
    polls = 0

    while True:
        polls += 1
        try:
            result = await client.backtest(
                symbol=symbol,
                num_periods=settings.apollo_backtest_periods,
            )
            return result, polls
        except ApolloApiError as exc:
            if datetime.now(UTC).timestamp() >= deadline:
                raise ApolloApiError(
                    f"Apollo backtest não ficou disponível a tempo: {exc}",
                    status_code=exc.status_code,
                    payload=exc.payload,
                ) from exc
            await asyncio.sleep(settings.apollo_poll_interval_seconds)


@timed_async("Node: Forecast")
async def forecast_node(state: QuantAgentState) -> dict:
    """Apollo forecast node with safe training/backtest fallback."""
    settings = get_settings()
    client = get_apollo_client()
    window = build_prediction_window(
        lookback_days=settings.apollo_prediction_lookback_days,
    )

    warnings = list(state.forecast_warnings)
    training_attempts = 0
    backtest_error_pct: float | None = None
    prediction: ApolloPredictionOutput | None = None
    status = "ready"
    action_step = "apollo_predict"

    try:
        prediction = await client.predict(
            symbol=state.symbol,
            start_date=window.start_date,
            end_date=window.end_date,
        )
    except ApolloApiError as exc:
        if not exc.missing_model:
            warnings.append(f"forecast indisponível: {exc}")
            status = "unavailable"
        else:
            status = "training_required"
            action_step = "apollo_train"
            for attempt in range(1, settings.apollo_train_max_attempts + 1):
                try:
                    training_attempts = attempt
                    train_result = await client.train(
                        symbol=state.symbol,
                        lookback_days=settings.apollo_train_lookback_days,
                    )
                    warnings.append(
                        f"treino Apollo iniciado (tentativa {attempt}/{settings.apollo_train_max_attempts}): {train_result.status}"
                    )
                    backtest_result, _ = await _run_apollo_backtest_with_polling(state.symbol)
                    if len(backtest_result.results) < settings.apollo_backtest_periods:
                        warnings.append("backtest Apollo retornou menos períodos que o esperado")
                        continue
                    last_period = backtest_result.results[-1]
                    backtest_error_pct = calculate_error_pct(
                        last_period.current_price,
                        last_period.predicted_price,
                    )
                    if backtest_error_pct is None:
                        warnings.append("não foi possível calcular erro do quinto período do backtest")
                        continue
                    if backtest_error_pct <= settings.apollo_backtest_error_threshold_pct:
                        prediction = await client.predict(
                            symbol=state.symbol,
                            start_date=window.start_date,
                            end_date=window.end_date,
                        )
                        status = "trained_and_validated"
                        action_step = "apollo_backtest"
                        break
                    warnings.append(
                        "backtest Apollo reprovado no 5o período "
                        f"({backtest_error_pct:.2f}% > {settings.apollo_backtest_error_threshold_pct:.2f}%)"
                    )
                except ApolloApiError as train_exc:
                    warnings.append(f"falha Apollo na tentativa {attempt}: {train_exc}")
                    status = "training_error"
            if prediction is None:
                warnings.append("forecast Apollo descartado após limite de retreinos")
                status = "validation_failed"

    if prediction is None:
        return {
            **state.model_dump(),
            "messages": state.messages + [AIMessage(content="Apollo forecast indisponível ou não confiável.")],
            "next_action": NextAction.TREND_INTERPRET,
            "intermediate_steps_global": state.intermediate_steps_global + [(
                action_step,
                status if not warnings else " | ".join(warnings[-3:]),
            )],
            "forecast_period_start": window.start_date,
            "forecast_period_end": window.end_date,
            "forecast_training_attempts": training_attempts,
            "forecast_backtest_error_pct": backtest_error_pct,
            "forecast_status": status,
            "forecast_warnings": warnings,
            "forecast_actionable": False,
            "cot": "",
        }

    forecast_return_pct = calculate_return_pct(
        prediction.current_price,
        prediction.predicted_price,
    )
    model_mape = prediction.confidence_explanation.model_mape
    data_quality = prediction.confidence_explanation.data_quality
    quality = assess_forecast_quality(
        confidence=prediction.confidence,
        model_mape=model_mape,
        data_quality=data_quality,
        data_points=prediction.data_points,
        predicted_return_pct=forecast_return_pct,
        confidence_threshold=settings.apollo_confidence_threshold,
        mape_threshold=settings.apollo_mape_threshold,
    )
    warnings.extend(quality.warnings)
    if status == "ready":
        status = "ready_existing_model" if quality.actionable else "low_quality"
    elif status == "trained_and_validated" and not quality.actionable:
        status = "trained_but_low_quality"

    summary = (
        f"Apollo {prediction.direction} | conf={prediction.confidence:.1%} | "
        f"ret={(forecast_return_pct if forecast_return_pct is not None else 0.0):.2f}% | "
        f"mape={model_mape if model_mape is not None else -1:.2f}"
    )

    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=summary)],
        "next_action": NextAction.TREND_INTERPRET,
        "intermediate_steps_global": state.intermediate_steps_global + [(action_step, summary)],
        "forecast_period_start": prediction.period_start,
        "forecast_period_end": prediction.period_end,
        "forecast_data_points": prediction.data_points,
        "forecast_current_price": prediction.current_price,
        "forecast_predicted_price": prediction.predicted_price,
        "forecast_direction": prediction.direction,
        "forecast_confidence": prediction.confidence,
        "forecast_model_mape": model_mape,
        "forecast_period_volatility": prediction.confidence_explanation.period_volatility,
        "forecast_data_quality": data_quality,
        "forecast_return_pct": forecast_return_pct,
        "forecast_backtest_error_pct": backtest_error_pct,
        "forecast_training_attempts": training_attempts,
        "forecast_status": status,
        "forecast_actionable": quality.actionable,
        "forecast_warnings": warnings,
        "cot": "",
    }


@timed_async("Node: TrendInterpretation")
async def trend_interpret_node(state: QuantAgentState) -> dict:
    """
    Multi-timeframe interpretation node.
    Applies hierarchical rules to combine macro, setup, and execution layers.
    """
    print(f"[DEBUG] ===== TREND INTERPRETATION NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}")
    
    # Initialize interpreter
    interpreter = MultiTimeframeInterpreter()
    
    # Determine MACD bullish status for each timeframe
    macro_macd_bullish = state.macro_macd_line is not None and state.macro_macd_signal is not None and state.macro_macd_line > state.macro_macd_signal
    setup_macd_bullish = state.setup_macd_line is not None and state.setup_macd_signal is not None and state.setup_macd_line > state.setup_macd_signal
    exec_macd_bullish = state.exec_macd_line is not None and state.exec_macd_signal is not None and state.exec_macd_line > state.exec_macd_signal
    
    # Run multi-timeframe interpretation
    interpretation = interpreter.interpret_multi_tf_signal(
        # Macro (1D) - regime detection
        macro_rsi=state.macro_rsi_14,
        macro_macd_bullish=macro_macd_bullish,
        macro_bb_pct_b=state.macro_bb_pct_b,
        macro_regime=state.macro_regime,
        # Setup (4H) - signal generation
        setup_rsi=state.setup_rsi_14,
        setup_macd_bullish=setup_macd_bullish,
        setup_bb_pct_b=state.setup_bb_pct_b,
        # Execution (1H) - timing
        exec_rsi=state.exec_rsi_14,
        exec_macd_bullish=exec_macd_bullish,
        # Additional context
        volatility_annualized=state.volatility_annualized,
        price_change_pct=state.price_change_pct,
    )
    
    print(f"[DEBUG] Trend interpretation - signal_direction: {interpretation['signal_direction']}")
    print(f"[DEBUG] Trend interpretation - signal_type: {interpretation['signal_type']}")
    print(f"[DEBUG] Trend interpretation - trend_state: {interpretation['trend_state']}")
    print(f"[DEBUG] Trend interpretation - confidence: {interpretation['confidence']}")
    print(f"[DEBUG] Trend interpretation - reasoning: {interpretation['reasoning']}")
    print(f"[DEBUG] ===== TREND INTERPRETATION NODE END =====")
    
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=interpretation['reasoning'])],
        "next_action": NextAction.DECISION_ENGINE,  # Route to decision engine (FINAL authority)
        "intermediate_steps_global": state.intermediate_steps_global + [("trend_interpret", interpretation['reasoning'])],
        "trend_state": interpretation['trend_state'],
        "trend_direction": interpretation['signal_direction'],
        "signal_type": interpretation['signal_type'],
        "execution_timing": interpretation['execution_timing'],
        "signal_direction": interpretation['signal_direction'],  # Legacy field for compatibility
        "signal_confidence": interpretation['confidence'],  # Legacy field for compatibility
        "cot": interpretation['reasoning'],
    }


@timed_async("Node: DecisionEngine")
async def decision_engine_node(state: QuantAgentState) -> dict:
    """
    Decision Engine - Deterministic multi-timeframe decision layer.
    This node has FINAL authority over MoE output.
    """
    print(f"[DEBUG] ===== DECISION ENGINE NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}")
    
    # Initialize decision engine
    engine = DecisionEngine()
    
    # Determine macro trend from 1D data
    if state.macro_bias == "bullish_stretched":
        macro_trend = MacroTrend.BULLISH
    elif state.macro_bias == "bearish_stretched":
        macro_trend = MacroTrend.BEARISH
    else:
        macro_trend = MacroTrend.NEUTRAL
    
    # Determine setup MACD bullish status
    setup_macd_bullish = (
        state.setup_macd_line is not None and 
        state.setup_macd_signal is not None and 
        state.setup_macd_line > state.setup_macd_signal
    )
    
    # Determine execution MACD bullish status
    exec_macd_bullish = (
        state.exec_macd_line is not None and 
        state.exec_macd_signal is not None and 
        state.exec_macd_line > state.exec_macd_signal
    )
    
    # Run decision engine with hierarchical rules
    decision = engine.decide(
        macro_trend=macro_trend,
        macro_rsi=state.macro_rsi_14,
        macro_bb_pct_b=state.macro_bb_pct_b,
        setup_rsi=state.setup_rsi_14,
        setup_macd_bullish=setup_macd_bullish,
        exec_rsi=state.exec_rsi_14,
        exec_macd_bullish=exec_macd_bullish,
        volatility_annualized=state.volatility_annualized,
        price_change_pct=state.price_change_pct,
        moe_signal=state.moe_final_signal,  # MoE as auxiliary input
        moe_confidence=state.moe_final_confidence,
    )

    forecast_quality = assess_forecast_quality(
        confidence=state.forecast_confidence,
        model_mape=state.forecast_model_mape,
        data_quality=state.forecast_data_quality,
        data_points=state.forecast_data_points,
        predicted_return_pct=state.forecast_return_pct,
        confidence_threshold=get_settings().apollo_confidence_threshold,
        mape_threshold=get_settings().apollo_mape_threshold,
    )
    overlay_signal, overlay_confidence, overlay_reason = overlay_forecast_on_signal(
        signal=decision.signal.value,
        confidence=decision.confidence,
        forecast_quality=forecast_quality,
        predicted_return_pct=state.forecast_return_pct,
    )
    if overlay_signal != decision.signal.value or overlay_confidence != decision.confidence:
        decision.signal = FinalSignal(overlay_signal)
        decision.confidence = overlay_confidence
        decision.reasoning = f"{decision.reasoning} | Overlay ML: {overlay_reason}"
    elif state.forecast_confidence is not None:
        decision.reasoning = f"{decision.reasoning} | Overlay ML: {overlay_reason}"
    
    print(f"[DEBUG] Decision Engine - final_signal: {decision.signal.value}")
    print(f"[DEBUG] Decision Engine - signal_type: {decision.signal_type.value}")
    print(f"[DEBUG] Decision Engine - confidence: {decision.confidence}")
    print(f"[DEBUG] Decision Engine - reasoning: {decision.reasoning}")
    print(f"[DEBUG] Decision Engine - execution_timing: {decision.execution_timing}")
    print(f"[DEBUG] ===== DECISION ENGINE NODE END =====")
    
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=decision.reasoning)],
        "next_action": NextAction.RISK,  # Route to risk after decision
        "intermediate_steps_global": state.intermediate_steps_global + [("decision_engine", decision.reasoning)],
        "final_signal": decision.signal.value,
        "final_signal_type": decision.signal_type.value,
        "final_confidence": decision.confidence,
        "final_reasoning": decision.reasoning,
        # Override legacy fields with final decision
        "signal_direction": decision.signal.value,
        "signal_confidence": decision.confidence,
        "cot": decision.reasoning,
    }


@timed_async("Node: FeatureEngineering")
async def feature_engineering_node(state: QuantAgentState) -> dict:
    print(f"[DEBUG] ===== FEATURE ENGINEERING NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}, live_price: {state.live_price}")
    analysis_interval = _get_analysis_interval(state.timeframe)
    context = (
        f"Symbol: {state.symbol} | Timeframe: {state.timeframe} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{analysis_interval}\" — use ESTE intervalo em TODAS as ferramentas | "
        f"Preço atual: {state.live_price} | "
        f"Anomalias: {state.anomalies_detected}"
    )
    print(f"[DEBUG] Context: {context}")
    node_config = NODE_CONFIG["feature_engineering"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("feature_engineering"), _FEATURE_SYSTEM, context,
        force_interval=analysis_interval, node_name="FeatureEngineering", enable_cot=node_config["cot"]
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

    # Track intervals used for consistency check
    feature_intervals = {}

    for tool_name, result in steps:
        print(f"[DEBUG] Processing feature tool: {tool_name}, result preview: {result[:100] if result else 'None'}")
        try:
            if "error" in result:
                continue
            if tool_name == "get_feature_rsi":
                parsed = RSIOutput.model_validate_json(result)
                rsi_14 = parsed.rsi_14
                feature_intervals["rsi"] = parsed.interval
            elif tool_name == "get_feature_macd":
                parsed = MACDOutput.model_validate_json(result)
                macd_line = parsed.macd_line
                macd_signal = parsed.signal
                macd_histogram = parsed.histogram
                feature_intervals["macd"] = parsed.interval
            elif tool_name == "get_feature_bollinger":
                parsed = BollingerBandsOutput.model_validate_json(result)
                bb_upper = parsed.upper
                bb_middle = parsed.middle
                bb_lower = parsed.lower
                feature_intervals["bollinger"] = parsed.interval
        except (json.JSONDecodeError, KeyError, TypeError, Exception):
            # Strict parsing: ignore invalid data (institutional rule)
            continue

    # Timeframe consistency check
    if feature_intervals:
        unique_intervals = set(feature_intervals.values())
        if len(unique_intervals) > 1:
            print(f"[DEBUG] WARNING: Inconsistent intervals detected: {feature_intervals}")
            # Force recalculation on next run by clearing state
            rsi_14 = None
            macd_line = None
            bb_upper = None

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
        "cot":                cot,
    }


@timed_async("Node: ForecastQuestion")
async def forecast_question_node(state: QuantAgentState) -> dict:
    """Responde perguntas simples sobre previsões futuras do Apollo."""
    if not state.forecast_confidence and state.forecast_status not in ["ready_existing_model", "trained_and_validated"]:
        fallback = f"A previsão do Apollo não está disponível: {state.forecast_status}. "
        fallback += "Verifique a disponibilidade do serviço Apollo e tente novamente."
        return {
            **state.model_dump(),
            "final_answer": fallback,
        }

    settings = get_settings()
    forecast_summary = f"""
PREVISÃO APOLLO:
- Símbolo: {state.symbol}
- Período: {state.forecast_period_start} até {state.forecast_period_end}
- Direção: {state.forecast_direction}
- Preço Atual: ${state.forecast_current_price:.2f if state.forecast_current_price else 'N/A'}
- Preço Previsto: ${state.forecast_predicted_price:.2f if state.forecast_predicted_price else 'N/A'}
- Retorno Esperado: {state.forecast_return_pct:.2f if state.forecast_return_pct is not None else 'N/A'}%
- Confiança: {state.forecast_confidence:.1%}
- MAPE do Modelo: {state.forecast_model_mape:.2f if state.forecast_model_mape is not None else 'N/A'}%
- Qualidade dos Dados: {state.forecast_data_quality or 'N/A'}
- Volatilidade: {state.forecast_period_volatility:.2f if state.forecast_period_volatility else 'N/A'}%
- Acionável: {state.forecast_actionable}
- Data Points: {state.forecast_data_points}
- Status: {state.forecast_status}
- Avisos: {'; '.join(state.forecast_warnings) if state.forecast_warnings else 'Nenhum'}

CRITÉRIOS DE QUALIDADE:
- Threshold Confiança: >= {settings.apollo_confidence_threshold:.0%}
- Threshold MAPE: <= {settings.apollo_mape_threshold:.2f}%
"""

    user_question = state.messages[-1].content if state.messages else ""
    llm = _make_llm(model="gpt-4o-mini", temperature=0.1)

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": _FORECAST_RESPONSE_SYSTEM},
            {"role": "user", "content": f"{user_question}\n\n{forecast_summary}"}
        ])
        final_answer = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        final_answer = f"Erro ao processar previsão: {str(e)}"

    return {
        **state.model_dump(),
        "final_answer": final_answer,
        "messages": state.messages + [AIMessage(content=final_answer)],
    }

@timed_async("Node: RiskAgent")
async def risk_agent_node(state: QuantAgentState) -> dict:
    """
    Risk agent: calculates risk metrics (CVaR, Sharpe, drawdown, volatility).
    Uses execution layer (1H) for risk metrics - multi-timeframe consistency.
    """
    print(f"[DEBUG] ===== RISK AGENT NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}")
    
    # Use 1H interval for risk metrics (execution layer)
    exec_interval = _get_multi_tf_interval(TimeframeLayer.EXECUTION)
    context = (
        f"Symbol: {state.symbol} | Preço: {state.live_price} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{exec_interval}\" — 1H timeframe for risk metrics (execution layer) | "
        f"Vol 24h: {state.volume_24h} | Variação: {state.price_change_pct}%\n"
        f"DADOS JÁ COLETADOS: RSI={state.exec_rsi_14}\n"
        f"SUA TAREFA: Calcular métricas de risco usando as ferramentas disponíveis."
    )
    print(f"[DEBUG] Context: {context}")
    node_config = NODE_CONFIG["risk_agent"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("risk_agent"), _RISK_SYSTEM, context, clear_steps=True,
        force_interval=exec_interval, node_name="RiskAgent", enable_cot=node_config["cot"]
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

    # Track risk interval for consistency check
    risk_interval = None

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
                volatility_raw = parsed.volatility_20d if parsed.volatility_20d is not None else parsed.volatility_21
                volatility_interval = parsed.interval
                risk_interval = parsed.interval
                print(f"[DEBUG] Parsed calculate_risk: cvar={cvar_95}, sharpe={sharpe}, drawdown={max_drawdown}, interval={risk_interval}")
            elif tool_name == "get_feature_cvar":
                parsed = CVaROutput.model_validate_json(result)
                cvar_95 = parsed.cvar_95
                risk_interval = parsed.interval
                print(f"[DEBUG] Parsed get_feature_cvar: cvar={cvar_95}, interval={risk_interval}")
            elif tool_name == "get_feature_sharpe":
                parsed = SharpeOutput.model_validate_json(result)
                sharpe = parsed.sharpe
                risk_interval = parsed.interval
                print(f"[DEBUG] Parsed get_feature_sharpe: sharpe={sharpe}, interval={risk_interval}")
            elif tool_name == "get_feature_max_drawdown":
                parsed = MaxDrawdownOutput.model_validate_json(result)
                max_drawdown = parsed.max_drawdown
                risk_interval = parsed.interval
                print(f"[DEBUG] Parsed get_feature_max_drawdown: drawdown={max_drawdown}, interval={risk_interval}")
            elif tool_name == "get_feature_volatility":
                parsed = VolatilityOutput.model_validate_json(result)
                volatility_raw = parsed.volatility_raw
                volatility_annualized = parsed.volatility_annualized
                volatility_interval = parsed.data_interval
                risk_interval = parsed.data_interval
                print(f"[DEBUG] Parsed get_feature_volatility: vol_raw={volatility_raw}, vol_ann={volatility_annualized}, interval={risk_interval}")
        except (json.JSONDecodeError, KeyError, TypeError, Exception) as e:
            # Strict parsing: ignore invalid data (institutional rule)
            print(f"[DEBUG] Exception parsing {tool_name}: {type(e).__name__}: {e}")
            continue

    # Timeframe consistency check: risk metrics must match execution layer (1H)
    expected_interval = _get_multi_tf_interval(TimeframeLayer.EXECUTION)
    if risk_interval and risk_interval != expected_interval:
        print(f"[DEBUG] WARNING: Risk interval {risk_interval} != expected {expected_interval}")
        print(f"[DEBUG] Clearing risk metrics to force recalculation with correct interval")
        cvar_95 = None
        sharpe = None
        max_drawdown = None
        volatility_annualized = None

    print(f"[DEBUG] Risk metrics before calculation - cvar: {cvar_95}, sharpe: {sharpe}, drawdown: {max_drawdown}, vol: {volatility_annualized}")

    # Store execution layer metrics separately for multi-timeframe consistency
    exec_sharpe = sharpe
    exec_volatility = volatility_annualized

    # ─── SANITY AUDIT — proteção contra valores implausíveis do backend ───
    risk_audit_anomalies: list[str] = []

    # A) max_drawdown bounds: deve estar em [0, 1]
    if max_drawdown is not None:
        if max_drawdown < 0 or max_drawdown > 1:
            risk_audit_anomalies.append(
                f"max_drawdown fora do range [0,1]: {max_drawdown:.4f} — descartado"
            )
            max_drawdown = None
        elif state.recent_high and state.recent_low and state.recent_high > 0:
            # Cross-check: drawdown do backend não pode exceder o range observado
            # * 1.5 (tolerância para janela maior do que nossos 100 candles).
            observed_range_dd = (state.recent_high - state.recent_low) / state.recent_high
            if max_drawdown > observed_range_dd * 1.5 and max_drawdown > 0.10:
                risk_audit_anomalies.append(
                    f"max_drawdown backend ({max_drawdown:.2%}) >> range observado "
                    f"({observed_range_dd:.2%}) × 1.5 — usando fallback observado"
                )
                max_drawdown = observed_range_dd

    # B) cvar_95: por convenção é uma perda (fração negativa ou próxima de 0 negativa)
    if cvar_95 is not None and (cvar_95 < -1.0 or cvar_95 > 1.0):
        risk_audit_anomalies.append(
            f"cvar_95 fora do range [-1,1]: {cvar_95:.4f} — descartado"
        )
        cvar_95 = None

    # C) sharpe: na prática financeira |sharpe| > 5 é extremamente suspeito
    if sharpe is not None and abs(sharpe) > 10:
        risk_audit_anomalies.append(
            f"sharpe implausível: {sharpe:.4f} — descartado"
        )
        sharpe = None

    # D) volatility_annualized: > 500% a.a. é essencialmente quebra de dados
    if volatility_annualized is not None and volatility_annualized > 5.0:
        risk_audit_anomalies.append(
            f"volatility_annualized implausível: {volatility_annualized:.2%} — descartado"
        )
        volatility_annualized = None

    if risk_audit_anomalies:
        print(f"[DEBUG] Risk audit caught anomalies: {risk_audit_anomalies}")

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
            print(f"[DEBUG] Risk agent output - risk_level: {risk_level}")
    print(f"[DEBUG] ===== RISK AGENT NODE END =====")
    return {
        **state.model_dump(),
        "messages":           state.messages + [AIMessage(content=content)],
        "next_action":        NextAction.SIGNAL,
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "intermediate_steps_agent": steps,
        "var_95":             var_95,
        "cvar_95":            cvar_95,
        "sharpe":             sharpe,
        "max_drawdown":       max_drawdown,
        "volatility_annualized": volatility_annualized,
        "risk_level":         risk_level,
        "exec_sharpe":        exec_sharpe,      # Execution layer (1H) sharpe
        "exec_volatility":    exec_volatility,  # Execution layer (1H) volatility
        "anomalies_detected": state.anomalies_detected + risk_audit_anomalies,
        "cot":                cot,
    }

@timed_async("Node: SignalAgent")
async def signal_agent_node(state: QuantAgentState) -> dict:
    print(f"[DEBUG] ===== SIGNAL AGENT NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}")
    
    # Use 4H interval for signal generation (setup layer)
    setup_interval = _get_multi_tf_interval(TimeframeLayer.SETUP)
    context = (
        f"Symbol: {state.symbol} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{setup_interval}\" — 4H timeframe for signal generation (setup layer) | "
        f"REGIME MACRO (1D): {state.macro_regime} | BIAS: {state.macro_bias}\n"
        f"SETUP DATA (4H): RSI={state.setup_rsi_14}, MACD={state.setup_macd_line}\n"
        f"SUA TAREFA: Gerar sinal de trading usando dados do timeframe 4H, respeitando o regime macro."
    )
    print(f"[DEBUG] Context: {context}")
    node_config = NODE_CONFIG["signal_agent"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("signal_agent"), _SIGNAL_SYSTEM, context,
        force_interval=setup_interval, node_name="SignalAgent", enable_cot=node_config["cot"]
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

    # QUANTITATIVE ENGINE: Deterministic signal calculation using setup layer (4H)
    # Respect macro regime bias from 1D
    print(f"[DEBUG] Input for quant engine (setup 4H) - rsi: {state.setup_rsi_14}, macd: {state.setup_macd_line}")
    print(f"[DEBUG] Macro regime (1D) - regime: {state.macro_regime}, bias: {state.macro_bias}")
    
    signal_output = compute_signal_score(
        rsi_14=state.setup_rsi_14,
        macd_line=state.setup_macd_line,
        macd_signal=None,  # Not available in setup layer
        bb_upper=state.setup_bb_upper,
        bb_lower=state.setup_bb_lower,
        live_price=state.live_price,
        price_change_pct=state.price_change_pct,
        volatility_annualized=state.volatility_annualized,
        sharpe=state.exec_sharpe,  # Use execution layer sharpe
    )
    
    # Apply macro regime filter: if macro is overbought + bullish, don't allow short signals
    if state.macro_regime == "overbought" and state.macro_bias == "bullish_stretched":
        if signal_output.direction in ["short", "weak_short"]:
            signal_output.direction = "neutral"
            signal_output.confidence = signal_output.confidence * 0.5  # Reduce confidence for neutral

    # Override regime with quantitative engine result
    regime = signal_output.regime
    direction = signal_output.direction
    signal_confidence = signal_output.confidence
    print(f"[DEBUG] Quant engine output - regime: {regime}, direction: {direction}, confidence: {signal_confidence}")

    print(f"[DEBUG] ===== SIGNAL AGENT NODE END =====")
    return {
        **state.model_dump(),
        "messages":           state.messages + [AIMessage(content=content)],
        "next_action":        NextAction.MOE,
        "signal_direction":   direction,
        "regime":             regime,
        "signal_confidence":  signal_confidence,
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "intermediate_steps_agent": steps,
        "cot":                cot,
    }

@timed_async("Node: MoE")
async def moe_node(state: QuantAgentState) -> dict:
    """
    Mixture of Experts node for signal generation.
    Combines multiple signal experts with gating network and risk layer.
    Multi-timeframe: Uses setup layer (4H) for signal, respects macro regime (1D), uses execution risk (1H).
    """
    print(f"[DEBUG] ===== MOE NODE START =====")
    print(f"[DEBUG] Input symbol: {state.symbol}")
    print(f"[DEBUG] Macro regime (1D): {state.macro_regime}, bias: {state.macro_bias}")
    print(f"[DEBUG] Setup data (4H): RSI={state.setup_rsi_14}, MACD={state.setup_macd_line}")
    print(f"[DEBUG] Execution risk (1H): Sharpe={state.exec_sharpe}, Vol={state.exec_volatility}")
    
    # Initialize MoE layer
    moe_layer = MoESignalLayer(gating_mode="rule_based", enable_risk_layer=True)
    
    # Compute MoE signal using multi-timeframe inputs
    # Use trend_state from hierarchical interpretation instead of simple regime
    moe_output = moe_layer.compute_signal(
        rsi_14=state.setup_rsi_14,        # Use 4H RSI for signal
        macd_line=state.setup_macd_line,   # Use 4H MACD for signal
        macd_signal=state.setup_macd_signal,
        bb_upper=state.setup_bb_upper,     # Use 4H BB for signal
        bb_lower=state.setup_bb_lower,
        live_price=state.live_price,
        price_change_pct=state.price_change_pct,
        volatility_annualized=state.exec_volatility,  # Use 1H volatility for risk
        trend_state=state.trend_state or "neutral",   # Use hierarchical trend state for gating
        sharpe=state.exec_sharpe,                       # Use 1H sharpe for risk layer
    )
    
    print(f"[DEBUG] MoE output - signal: {moe_output.final_signal:.3f}, confidence: {moe_output.final_confidence:.3f}")
    print(f"[DEBUG] MoE position size: {moe_output.position_size:.3f}, risk-adjusted: {moe_output.risk_adjusted_signal:.3f}")
    print(f"[DEBUG] MoE experts: {[e.value for e in moe_output.selected_experts]}")
    print(f"[DEBUG] ===== MOE NODE END =====")
    
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=f"MoE signal: {moe_output.final_signal:.3f}, confidence: {moe_output.final_confidence:.3f}")],
        "next_action": NextAction.RISK_GATE,
        "moe_final_signal": moe_output.final_signal,
        "moe_final_confidence": moe_output.final_confidence,
        "moe_selected_experts": [e.value for e in moe_output.selected_experts],
        "moe_expert_weights": {k.value: v for k, v in moe_output.expert_weights.items()},
        "moe_gating_reason": moe_output.gating_reason,
        "moe_position_size": moe_output.position_size,
        "moe_risk_adjusted_signal": moe_output.risk_adjusted_signal,
        "intermediate_steps_global": state.intermediate_steps_global + [("moe", moe_output.gating_reason)],
        "intermediate_steps_agent": [],
        "cot": "",
    }

@timed("Node: RiskGate")
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

    # Regra 9: Risk-signal alignment check
    # Block Long/Short signals when Sharpe is negative (indicates poor risk-adjusted performance)
    if state.signal_direction in ["long", "short"] and state.sharpe is not None and state.sharpe < 0:
        blocked = True
        reasons.append(f"Signal {state.signal_direction} com Sharpe negativo ({state.sharpe:.2f}) — conflito entre sinal e performance ajustada ao risco.")

    # Regra 10: Risk-confidence consistency — HIGH risk exige confiança mínima
    if state.risk_level == RiskLevel.HIGH and (state.signal_confidence or 0) < 0.40 \
            and state.signal_direction in ["long", "short"]:
        blocked = True
        reasons.append(
            f"Risk level HIGH com confiança {(state.signal_confidence or 0):.0%} — "
            f"threshold mínimo para HIGH = 40%."
        )

    # Regra 11: Data quality minimum — precisamos de pelo menos 2 indicadores técnicos
    indicators_available = sum(
        1 for x in [state.rsi_14, state.macd_line, state.bb_upper] if x is not None
    )
    if state.signal_direction in ["long", "short"] and indicators_available < 2:
        warnings.append(
            f"Apenas {indicators_available} indicador(es) técnico(s) disponível(is) — sinal frágil."
        )

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
        "cot":          "",
    }

def _fmt(val, *, pct: bool = False, unit: str = "", prec: int = 4) -> str:
    """Renderização explícita: None → '(não coletado)', números formatados."""
    if val is None:
        return "(não coletado)"
    if isinstance(val, (int, float)):
        if pct:
            return f"{val * 100:.2f}%"
        return f"{val:.{prec}f}{unit}"
    return str(val)


def _coverage_summary(state: QuantAgentState) -> tuple[int, int, list[str]]:
    """Conta quantos campos críticos estão preenchidos; retorna (ok, total, missing)."""
    fields = {
        "live_price": state.live_price,
        "price_change_pct": state.price_change_pct,
        "rsi_14": state.rsi_14,
        "macd_line": state.macd_line,
        "bb_upper": state.bb_upper,
        "sharpe": state.sharpe,
        "cvar_95": state.cvar_95,
        "max_drawdown": state.max_drawdown,
        "volatility_annualized": state.volatility_annualized,
    }
    missing = [k for k, v in fields.items() if v is None]
    return len(fields) - len(missing), len(fields), missing


@timed_async("Node: Execution")
async def execution_node(state: QuantAgentState) -> dict:
    """
    Consolida toda a análise e gera resposta final institucional.
    Hardened:
      • valida gate + campos críticos
      • renderiza None como '(não coletado)' explicitamente
      • inclui resumo de cobertura para evitar alucinação
    """
    # VALIDAÇÃO FINAL OBRIGATÓRIA
    if not state.gate_approved:
        raise Exception(f"Execution node chamado sem aprovação do risk gate. Motivo: {state.gate_reason}")
    if not state.symbol:
        raise Exception("Symbol não definido no estado.")
    if state.signal_direction is None:
        raise Exception("Signal direction não definido no estado.")
    if state.risk_level is None:
        raise Exception("Risk level não definido no estado.")

    ok, total, missing = _coverage_summary(state)
    coverage_line = f"Cobertura de dados: {ok}/{total} campos preenchidos"
    if missing:
        coverage_line += f" | AUSENTES: {', '.join(missing)}"

    # Formatação explícita evita o LLM confundir 0.0 com None
    context = f"""
RESUMO DE COBERTURA
  {coverage_line}

DADOS DE MERCADO
  Symbol:          {state.symbol}
  Timeframe:       {state.timeframe}
  Preço atual:     {_fmt(state.live_price, prec=2)}
  Variação 24h:    {_fmt(state.price_change_pct, prec=4)}%
  Volume 24h:      {_fmt(state.volume_24h, prec=2)}
  Range recente:   high={_fmt(state.recent_high, prec=2)} low={_fmt(state.recent_low, prec=2)}

INDICADORES TÉCNICOS
  RSI (14):        {_fmt(state.rsi_14, prec=2)}
  MACD Line:       {_fmt(state.macd_line)}
  MACD Signal:     {_fmt(state.macd_signal)}
  MACD Histogram:  {_fmt(state.macd_histogram)}
  Bollinger Upper: {_fmt(state.bb_upper, prec=2)}
  Bollinger Middle:{_fmt(state.bb_middle, prec=2)}
  Bollinger Lower: {_fmt(state.bb_lower, prec=2)}

MÉTRICAS DE RISCO
  Risk level:      {state.risk_level}
  VaR 95%:         {_fmt(state.var_95, pct=True)}
  CVaR 95%:        {_fmt(state.cvar_95, pct=True)}
  Sharpe:          {_fmt(state.sharpe, prec=3)}
  Max Drawdown:    {_fmt(state.max_drawdown, pct=True)}
  Vol. anualizada: {_fmt(state.volatility_annualized, pct=True)}

SINAL
  Regime:          {state.regime}
  Direção:         {state.signal_direction}
  Confiança:       {_fmt(state.signal_confidence, pct=True)}

QUALIDADE DOS DADOS
  Anomalias:       {state.anomalies_detected or 'Nenhuma'}

MODELO PREDITIVO (APOLLO)
  Status:          {state.forecast_status}
  Janela:          {state.forecast_period_start or '(não coletado)'} -> {state.forecast_period_end or '(não coletado)'}
  Data points:     {state.forecast_data_points if state.forecast_data_points is not None else '(não coletado)'}
  Current price:   {_fmt(state.forecast_current_price, prec=2)}
  Predicted price: {_fmt(state.forecast_predicted_price, prec=2)}
  Direção ML:      {state.forecast_direction or '(não coletado)'}
  Forecast ret.:   {_fmt(state.forecast_return_pct, prec=2)}%
  Confiança ML:    {_fmt(state.forecast_confidence, pct=True)}
  MAPE modelo:     {_fmt(state.forecast_model_mape, prec=2)}%
  Qualidade dados: {state.forecast_data_quality or '(não coletado)'}
  Acionável:       {state.forecast_actionable}
  Backtest p5 err: {_fmt(state.forecast_backtest_error_pct, prec=2)}%
  Avisos ML:       {state.forecast_warnings or 'Nenhum'}

RISK GATE
  Aprovado:        {state.gate_approved}
  Motivo:          {state.gate_reason}

HISTÓRICO DE ANÁLISE (últimos 8 steps):
{chr(10).join(f'  [{k}] {v[:120]}' for k, v in state.intermediate_steps_global[-8:])}
""".strip()

    msgs = [
        SystemMessage(content=_EXECUTION_SYSTEM),
        HumanMessage(content=context),
    ]
    node_config = NODE_CONFIG["execution"]
    llm = _get_llm_for_node("execution")
    
    # Add CoT instruction if enabled
    if node_config["cot"]:
        msgs.append(HumanMessage(content="\n\nFORMATO DE RESPOSTA OBRIGATÓRIO:\n<thought>Resumo CONCISO do seu raciocínio (max 2-3 frases). Ex: 'Obtive o indicador X que está em Y, indicando Z.'</thought>\n<answer>Sua resposta final aqui</answer>"))
    
    final_response = await llm.ainvoke(msgs)
    
    # Track execution LLM call
    cost_tracker = get_cost_tracker()
    if hasattr(final_response, 'response_metadata'):
        metadata = final_response.response_metadata
        input_tokens = metadata.get('token_usage', {}).get('prompt_tokens', 0)
        output_tokens = metadata.get('token_usage', {}).get('completion_tokens', 0)
        model_name = getattr(llm, 'model_name', 'unknown') or getattr(llm, 'model', 'unknown')
        if input_tokens > 0 or output_tokens > 0:
            cost_tracker.add_call(model_name, input_tokens, output_tokens, "Execution")
    
    # Get cost summary
    cost_summary = cost_tracker.get_summary()
    cost_info = f"""

💰 CUSTO LLM
  Total de chamadas: {cost_summary['total_calls']}
  Tokens de entrada: {cost_summary['total_input_tokens']}
  Tokens de saída: {cost_summary['total_output_tokens']}
  Total de tokens: {cost_summary['total_tokens']}
  Custo total: ${cost_summary['total_cost_usd']:.6f} USD
"""
    
    # Append cost info to final response
    final_answer_with_cost = final_response.content + cost_info
    
    # Extract CoT if enabled
    cot, answer = _extract_cot_and_answer(final_response.content) if node_config["cot"] else ("", final_response.content)
    final_answer_with_cost = answer + cost_info

    return {
        **state.model_dump(),
        "messages":     state.messages + [final_response],
        "next_action":  NextAction.FINALIZE,
        "final_answer": final_answer_with_cost,
        "cost_summary": cost_summary,
        "cot":          cot,
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
    workflow.add_node("features_macro", features_macro_node)    # 1D - regime detection
    workflow.add_node("features_setup", features_setup_node)    # 4H - signal generation
    workflow.add_node("features_exec",  features_exec_node)     # 1H - execution timing
    workflow.add_node("forecast",       forecast_node)          # Apollo ML forecast
    workflow.add_node("trend_interpret", trend_interpret_node)  # Multi-timeframe hierarchical interpretation
    workflow.add_node("decision_engine", decision_engine_node)  # Deterministic decision layer (FINAL authority)
    workflow.add_node("features",       feature_engineering_node)  # Legacy (will be phased out)
    workflow.add_node("risk",           risk_agent_node)
    workflow.add_node("signal",         signal_agent_node)
    workflow.add_node("moe",           moe_node)  # Auxiliary input to decision engine
    workflow.add_node("risk_gate",      pre_trade_risk_gate_node)
    workflow.add_node("execution",      execution_node)
    workflow.add_node("blocked",        blocked_node)
    workflow.add_node("forecast_question", forecast_question_node)  # Simple forecast Q&A
    workflow.add_node("finalize",       finalize_node)

    # Entry
    workflow.set_entry_point("orchestrator")

    # Fixed edges (multi-timeframe pipeline with decision engine as FINAL authority)
    workflow.add_edge("orchestrator", "market_data")
    workflow.add_edge("market_data",  "features_macro")   # 1D for regime
    workflow.add_edge("features_macro", "features_setup")  # 4H for signal
    workflow.add_edge("features_setup", "features_exec")   # 1H for execution
    workflow.add_edge("features_exec",  "forecast")        # Apollo forecast

    # Conditional: forecast → question answering or full analysis
    def route_after_forecast(state: QuantAgentState) -> str:
        msg = (state.messages[-1].content if state.messages else "").lower()
        simple_forecast_keywords = [
            "como vai", "como estará", "prevê", "forecast", "quando", "próximo",
            "amanhã", "tomorrow", "predict", "previsão", "vai estar", "estará",
            "qual é a previsão", "preço do", "preço para", "valor para"
        ]
        # Se é pergunta simples sobre forecast, roteia direto (mesmo sem confiança)
        is_simple_question = any(kw in msg for kw in simple_forecast_keywords)
        if is_simple_question:
            return "forecast_question"
        return "trend_interpret"

    workflow.add_conditional_edges(
        "forecast",
        route_after_forecast,
        {
            "forecast_question": "forecast_question",
            "trend_interpret": "trend_interpret",
        },
    )

    workflow.add_edge("trend_interpret", "decision_engine") # Decision engine (FINAL authority)
    workflow.add_edge("decision_engine", "risk")          # Risk uses final decision
    workflow.add_edge("risk",         "risk_gate")        # Risk gate uses final decision (bypass signal/moe)

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
    workflow.add_edge("forecast_question", "finalize")
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
