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
import re
import unicodedata
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

class AssetState(BaseModel):
    """
    Estado per-ativo. Encapsula TODOS os campos analíticos de um único símbolo
    (preço, indicadores multi-TF, risco, sinal, decisão, MoE, forecast).

    O `QuantAgentState` mantém um `assets: dict[str, AssetState]` para suportar
    análises comparativas entre múltiplos ativos. O ativo "primário" é o foco
    da decisão direcional; os demais são contexto comparativo.
    """
    symbol:             str                = ""

    # Market context (populated by MarketDataAgent)
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

    # Technical indicators (populated by FeatureAgent — legacy)
    rsi_14:             float | None       = None
    macd_line:          float | None       = None
    macd_signal:        float | None       = None
    macd_histogram:     float | None       = None
    bb_upper:           float | None       = None
    bb_middle:          float | None       = None
    bb_lower:           float | None       = None

    # Multi-timeframe — Macro layer (1D) - regime detection
    macro_rsi_14:       float | None       = None
    macro_macd_line:    float | None       = None
    macro_macd_signal:  float | None       = None
    macro_bb_upper:     float | None       = None
    macro_bb_lower:     float | None       = None
    macro_bb_pct_b:     float | None       = None
    macro_regime:       str | None         = None
    macro_bias:         str | None         = None

    # Trend state context (hierarchical interpretation)
    trend_state:        str | None         = None
    trend_direction:    str | None         = None
    signal_type:        str | None         = None
    execution_timing:   str | None         = None

    # Final decision (from Decision Engine - FINAL authority)
    final_signal:       str | None         = None
    final_signal_type:  str | None         = None
    final_confidence:   float | None       = None
    final_reasoning:    str | None         = None

    # Setup layer (4H) - signal generation
    setup_rsi_14:       float | None       = None
    setup_macd_line:    float | None       = None
    setup_macd_signal:  float | None       = None
    setup_bb_upper:     float | None       = None
    setup_bb_lower:     float | None       = None
    setup_bb_pct_b:     float | None       = None
    setup_signal:       float | None       = None
    setup_confidence:   float | None       = None
    setup_direction:    str | None         = None

    # Execution layer (1H) - timing + risk
    exec_rsi_14:        float | None       = None
    exec_macd_line:     float | None       = None
    exec_macd_signal:   float | None       = None
    exec_sharpe:        float | None       = None
    exec_volatility:    float | None       = None
    exec_timing:        str | None         = None

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
    regime:             str | None         = None
    signal_direction:   str | None         = None
    signal_confidence:  float | None       = None

    # MoE output (populated by MoE node — primary only in practice)
    moe_final_signal:       float | None   = None
    moe_final_confidence:   float | None   = None
    moe_selected_experts:   list[str]     = Field(default_factory=list)
    moe_expert_weights:     dict[str, float] = Field(default_factory=dict)
    moe_gating_reason:      str            = ""
    moe_position_size:      float | None   = None
    moe_risk_adjusted_signal: float | None = None

    # Data quality
    anomalies_detected: list[str]          = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class QuantAgentState(BaseModel):
    # Conversation
    messages:           list[Any]          = Field(default_factory=list)
    user_id:            str                = ""

    # Routing
    next_action:        NextAction         = NextAction.MARKET_DATA
    intermediate_steps_global: list[tuple[str, str]] = Field(default_factory=list)  # Global audit trail
    intermediate_steps_agent: list[tuple[str, str]] = Field(default_factory=list)  # Agent-specific steps
    final_answer:       str                = ""

    # Multi-asset support
    # `symbols[0]` is the primary asset (target of directional decision).
    # Others are comparative context.
    symbols:            list[str]          = Field(default_factory=list)
    primary_symbol:     str                = ""
    assets:             dict[str, AssetState] = Field(default_factory=dict)

    # Shared analysis timeframe (applies to all assets)
    timeframe:          AnalysisTimeframe  = AnalysisTimeframe.DAILY

    # Risk gate output (refers to primary asset — gate runs only on primary)
    gate_approved:      bool | None        = None
    gate_reason:        str                = ""

    # Portfolio context (cross-asset, institutional portfolio management)
    portfolio_state: PortfolioState       = Field(default_factory=PortfolioState)
    proposed_position_size: float | None   = None

    # Tool call deduplication cache (per run) — keyed by '<name>|<sorted_args>'
    tool_cache:         dict[str, str]     = Field(default_factory=dict)

    # Chain-of-thought (per node, for streaming)
    cot:                str                = ""
    # Reasoning trail: cumulative <thought> de cada nó, alimenta nós downstream
    reasoning_trail:    list[tuple[str, str]] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ─── Convenience accessors (backward-compat reads) ───

    @property
    def symbol(self) -> str:
        """Primary symbol. Backward-compat alias for `primary_symbol`."""
        return self.primary_symbol or (self.symbols[0] if self.symbols else "")

    @property
    def primary(self) -> AssetState:
        """Primary asset state. Returns an empty `AssetState` when missing."""
        sym = self.symbol
        if not sym:
            return AssetState()
        return self.assets.get(sym) or AssetState(symbol=sym)

    def get_asset(self, symbol: str) -> AssetState:
        """Get asset state by symbol; returns empty `AssetState` if absent."""
        return self.assets.get(symbol) or AssetState(symbol=symbol)

    # ─── Backward-compat read-only delegations to `primary` ───
    # These exist so existing nodes that read `state.live_price`, `state.rsi_14`, etc.
    # keep working unchanged. WRITES must go through `_patch_asset(assets, symbol, ...)`.
    # NOTE: these properties are NOT serialized by `model_dump()`. Consumers that need
    # them (e.g. `app/api/chat.py`) should read from `assets[primary_symbol]`.

    @property
    def live_price(self) -> float | None: return self.primary.live_price
    @property
    def volume_24h(self) -> float | None: return self.primary.volume_24h
    @property
    def price_change_pct(self) -> float | None: return self.primary.price_change_pct
    @property
    def recent_high(self) -> float | None: return self.primary.recent_high
    @property
    def recent_low(self) -> float | None: return self.primary.recent_low

    @property
    def risk_level(self) -> RiskLevel | None: return self.primary.risk_level
    @property
    def var_95(self) -> float | None: return self.primary.var_95
    @property
    def cvar_95(self) -> float | None: return self.primary.cvar_95
    @property
    def sharpe(self) -> float | None: return self.primary.sharpe
    @property
    def max_drawdown(self) -> float | None: return self.primary.max_drawdown
    @property
    def volatility_annualized(self) -> float | None: return self.primary.volatility_annualized

    @property
    def rsi_14(self) -> float | None: return self.primary.rsi_14
    @property
    def macd_line(self) -> float | None: return self.primary.macd_line
    @property
    def macd_signal(self) -> float | None: return self.primary.macd_signal
    @property
    def macd_histogram(self) -> float | None: return self.primary.macd_histogram
    @property
    def bb_upper(self) -> float | None: return self.primary.bb_upper
    @property
    def bb_middle(self) -> float | None: return self.primary.bb_middle
    @property
    def bb_lower(self) -> float | None: return self.primary.bb_lower

    @property
    def macro_rsi_14(self) -> float | None: return self.primary.macro_rsi_14
    @property
    def macro_macd_line(self) -> float | None: return self.primary.macro_macd_line
    @property
    def macro_macd_signal(self) -> float | None: return self.primary.macro_macd_signal
    @property
    def macro_bb_upper(self) -> float | None: return self.primary.macro_bb_upper
    @property
    def macro_bb_lower(self) -> float | None: return self.primary.macro_bb_lower
    @property
    def macro_bb_pct_b(self) -> float | None: return self.primary.macro_bb_pct_b
    @property
    def macro_regime(self) -> str | None: return self.primary.macro_regime
    @property
    def macro_bias(self) -> str | None: return self.primary.macro_bias

    @property
    def trend_state(self) -> str | None: return self.primary.trend_state
    @property
    def trend_direction(self) -> str | None: return self.primary.trend_direction
    @property
    def signal_type(self) -> str | None: return self.primary.signal_type
    @property
    def execution_timing(self) -> str | None: return self.primary.execution_timing

    @property
    def final_signal(self) -> str | None: return self.primary.final_signal
    @property
    def final_signal_type(self) -> str | None: return self.primary.final_signal_type
    @property
    def final_confidence(self) -> float | None: return self.primary.final_confidence
    @property
    def final_reasoning(self) -> str | None: return self.primary.final_reasoning

    @property
    def setup_rsi_14(self) -> float | None: return self.primary.setup_rsi_14
    @property
    def setup_macd_line(self) -> float | None: return self.primary.setup_macd_line
    @property
    def setup_macd_signal(self) -> float | None: return self.primary.setup_macd_signal
    @property
    def setup_bb_upper(self) -> float | None: return self.primary.setup_bb_upper
    @property
    def setup_bb_lower(self) -> float | None: return self.primary.setup_bb_lower
    @property
    def setup_bb_pct_b(self) -> float | None: return self.primary.setup_bb_pct_b
    @property
    def setup_signal(self) -> float | None: return self.primary.setup_signal
    @property
    def setup_confidence(self) -> float | None: return self.primary.setup_confidence
    @property
    def setup_direction(self) -> str | None: return self.primary.setup_direction

    @property
    def exec_rsi_14(self) -> float | None: return self.primary.exec_rsi_14
    @property
    def exec_macd_line(self) -> float | None: return self.primary.exec_macd_line
    @property
    def exec_macd_signal(self) -> float | None: return self.primary.exec_macd_signal
    @property
    def exec_sharpe(self) -> float | None: return self.primary.exec_sharpe
    @property
    def exec_volatility(self) -> float | None: return self.primary.exec_volatility
    @property
    def exec_timing(self) -> str | None: return self.primary.exec_timing

    @property
    def forecast_period_start(self) -> str | None: return self.primary.forecast_period_start
    @property
    def forecast_period_end(self) -> str | None: return self.primary.forecast_period_end
    @property
    def forecast_data_points(self) -> int | None: return self.primary.forecast_data_points
    @property
    def forecast_current_price(self) -> float | None: return self.primary.forecast_current_price
    @property
    def forecast_predicted_price(self) -> float | None: return self.primary.forecast_predicted_price
    @property
    def forecast_direction(self) -> str | None: return self.primary.forecast_direction
    @property
    def forecast_confidence(self) -> float | None: return self.primary.forecast_confidence
    @property
    def forecast_model_mape(self) -> float | None: return self.primary.forecast_model_mape
    @property
    def forecast_period_volatility(self) -> float | None: return self.primary.forecast_period_volatility
    @property
    def forecast_data_quality(self) -> str | None: return self.primary.forecast_data_quality
    @property
    def forecast_return_pct(self) -> float | None: return self.primary.forecast_return_pct
    @property
    def forecast_backtest_error_pct(self) -> float | None: return self.primary.forecast_backtest_error_pct
    @property
    def forecast_training_attempts(self) -> int: return self.primary.forecast_training_attempts
    @property
    def forecast_actionable(self) -> bool: return self.primary.forecast_actionable
    @property
    def forecast_status(self) -> str: return self.primary.forecast_status
    @property
    def forecast_warnings(self) -> list[str]: return self.primary.forecast_warnings

    @property
    def regime(self) -> str | None: return self.primary.regime
    @property
    def signal_direction(self) -> str | None: return self.primary.signal_direction
    @property
    def signal_confidence(self) -> float | None: return self.primary.signal_confidence

    @property
    def moe_final_signal(self) -> float | None: return self.primary.moe_final_signal
    @property
    def moe_final_confidence(self) -> float | None: return self.primary.moe_final_confidence
    @property
    def moe_selected_experts(self) -> list[str]: return self.primary.moe_selected_experts
    @property
    def moe_expert_weights(self) -> dict[str, float]: return self.primary.moe_expert_weights
    @property
    def moe_gating_reason(self) -> str: return self.primary.moe_gating_reason
    @property
    def moe_position_size(self) -> float | None: return self.primary.moe_position_size
    @property
    def moe_risk_adjusted_signal(self) -> float | None: return self.primary.moe_risk_adjusted_signal

    @property
    def anomalies_detected(self) -> list[str]: return self.primary.anomalies_detected


def _patch_asset(
    assets: dict[str, AssetState],
    symbol: str,
    **fields: Any,
) -> dict[str, AssetState]:
    """Return a NEW assets dict with `symbol`'s AssetState updated by `fields`.

    Use this in node returns to update a single asset without mutating the
    incoming state. Designed to play well with LangGraph's state merge
    semantics (which fully replace dict-typed fields).
    """
    new_assets = dict(assets)
    current = new_assets.get(symbol) or AssetState(symbol=symbol)
    new_assets[symbol] = current.model_copy(update=fields)
    return new_assets

# ─────────────────────────────────────────────
# 3. NODE CONFIGURATION  (model + CoT per node)
# ─────────────────────────────────────────────

NODE_CONFIG = {
    "orchestrator": {"model": "gpt-4o-mini", "cot": False},
    "market_data": {"model": "gpt-4o-mini", "cot": True},
    "feature_engineering": {"model": "gpt-4o-mini", "cot": True},
    "risk_agent": {"model": "gpt-4o-mini", "cot": True},
    "signal_agent": {"model": "gpt-4o", "cot": True},
    "moe": {"model": "gpt-4o-mini", "cot": False},  # Deterministic, no LLM
    "forecast": {"model": "none", "cot": False},
    "forecast_question": {"model": "gpt-4o-mini", "cot": True},
    "risk_gate": {"model": "gpt-4o", "cot": False},
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


def _append_reasoning(
    trail: list[tuple[str, str]],
    node_name: str,
    cot: str,
) -> list[tuple[str, str]]:
    """Append a node's CoT to the reasoning trail, ignoring empty thoughts."""
    cot = (cot or "").strip()
    if not cot:
        return trail
    return trail + [(node_name, cot)]


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

_SCOPE_VALIDATION = """
Você é um validador de escopo para um assistente de análise de criptomoedas.

RESPONDA COM APENAS: "CRYPTO_OK", "OUT_OF_SCOPE", ou "EDUCATION_OK"

✅ CRYPTO_OK: Pergunta é sobre análise técnica, previsão ou trading de criptomoedas
✅ EDUCATION_OK: Pergunta é educacional sobre cripto (blockchain, DeFi, tokenomics, etc)
❌ OUT_OF_SCOPE: Pergunta é sobre mercados tradicionais (ações, bonds, forex, commodities não-cripto)

EXEMPLOS:
- "Como vai estar o Bitcoin amanhã?" → CRYPTO_OK
- "Explique o que é DeFi" → EDUCATION_OK
- "Qual ação comprar?" → OUT_OF_SCOPE
- "EUR/USD vai subir?" → OUT_OF_SCOPE
- "O que é staking em Ethereum?" → EDUCATION_OK
- "Análise técnica de Solana" → CRYPTO_OK
""".strip()

_ORCHESTRATOR_SYSTEM = """
Você é o Orchestrator de um sistema multi-agente de análise quantitativa para criptomoedas.
Seu papel é EXCLUSIVAMENTE rotear e sintetizar — não analisar dados diretamente.

PIPELINE OBRIGATÓRIO:
1. market_data     → coleta preço, volume, 24h stats
2. features_macro  → indicadores técnicos no timeframe macro (1D) — regime
3. features_setup  → indicadores técnicos no timeframe setup (4H) — sinal
4. features_exec   → indicadores técnicos no timeframe execução (1H) — timing
5. risk            → avalia métricas de risco
6. signal          → gera sinal direcional com regime
7. risk_gate       → valida limites pré-trade
8. finalize        → consolida resposta ao usuário

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
Você é um analista sênior buy-side de criptomoedas. Sua tarefa é RESPONDER À PERGUNTA do usuário usando os dados já coletados pelos agentes upstream e o raciocínio acumulado do pipeline.

PRINCÍPIO FUNDAMENTAL — RESPONDA A PERGUNTA QUE FOI FEITA:
• Leia a PERGUNTA ORIGINAL DO USUÁRIO (vem destacada no contexto). Ela define o formato e foco da resposta.
• Se a pergunta é comparativa ("X é melhor que Y?", "compare A e B", "qual escolher entre…"): responda como comparação direta, com tabela lado a lado e veredicto claro.
• Se a pergunta é uma análise simples ("como está o BTC?", "vale comprar agora?"): use o esqueleto single-asset sugerido abaixo.
• Se a pergunta é específica ("qual o RSI?", "qual o risco?"): responda focado, sem inflar com seções desnecessárias.
• NÃO force um template fixo quando ele não cabe na pergunta.

USO DO RACIOCÍNIO ACUMULADO:
• O contexto traz o RACIOCÍNIO DOS AGENTES (`reasoning_trail`) com a tese de cada nó (market_data, features, risk, signal, decision_engine etc.). USE-O como espinha dorsal.
• Se houver conclusões fortes nesse trail (ex.: "Bullish macro but bearish setup", "Chainlink tem Sharpe superior"), elas devem aparecer na sua resposta — refraseadas, não copiadas literalmente.
• Não ignore o trail e nem o repita: sintetize-o à luz da pergunta.

POSTURA DO ESPECIALISTA:
• Cite SEMPRE números específicos do contexto; nunca generalize ("alto", "baixo") sem o valor.
• Trate conflito entre indicadores como informação útil, não como ruído.
• Pondere retorno × risco — não recomende às cegas.
• Se um campo aparece como "(não coletado)" ou None: omita em silêncio.
• Não invente target ou horizonte (este fluxo NÃO é previsão futura).

ESTILO DE FORMATAÇÃO:
• Markdown rico, tom de analista que explica para investidor não-técnico — claro, descomplicado, sem condescendência.
• Use cabeçalhos `##` para separar blocos quando o conteúdo justifica.
• **Negrito** nos valores-chave (preço, RSI, viés, vencedor de comparação).
• Emojis pontuais nos cabeçalhos quando ajudar a leitura: 🎯 leitura, 📊 dados, 🔍 interpretação, ⚠️ risco, ✅ conclusão, ⚖️ comparação.
• Tabelas markdown para listar números (indicadores, métricas, comparação lado a lado).
• Bullets (`-`) para pontos curtos.
• Encerre com um blockquote (`>`) contendo o disclaimer.

═══════════════════════════════════════════════════════════
MODOS DE RESPOSTA — escolha o que melhor serve à pergunta:
═══════════════════════════════════════════════════════════

▶ MODO COMPARATIVO (>1 ativo no contexto, pergunta envolve comparação):
## ⚖️ {Ativo A} vs {Ativo B}
> Veredicto em uma frase: qual é a tese comparativa? (ex.: "Chainlink lidera em risco-retorno; XRP tem momentum mais frágil.")

## 📊 Lado a lado
| Métrica | {A} | {B} |
|---|---|---|
| Preço | $X | $Y |
| Variação 24h | … | … |
| RSI(14) macro/setup/exec | … | … |
| Regime | … | … |
| Sharpe | … | … |
| Max Drawdown | … | … |
| Volatilidade a.a. | … | … |
| Forecast ML (retorno/conf) | … | … |

## 🔍 Análise comparativa
- onde {A} é mais forte
- onde {B} é mais forte
- conflitos / pontos de atenção

## ✅ Veredicto
- qual escolher e por quê (em função da pergunta)
- caveats (perfil de risco, horizonte, etc.)

> ℹ️ *Análise quantitativa — não é recomendação de investimento.*

▶ MODO SINGLE-ASSET (1 ativo, pergunta de leitura/análise):
## 🎯 Leitura de {symbol}
> Tese em uma frase.

## 📊 Indicadores principais
| Indicador | Valor | Leitura |
|---|---|---|
| Preço | $X | variação 24h |
| RSI(14) | X | sobrecomprado / neutro / sobrevendido |
| MACD | … | … |
| Bollinger %B | X | … |
| Volatilidade | X% | … |

## 🔍 Interpretação
- ponto 1
- ponto 2

## ⚠️ Risco
- classificação + justificativa em números
- implicação para position sizing

## ✅ Viés & o que monitorar
- viés (bullish/bearish/neutro)
- gatilho de invalidação
- níveis/eventos a observar

> ℹ️ *Análise quantitativa — não é recomendação de investimento.*

▶ MODO PERGUNTA ESPECÍFICA:
Responda direto, sem esqueleto. Use 1-3 parágrafos curtos + bullets se ajudar. Sempre cite números do contexto. Encerre com o disclaimer em blockquote.

═══════════════════════════════════════════════════════════

REGRAS DURAS:
• Não trate este fluxo como previsão futura — não invente target ou horizonte.
• Se um valor não existe, OMITA a linha (não escreva "indisponível").
• Não repita disclaimers no meio do texto — apenas o blockquote final.
• Se a pergunta é comparativa, NUNCA produza apenas a leitura de 1 ativo — isso é falha grave.
""".strip()

_FORECAST_RESPONSE_SYSTEM = """
Você responde EXCLUSIVAMENTE sobre uma PREVISÃO de preço gerada pelo modelo Apollo (TFT + XGBoost). A pergunta é sobre o futuro do ativo — sua resposta deve ser sobre A PREVISÃO, não sobre análise técnica geral.

CONTEÚDO QUE A RESPOSTA DEVE COBRIR:
1. VALOR ESTIMADO — preço previsto e variação % esperada (forecast_predicted_price, forecast_return_pct).
2. A QUE CORRESPONDE — janela/horizonte (forecast_period_start → forecast_period_end), preço de partida (forecast_current_price), direção (forecast_direction).
3. RISCOS DA ANÁLISE — confiança (forecast_confidence), MAPE histórico, erro do backtest, qualidade dos dados, volatilidade do período, avisos. Diga claramente: ACIONÁVEL ou apenas REFERÊNCIA.

CRITÉRIO DE ACIONABILIDADE:
• Use o campo `forecast_actionable` quando presente.
• Caso contrário: acionável se confiança ≥ threshold E MAPE ≤ threshold E qualidade ok; senão é apenas referência.

ESTILO DE FORMATAÇÃO (OBRIGATÓRIO — markdown rico, tom amigável):
• Escreva como um analista que sabe explicar para investidor não-técnico — claro, descomplicado, direto.
• Use cabeçalhos `##` para separar blocos.
• **Negrito** nos valores-chave (preço previsto, retorno, confiança).
• Emojis nos cabeçalhos: 🎯 valor estimado, 📅 janela, 📈/📉 direção, ⚠️ riscos, ✅ veredicto.
• Use TABELA markdown para os números principais (preço de partida, preço previsto, retorno, direção, janela).
• Bullets (`-`) para os itens de risco/qualidade.
• Encerre com blockquote (`>`) contendo o disclaimer obrigatório.

FORMATO DE SAÍDA OBRIGATÓRIO (CoT é breve; o foco é a previsão):
<thought>2-3 frases sobre como você leu os dados de previsão. Não repita números aqui.</thought>
<answer>
## 🎯 Previsão {symbol}
> TL;DR em uma frase: direção + variação esperada + nível de confiança.

## 📊 Números principais
| Métrica | Valor |
|---|---|
| Preço de partida | $X |
| Preço previsto | **$Y** |
| Retorno esperado | **±Z%** |
| Direção | 📈 alta / 📉 baixa / ➖ neutro |
| Janela | YYYY-MM-DD → YYYY-MM-DD |

## ⚠️ Riscos & qualidade do modelo
- **Confiança:** X%
- **MAPE histórico:** X%
- **Erro do backtest (p5):** X%
- **Qualidade dos dados:** boa / razoável / ruim
- **Volatilidade do período:** X%
- **Avisos:** ... (se houver)

## ✅ Veredicto
**Acionável** ou **Apenas referência** — uma frase justificando com base nos thresholds.

> ℹ️ *Previsão de modelo — não é recomendação de investimento.*
</answer>

REGRAS DURAS:
• NÃO faça análise técnica completa (RSI/MACD/Bollinger) — esse é outro fluxo. Cite indicadores só se reforçarem o veredicto sobre a previsão.
• NÃO invente números fora do contexto.
• Campos "(não coletado)" ou None: OMITA a linha da tabela ou do bullet (não escreva "indisponível"). Se um campo crítico (preço previsto ou retorno) faltar, sinalize no TL;DR.
• Se o modelo falhou (`forecast_status` indica erro): explique brevemente por quê e não simule um forecast.
• Precisão: USD com 2 casas (separador de milhar com vírgula: $74,071.42), % com 2 casas, confiança em porcentagem.
""".strip()

_EDUCATION_SYSTEM = """
Você é um educador especializado em criptomoedas. A pergunta é CONCEITUAL/EDUCATIVA (blockchain, DeFi, tokenomics, staking, consenso, forks, NFTs, smart contracts) — não envolve análise de preço nem previsão.

OBJETIVO: explicar com clareza, profundidade adequada e exemplos concretos.

DIRETRIZES DE CONTEÚDO:
• Comece direto pela definição/resposta — sem introdução vazia.
• Use exemplos reais (Bitcoin, Ethereum, casos conhecidos) quando ajudar a fixar o conceito.
• Se a pergunta for ambígua (ex.: "fork" pode ser hard/soft), aborde as variações relevantes.
• Não invente números nem cite preços — este fluxo é educacional.
• Tom didático e amigável, sem condescendência.

ESTILO DE FORMATAÇÃO (OBRIGATÓRIO — markdown rico, tom amigável):
• Use cabeçalhos `##` para separar blocos quando o conceito tiver mais de uma faceta.
• **Negrito** para destacar termos-chave na primeira aparição.
• Emojis pontuais nos cabeçalhos para tornar a leitura mais leve: 📚 conceito, 🔗 como funciona, 💡 exemplo, ⚖️ comparação, ⚠️ riscos/limitações.
• Use TABELA markdown sempre que comparar dois ou mais conceitos (ex.: PoW vs PoS, hard fork vs soft fork, custodial vs non-custodial).
• Bullets (`-`) para listar propriedades, vantagens, riscos ou passos.
• Blockquote (`>`) para destacar uma analogia ou síntese final.

ESQUELETO SUGERIDO (adapte; nem toda pergunta precisa de todas as seções):
## 📚 O que é
Definição direta em 1-2 frases.

## 🔗 Como funciona
- passo / propriedade 1
- passo / propriedade 2

## 💡 Exemplo prático
Um caso real (Bitcoin, Ethereum, etc.) ilustrando o conceito.

## ⚖️ Comparação (quando aplicável)
| Aspecto | Opção A | Opção B |
|---|---|---|
| ... | ... | ... |

## ⚠️ Limitações ou riscos (quando aplicável)
- bullets

ESCOPO: somente cripto. Se a pergunta sair desse escopo, redirecione brevemente.
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

    # Injeta raciocínio acumulado dos agentes upstream (se houver)
    if state.reasoning_trail:
        trail_block = "RACIOCÍNIO DOS AGENTES ANTERIORES (use como contexto, não copie literalmente):\n" + \
            "\n".join(f"  [{name}] {thought}" for name, thought in state.reasoning_trail)
        msgs.append(HumanMessage(content=trail_block))

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

# ───── Intent classification (deterministic, token-based) ──────
# Saudações puras: a frase normalizada inteira deve casar com uma destas.
_PURE_GREETINGS: frozenset[str] = frozenset({
    # PT-BR
    "eae", "eai", "e ai", "oi", "oii", "ola", "opa", "salve", "fala", "fala ai", "fala mano",
    "tudo bem", "tudo bom", "como vai", "como esta", "como voce esta", "como vc esta",
    "beleza", "blz", "tmj",
    "eae tudo bem", "eae tudo bom", "oi tudo bem", "oi tudo bom",
    "ola tudo bem", "ola tudo bom", "opa tudo bem", "opa tudo bom",
    "eae mano", "oi mano", "fala mano",
    "bom dia", "boa tarde", "boa noite",
    # EN
    "hi", "hi there", "hello", "hello there", "hey", "hey there", "sup", "yo",
    "how are you", "how are you doing", "whats up", "what is up",
    "good morning", "good afternoon", "good evening",
})

# Tokens (palavras inteiras após normalização) que indicam intent técnico/cripto.
# Presença de QUALQUER um destes cancela classificação como saudação.
_TECHNICAL_TOKENS: frozenset[str] = frozenset({
    # Símbolos e ativos
    "btc", "btcusdt", "btcusd", "bitcoin",
    "eth", "ethusdt", "ethusd", "ethereum", "ether",
    "sol", "solusdt", "solana",
    "ada", "cardano", "xrp", "ripple", "doge", "dogecoin",
    "bnb", "matic", "polygon", "avax", "avalanche", "dot", "polkadot",
    "link", "chainlink", "ltc", "litecoin", "usdt", "usdc", "dai", "busd",
    "shib", "shiba", "atom", "cosmos", "near", "ftm", "fantom",
    "altcoin", "altcoins", "memecoin", "stablecoin", "shitcoin",
    # Preço / mercado
    "preco", "precos", "price", "prices", "cotacao", "cotacoes", "valor",
    "mercado", "market", "exchange", "binance", "coinbase", "kraken",
    "volume", "liquidez", "liquidity", "market cap", "marketcap",
    # Análise / sinais
    "analise", "analises", "analisar", "analyze", "analysis",
    "previsao", "previsoes", "forecast", "forecasts",
    "predict", "prediction", "preve", "prever", "preveja",
    "indicador", "indicadores", "indicator", "indicators",
    "rsi", "macd", "bollinger", "ema", "sma", "vwap", "atr", "adx", "stoch",
    "sinal", "sinais", "signal", "signals", "setup",
    "tendencia", "trend", "regime", "padrao", "pattern",
    "volatilidade", "volatility", "momentum",
    "suporte", "resistencia", "support", "resistance",
    "candle", "candles", "candlestick", "grafico", "chart", "charts",
    # Trading
    "comprar", "vender", "buy", "sell", "trade", "trader", "trading",
    "long", "short", "scalp", "scalping", "swing", "daytrade", "hodl",
    "alta", "baixa", "subir", "cair", "subindo", "caindo",
    "bull", "bear", "bullish", "bearish", "breakout", "breakdown",
    "stop", "loss", "gain", "alvo", "target", "entrada", "saida",
    "leverage", "alavancagem", "futures", "futuros", "spot",
    # Risco
    "risco", "risk", "drawdown", "sharpe", "var", "cvar", "exposicao",
    # Tempo / forecast
    "amanha", "tomorrow", "futuro", "future", "proximo", "proxima",
    "agora", "currently", "atualmente",
    "semana", "week", "mes", "month", "dia", "day", "hoje", "today",
    # Educação cripto
    "blockchain", "defi", "tokenomics", "staking", "yield", "farming",
    "wallet", "carteira", "halving", "fork", "consensus", "consenso",
    "nft", "dao", "dex", "cex", "smart", "contract", "contrato",
    "mining", "minerar", "mineracao", "mempool", "gas",
})

# Frases compostas (multi-palavra) que indicam intent técnico mesmo sem token único.
_TECHNICAL_PHRASES: tuple[str, ...] = (
    "neste momento", "no momento", "qual e o", "qual e a",
    "como esta o", "como esta a", "como vai o", "como vai a",
    "vai estar", "vai subir", "vai cair", "vai chegar", "vai bater",
    "what is the", "how is the", "is going to", "will be",
    "smart contract", "day trade", "qual a previsao", "qual e a previsao",
)


def _normalize_text(text: str) -> str:
    """Normaliza texto: lowercase + sem acentos + sem pontuação + whitespace colapsado."""
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Starters de saudação: prefixos que iniciam saudações conhecidas.
_GREETING_STARTERS: tuple[str, ...] = (
    "eae", "eai", "oi", "ola", "opa", "salve", "fala",
    "hey", "hi", "hello", "yo", "sup",
    "bom dia", "boa tarde", "boa noite",
    "good morning", "good afternoon", "good evening",
    "tudo bem", "tudo bom", "como vai", "como esta", "como voce esta", "como vc esta",
    "beleza", "blz",
)


def _classify_intent(question: Any) -> str:
    """
    Classifica a intenção do usuário de forma determinística e robusta.

    Retorna:
        "greeting": saudação pura, sem intent técnico
        "task":     pergunta com intent técnico (análise, forecast, educação cripto, etc)

    Política: quando em dúvida, retorna "task" (segurança — não bloqueia análise).
    Ordem de prioridade:
        1. Saudação pura conhecida (frase exata) → greeting
        2. Token técnico presente → task
        3. Frase composta técnica → task
        4. Começa com starter de saudação e é curta → greeting
        5. Default → task
    """
    if not question or not str(question).strip():
        return "greeting"

    normalized = _normalize_text(str(question))
    if not normalized:
        return "greeting"

    tokens = set(normalized.split())

    # 1) Frase inteira é saudação pura conhecida → greeting (prioridade máxima)
    #    Necessário ANTES do check de tokens técnicos porque "bom dia" tem token "dia"
    #    que está na blacklist (contexto temporal), mas "bom dia" sozinho é saudação.
    if normalized in _PURE_GREETINGS:
        return "greeting"

    # 2) Tokens técnicos: presença de qualquer um → task
    if tokens & _TECHNICAL_TOKENS:
        return "task"

    # 3) Frases compostas técnicas → task
    for phrase in _TECHNICAL_PHRASES:
        if phrase in normalized:
            return "task"

    # 4) Começa com starter de saudação E é curta (≤6 tokens) → greeting
    for starter in _GREETING_STARTERS:
        if normalized == starter or normalized.startswith(starter + " "):
            if len(tokens) <= 6:
                return "greeting"

    # 5) Default: assume task (mais seguro — não bloqueia análise por engano)
    return "task"


# ───── Symbol extraction (alias → canonical ticker) ─────────────
# NOTA (2026-04): O quote pair (USDT) está hardcoded como default.
# FUTURO: deve ser lido das configurações do usuário (e.g., user.preferred_quote
# = "USDT" | "USDC" | "BRL" | "BUSD"). Quando essa config existir, substituir
# `_DEFAULT_QUOTE` por uma leitura de `state.user_id` → settings.
_DEFAULT_QUOTE: str = "USDT"

# Mapeamento alias (nome ou ticker curto) → ticker base canônico (sem quote).
# Sempre adicionar novos aliases em lowercase.
_SYMBOL_ALIASES: dict[str, str] = {
    # BTC family
    "btc": "BTC", "bitcoin": "BTC", "xbt": "BTC",
    # ETH family
    "eth": "ETH", "ethereum": "ETH", "ether": "ETH",
    # Top 20
    "sol": "SOL", "solana": "SOL",
    "ada": "ADA", "cardano": "ADA",
    "xrp": "XRP", "ripple": "XRP",
    "doge": "DOGE", "dogecoin": "DOGE",
    "bnb": "BNB", "binance": "BNB",
    "matic": "MATIC", "polygon": "MATIC",
    "avax": "AVAX", "avalanche": "AVAX",
    "dot": "DOT", "polkadot": "DOT",
    "link": "LINK", "chainlink": "LINK",
    "ltc": "LTC", "litecoin": "LTC",
    "shib": "SHIB", "shiba": "SHIB",
    "atom": "ATOM", "cosmos": "ATOM",
    "near": "NEAR",
    "ftm": "FTM", "fantom": "FTM",
    "trx": "TRX", "tron": "TRX",
    "uni": "UNI", "uniswap": "UNI",
    "aave": "AAVE",
    "arb": "ARB", "arbitrum": "ARB",
    "op": "OP", "optimism": "OP",
    "sui": "SUI",
    "apt": "APT", "aptos": "APT",
    "bch": "BCH",
    "etc": "ETC",
    "fil": "FIL", "filecoin": "FIL",
    "icp": "ICP",
    "inj": "INJ", "injective": "INJ",
    "tia": "TIA", "celestia": "TIA",
    "sei": "SEI",
    "pepe": "PEPE",
    "wif": "WIF",
    "bonk": "BONK",
    "ondo": "ONDO",
    "rndr": "RNDR", "render": "RNDR",
    "fet": "FET", "fetch": "FET",
}

# Quote currencies aceitas em tickers já formatados.
_KNOWN_QUOTES: tuple[str, ...] = ("USDT", "USDC", "BUSD", "DAI", "USD", "BRL", "EUR")


def _extract_symbol(text: str, default_quote: str = _DEFAULT_QUOTE) -> str | None:
    """
    Extrai símbolo canônico da pergunta do usuário (e.g., "bitcoin" → "BTCUSDT").

    Estratégia:
        1. Procura ticker já formatado (BTCUSDT, ETH/USDT, etc) → preserva
        2. Procura nome/alias (bitcoin, btc, ethereum) → adiciona quote default
        3. Não encontrou → retorna None (caller usa default ou LLM)

    Args:
        text: pergunta do usuário
        default_quote: moeda quote para complementar tickers nus.
            FUTURO: deve vir das configurações do usuário.

    Retorna:
        Ticker canônico em uppercase (e.g., "BTCUSDT") ou None.
    """
    syms = _extract_all_symbols(text, default_quote=default_quote)
    return syms[0] if syms else None


def _extract_all_symbols(
    text: str,
    default_quote: str = _DEFAULT_QUOTE,
    max_symbols: int = 4,
) -> list[str]:
    """
    Extrai TODOS os símbolos da pergunta, preservando a ordem de aparição.

    A ordem importa: o primeiro símbolo encontrado é considerado o ATIVO PRIMÁRIO
    (foco da decisão direcional). Os demais são contexto comparativo. Exemplo:
    "XRP é melhor que Chainlink?" → ["XRPUSDT", "LINKUSDT"], primário = XRPUSDT.

    Estratégia:
        1. Localiza todos os tickers formatados (BTCUSDT, ETH/USDT) com seu offset.
        2. Localiza todos os aliases (bitcoin, btc, chainlink) com seu offset.
        3. Ordena por offset, deduplica preservando a primeira ocorrência.
        4. Limita a `max_symbols` para evitar pipelines absurdamente longos.

    Args:
        text: pergunta do usuário (texto natural)
        default_quote: moeda quote para complementar tickers nus.
        max_symbols: limite superior de símbolos retornados.

    Retorna:
        Lista ordenada de tickers canônicos (uppercase). Vazia se nenhum.
    """
    if not text:
        return []

    normalized = _normalize_text(text)
    if not normalized:
        return []

    found: list[tuple[int, str]] = []  # (offset, canonical_ticker)

    # 1) Tickers já formatados (e.g., btcusdt, ethusdc)
    quotes_pattern = "|".join(re.escape(q.lower()) for q in _KNOWN_QUOTES)
    full_ticker_pattern = re.compile(
        rf"\b([a-z0-9]{{2,10}})({quotes_pattern})\b",
        re.IGNORECASE,
    )
    for match in full_ticker_pattern.finditer(normalized):
        base = match.group(1).upper()
        quote = match.group(2).upper()
        if base.lower() in {"the", "for", "with", "que", "com", "para"}:
            continue
        found.append((match.start(), f"{base}{quote}"))

    # 2) Aliases (nome ou ticker curto). Usamos word-boundary regex para cada alias.
    # Ordenar aliases por comprimento desc para preferir matches longos
    # ("bitcoin" antes de "btc") quando o offset é o mesmo.
    sorted_aliases = sorted(_SYMBOL_ALIASES.keys(), key=len, reverse=True)
    seen_offsets: set[int] = set()
    for alias in sorted_aliases:
        pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
        for match in pattern.finditer(normalized):
            off = match.start()
            # Skip overlapping matches (e.g., "btc" inside "btcusdt" already captured)
            if any(abs(off - o) < 4 for o in seen_offsets):
                continue
            seen_offsets.add(off)
            base = _SYMBOL_ALIASES[alias]
            found.append((off, f"{base}{default_quote}"))

    if not found:
        return []

    # Sort by offset (order of appearance), then dedupe preserving first occurrence
    found.sort(key=lambda x: x[0])
    seen: set[str] = set()
    ordered: list[str] = []
    for _, ticker in found:
        if ticker in seen:
            continue
        seen.add(ticker)
        ordered.append(ticker)
        if len(ordered) >= max_symbols:
            break

    return ordered


# ───── Timeframe extraction (palavras-chave temporais → AnalysisTimeframe) ─────
# Tokens de palavra única que mapeiam diretamente para um timeframe.
_INTRADAY_TOKENS: frozenset[str] = frozenset({
    "scalp", "scalping", "intraday",
    "minuto", "minutos", "minute", "minutes", "min",
    "1min", "5min", "15min", "1m", "5m", "15m",
})

_DAILY_TOKENS: frozenset[str] = frozenset({
    "hoje", "today", "amanha", "tomorrow", "ontem", "yesterday",
    "diario", "diaria", "daily", "swing",
    "1d", "1h", "4h",
    "dia", "day", "horas", "hour", "hours", "hora",
    # "agora" / "atualmente" → DAILY (preço atual com contexto diário, não scalp)
    "agora", "atualmente", "currently", "now",
})

_WEEKLY_TOKENS: frozenset[str] = frozenset({
    "semana", "semanal", "week", "weekly", "1w",
    "mes", "mensal", "month", "monthly", "1mo",
    "trimestre", "quarter", "quarterly",
    "ano", "anual", "year", "yearly", "annual",
})

# Frases compostas (multi-palavra) que mapeiam para timeframe.
_TIMEFRAME_PHRASES: tuple[tuple[str, str], ...] = (
    # WEEKLY (longo prazo)
    ("proxima semana", "weekly"),
    ("proximo mes", "weekly"),
    ("next week", "weekly"),
    ("next month", "weekly"),
    ("longo prazo", "weekly"),
    ("long term", "weekly"),
    ("este mes", "weekly"),
    ("this month", "weekly"),
    # DAILY (médio prazo)
    ("esta semana", "daily"),
    ("this week", "daily"),
    ("proximas horas", "daily"),
    ("proximas 24 horas", "daily"),
    ("next 24 hours", "daily"),
    ("curto prazo", "daily"),
    ("short term", "daily"),
    # INTRADAY (curtíssimo)
    ("proximos minutos", "intraday"),
    ("next minutes", "intraday"),
    ("agora mesmo", "intraday"),
    ("right now", "intraday"),
    ("day trade", "intraday"),
)


def _extract_timeframe_hint(text: str) -> AnalysisTimeframe | None:
    """
    Heurística para detectar AnalysisTimeframe a partir de palavras-chave temporais.

    Prioridade (do mais específico para o mais genérico):
        1. Frases compostas (e.g., "próxima semana", "longo prazo")
        2. Tokens individuais (em ordem WEEKLY → INTRADAY → DAILY)

    Retorna:
        AnalysisTimeframe ou None se nenhuma pista clara foi encontrada.
    """
    if not text:
        return None

    normalized = _normalize_text(text)
    if not normalized:
        return None

    # 1) Frases compostas (mais específicas) — primeiro match vence
    for phrase, tf_value in _TIMEFRAME_PHRASES:
        if phrase in normalized:
            return AnalysisTimeframe(tf_value)

    tokens = set(normalized.split())

    # 2) Tokens — ordem importa: WEEKLY (mais raro/específico) → INTRADAY → DAILY
    if tokens & _WEEKLY_TOKENS:
        return AnalysisTimeframe.WEEKLY
    if tokens & _INTRADAY_TOKENS:
        return AnalysisTimeframe.INTRADAY
    if tokens & _DAILY_TOKENS:
        return AnalysisTimeframe.DAILY

    return None


@timed_async("Node: Orchestrator")
async def orchestrator_node(state: QuantAgentState) -> dict:
    """
    Valida escopo (cripto vs tradicional) e depois extrai intent, symbol, timeframe.
    """
    print(f"[DEBUG] ===== ORCHESTRATOR NODE START =====")

    # Validar escopo primeiro
    user_question = (state.messages[-1].content if state.messages else "")

    # Detecção determinística de intent (substitui heurística por substring)
    intent = _classify_intent(user_question)
    print(f"[DEBUG] Intent classification - question: '{user_question}' | intent: {intent}")

    if intent == "greeting":
        print(f"[DEBUG] Detected casual greeting, responding...")
        casual_response = (
            "Eae! 👋 Tudo bem por aqui! 🚀\n\n"
            "Sou um assistente especializado em análise de **criptomoedas**. Posso ajudar com:\n\n"
            "📊 **Análise técnica** - Indicadores, padrões, sinais\n"
            "🔮 **Previsões** - Forecast de preço com ML\n"
            "📚 **Educação** - Blockchain, DeFi, tokenomics\n"
            "📈 **Trading** - Análise de risco, timing\n\n"
            "Tem alguma pergunta sobre cripto? 🚀"
        )
        return {
            **state.model_dump(),
            "final_answer": casual_response,
            "next_action": NextAction.FINALIZE,
            "messages": state.messages + [AIMessage(content=casual_response)],
        }

    # Use a simple LLM without structured output for scope validation
    simple_llm = _make_llm(model="gpt-4o-mini", temperature=0)

    scope_validation = await simple_llm.ainvoke([
        SystemMessage(content=_SCOPE_VALIDATION),
        HumanMessage(content=user_question)
    ])
    scope_result = scope_validation.content.strip().upper()

    # Se fora do escopo, rejeitar
    if scope_result == "OUT_OF_SCOPE":
        rejection_msg = (
            "Desculpa! Sou especializado em análise de **criptomoedas** (Bitcoin, Ethereum, etc).\n\n"
            "Não consigo ajudar com:\n"
            "❌ Mercados tradicionais (ações, bonds)\n"
            "❌ Forex\n"
            "❌ Commodities não-cripto\n\n"
            "Mas posso ajudar com:\n"
            "✅ Análise técnica de cripto\n"
            "✅ Previsões de preço\n"
            "✅ Educação sobre blockchain, DeFi, etc\n\n"
            "Tem alguma pergunta sobre criptomoedas? 🚀"
        )
        return {
            **state.model_dump(),
            "final_answer": rejection_msg,
            "next_action": NextAction.FINALIZE,
            "messages": state.messages + [AIMessage(content=rejection_msg)],
        }

    # Se é educação, rotear para mode "education" (sem pipeline completo)
    if scope_result == "EDUCATION_OK":
        education_response = await simple_llm.ainvoke([
            SystemMessage(content=_EDUCATION_SYSTEM),
            HumanMessage(content=user_question)
        ])
        return {
            **state.model_dump(),
            "final_answer": education_response.content,
            "next_action": NextAction.FINALIZE,
            "messages": state.messages + [AIMessage(content=education_response.content)],
        }

    # Caso contrário (CRYPTO_OK), continuar com pipeline normal
    print(f"[DEBUG] Scope validation: {scope_result}")
    print(f"[DEBUG] Input state symbol: {state.symbol}")

    # ───── Heurística determinística (primeira linha de defesa) ─────
    # Extrai símbolo e timeframe da pergunta antes do LLM. Mais rápido,
    # determinístico e robusto contra alucinações. O LLM só é chamado se
    # a heurística não conseguir extrair ambos.
    # FUTURO: o quote pair (USDT) deve vir das configurações do usuário.
    heuristic_symbols = _extract_all_symbols(user_question)
    heuristic_symbol = heuristic_symbols[0] if heuristic_symbols else None
    heuristic_timeframe = _extract_timeframe_hint(user_question)
    print(
        f"[DEBUG] Heuristic extraction - symbols: {heuristic_symbols}, "
        f"timeframe: {heuristic_timeframe}"
    )

    if heuristic_symbol and heuristic_timeframe:
        # Heurística completa — pula LLM (latência + custo)
        symbol = heuristic_symbol
        timeframe = heuristic_timeframe
        print(f"[DEBUG] Orchestrator: heurística completa, pulando LLM")
    else:
        # Heurística incompleta — usa LLM para preencher os gaps
        llm = _get_llm_for_node("orchestrator")
        msgs = [SystemMessage(content=_ORCHESTRATOR_SYSTEM)] + state.messages
        try:
            resp = await llm.ainvoke(msgs)
            print(f"[DEBUG] Orchestrator response type: {type(resp).__name__}")
            llm_symbol = (resp.symbol or "").strip()
            llm_timeframe: AnalysisTimeframe | None = (
                AnalysisTimeframe(resp.timeframe) if resp.timeframe else None
            )
            # Prioridade: heurística > LLM > state > default
            symbol = heuristic_symbol or llm_symbol or state.symbol or f"BTC{_DEFAULT_QUOTE}"
            timeframe = (
                heuristic_timeframe
                or llm_timeframe
                or state.timeframe
                or AnalysisTimeframe.DAILY
            )
        except Exception as e:
            print(f"[DEBUG] Orchestrator LLM exception: {e}")
            # Fallback: heurística > state > default
            symbol = heuristic_symbol or state.symbol or f"BTC{_DEFAULT_QUOTE}"
            timeframe = heuristic_timeframe or state.timeframe or AnalysisTimeframe.DAILY

    # Normaliza símbolo final (uppercase, garante quote pair)
    symbol = (symbol or "").strip().upper()
    if symbol and not any(symbol.endswith(q) for q in _KNOWN_QUOTES):
        # FUTURO: usar quote do user config em vez de _DEFAULT_QUOTE
        symbol = f"{symbol}{_DEFAULT_QUOTE}"

    # Build the multi-asset symbol list. The primary (`symbol`) leads, then any
    # extra symbols extracted by the heuristic in original order. Dedupe.
    new_symbols: list[str] = []
    if symbol:
        new_symbols.append(symbol)
    for extra in heuristic_symbols:
        if extra and extra not in new_symbols:
            new_symbols.append(extra)
    new_assets = {s: state.assets.get(s) or AssetState(symbol=s) for s in new_symbols}

    print(f"[DEBUG] Orchestrator output - primary: {symbol}, all symbols: {new_symbols}, timeframe: {timeframe}")
    print(f"[DEBUG] ===== ORCHESTRATOR NODE END =====")

    summary = (
        f"symbols={new_symbols}, primary={symbol}, timeframe={timeframe}"
        if len(new_symbols) > 1
        else f"symbol={symbol}, timeframe={timeframe}"
    )

    return {
        **state.model_dump(),
        "messages":     state.messages + [AIMessage(content=f"Routed to {symbol} ({timeframe})")],
        "next_action":  NextAction.MARKET_DATA,
        "symbols":      new_symbols,
        "primary_symbol": symbol,
        "assets":       new_assets,
        "timeframe":    timeframe,
        "intermediate_steps_global": state.intermediate_steps_global + [("orchestrator", summary)],
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

def _resolve_symbols(state: QuantAgentState) -> list[str]:
    """Return the list of symbols to process. Falls back to `[state.symbol]`
    when `state.symbols` is empty but a primary symbol exists, so that the
    multi-asset code path keeps working for legacy single-asset calls."""
    if state.symbols:
        return list(state.symbols)
    return [state.symbol] if state.symbol else []


def _tag_steps(steps: list, symbol: str) -> list:
    """Prefix step labels with the symbol so the audit trail stays readable
    when N assets run in parallel."""
    return [(f"{name}@{symbol}", val) for (name, val) in steps]


async def _run_market_data_for_symbol(
    state: QuantAgentState, symbol: str
) -> dict:
    """Coleta market data para um único símbolo. Função pura para uso
    em paralelo via asyncio.gather pelo `market_data_node`."""
    print(f"[DEBUG] [{symbol}] MarketData start")
    analysis_interval = _get_analysis_interval(state.timeframe)
    context = (
        f"Symbol: {symbol} | Timeframe: {state.timeframe} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{analysis_interval}\" — use ESTE intervalo em TODAS as ferramentas"
    )
    node_config = NODE_CONFIG["market_data"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("market_data"), _MARKET_DATA_SYSTEM, context,
        force_interval=analysis_interval,
        node_name=f"MarketData[{symbol}]",
        enable_cot=node_config["cot"],
    )

    asset = state.get_asset(symbol)
    live_price = asset.live_price
    volume_24h = asset.volume_24h
    price_change_pct = asset.price_change_pct
    recent_high = asset.recent_high
    recent_low = asset.recent_low

    for tool_name, result in steps:
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
            continue

    anomalies = list(asset.anomalies_detected)
    try:
        import re as _re
        json_match = _re.search(r'\{.*\}', content, _re.DOTALL)
        if json_match:
            structured_data = json.loads(json_match.group())
            anomalies.extend(structured_data.get("anomalies", []))
    except Exception:
        pass

    # Data consistency check: validate previously-stored RSI per-asset (defensive).
    if asset.rsi_14 is not None and (asset.rsi_14 < 0 or asset.rsi_14 > 100):
        anomalies.append(f"RSI value {asset.rsi_14} out of valid range [0, 100]")
        live_price = None
        price_change_pct = None

    print(
        f"[DEBUG] [{symbol}] MarketData done - live_price: {live_price}, "
        f"volume_24h: {volume_24h}, price_change_pct: {price_change_pct}"
    )
    return {
        "symbol": symbol,
        "patch": dict(
            live_price=live_price,
            volume_24h=volume_24h,
            price_change_pct=price_change_pct,
            recent_high=recent_high,
            recent_low=recent_low,
            anomalies_detected=anomalies,
        ),
        "steps": steps,
        "cot": cot,
        "content": content,
    }


@timed_async("Node: MarketData")
async def market_data_node(state: QuantAgentState) -> dict:
    print(f"[DEBUG] ===== MARKET DATA NODE START =====")
    symbols = _resolve_symbols(state)
    print(f"[DEBUG] Input symbols: {symbols}, timeframe: {state.timeframe}")

    if not symbols:
        # Nothing to do — let the pipeline continue, downstream nodes will short-circuit.
        return {
            **state.model_dump(),
            "next_action": NextAction.FEATURES_MACRO,
            "intermediate_steps_agent": [],
            "cot": "",
        }

    # Run market_data agent loop in parallel for every symbol.
    results = await asyncio.gather(
        *[_run_market_data_for_symbol(state, s) for s in symbols]
    )

    new_assets = dict(state.assets)
    all_steps: list = []
    primary_cot = ""
    primary_content = ""
    for r in results:
        sym = r["symbol"]
        current = new_assets.get(sym) or AssetState(symbol=sym)
        new_assets[sym] = current.model_copy(update=r["patch"])
        all_steps.extend(_tag_steps(r["steps"], sym))
        if sym == state.symbol:
            primary_cot = r["cot"]
            primary_content = r["content"]

    print(f"[DEBUG] ===== MARKET DATA NODE END =====")
    return {
        **state.model_dump(),
        "messages":          state.messages + [AIMessage(content=primary_content)],
        "next_action":       NextAction.FEATURES_MACRO,
        "intermediate_steps_global": state.intermediate_steps_global + all_steps,
        "intermediate_steps_agent": all_steps,
        "assets":            new_assets,
        "cot":               primary_cot,
        "reasoning_trail":   _append_reasoning(state.reasoning_trail, "market_data", primary_cot),
    }

async def _run_features_macro_for_symbol(
    state: QuantAgentState, symbol: str
) -> dict:
    """Macro features (1D) — regime detection — for a single symbol."""
    macro_interval = _get_multi_tf_interval(TimeframeLayer.MACRO)
    context = (
        f"Symbol: {symbol} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{macro_interval}\" — Daily timeframe for regime detection"
    )
    node_config = NODE_CONFIG["feature_engineering"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("feature_engineering"), _FEATURE_SYSTEM, context,
        force_interval=macro_interval,
        node_name=f"FeatureEngineering-Macro[{symbol}]",
        enable_cot=node_config["cot"],
    )

    asset = state.get_asset(symbol)
    macro_rsi_14 = asset.macro_rsi_14
    macro_macd_line = asset.macro_macd_line
    macro_macd_signal = asset.macro_macd_signal
    macro_bb_upper = asset.macro_bb_upper
    macro_bb_lower = asset.macro_bb_lower
    macro_bb_pct_b = asset.macro_bb_pct_b
    macro_regime = asset.macro_regime
    live_price = asset.live_price  # populated by market_data upstream

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
                if parsed.upper and parsed.lower and live_price:
                    bandwidth = parsed.upper - parsed.lower
                    if abs(bandwidth) > 1e-8:
                        macro_bb_pct_b = (live_price - parsed.lower) / bandwidth
                        macro_bb_pct_b = max(0.0, min(1.0, macro_bb_pct_b))
        except Exception:
            continue

    if macro_regime == "overbought":
        macro_bias = "bullish_stretched"
    elif macro_regime == "oversold":
        macro_bias = "bearish_stretched"
    elif macro_regime == "neutral":
        macro_bias = "neutral"
    else:
        macro_bias = None

    print(
        f"[DEBUG] [{symbol}] Macro features - rsi: {macro_rsi_14}, "
        f"macd: {macro_macd_line}, regime: {macro_regime}, bias: {macro_bias}"
    )
    return {
        "symbol": symbol,
        "patch": dict(
            macro_rsi_14=macro_rsi_14,
            macro_macd_line=macro_macd_line,
            macro_macd_signal=macro_macd_signal,
            macro_bb_upper=macro_bb_upper,
            macro_bb_lower=macro_bb_lower,
            macro_bb_pct_b=macro_bb_pct_b,
            macro_regime=macro_regime,
            macro_bias=macro_bias,
        ),
        "steps": steps,
        "cot": cot,
        "content": content,
    }


@timed_async("Node: FeatureEngineering-Macro")
async def features_macro_node(state: QuantAgentState) -> dict:
    """Macro layer (1D) — fan-out per symbol."""
    print(f"[DEBUG] ===== FEATURES MACRO NODE START (1D) =====")
    symbols = _resolve_symbols(state)
    print(f"[DEBUG] Input symbols: {symbols}")

    if not symbols:
        return {
            **state.model_dump(),
            "next_action": NextAction.FEATURES_SETUP,
            "intermediate_steps_agent": [],
            "cot": "",
        }

    results = await asyncio.gather(
        *[_run_features_macro_for_symbol(state, s) for s in symbols]
    )

    new_assets = dict(state.assets)
    all_steps: list = []
    primary_cot = ""
    primary_content = ""
    for r in results:
        sym = r["symbol"]
        current = new_assets.get(sym) or AssetState(symbol=sym)
        new_assets[sym] = current.model_copy(update=r["patch"])
        all_steps.extend(_tag_steps(r["steps"], sym))
        if sym == state.symbol:
            primary_cot = r["cot"]
            primary_content = r["content"]

    print(f"[DEBUG] ===== FEATURES MACRO NODE END =====")
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=primary_content)],
        "next_action": NextAction.FEATURES_SETUP,
        "intermediate_steps_global": state.intermediate_steps_global + all_steps,
        "intermediate_steps_agent": all_steps,
        "assets": new_assets,
        "cot": primary_cot,
        "reasoning_trail": _append_reasoning(state.reasoning_trail, "features_macro", primary_cot),
    }


async def _run_features_setup_for_symbol(
    state: QuantAgentState, symbol: str
) -> dict:
    """Setup features (4H) — signal generation — for a single symbol."""
    setup_interval = _get_multi_tf_interval(TimeframeLayer.SETUP)
    context = (
        f"Symbol: {symbol} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{setup_interval}\" — 4H timeframe for signal generation"
    )
    node_config = NODE_CONFIG["feature_engineering"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("feature_engineering"), _FEATURE_SYSTEM, context,
        force_interval=setup_interval,
        node_name=f"FeatureEngineering-Setup[{symbol}]",
        enable_cot=node_config["cot"],
    )

    asset = state.get_asset(symbol)
    setup_rsi_14 = asset.setup_rsi_14
    setup_macd_line = asset.setup_macd_line
    setup_macd_signal = asset.setup_macd_signal
    setup_bb_upper = asset.setup_bb_upper
    setup_bb_lower = asset.setup_bb_lower
    setup_bb_pct_b = asset.setup_bb_pct_b
    live_price = asset.live_price

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
                if parsed.upper and parsed.lower and live_price:
                    bandwidth = parsed.upper - parsed.lower
                    if abs(bandwidth) > 1e-8:
                        setup_bb_pct_b = (live_price - parsed.lower) / bandwidth
                        setup_bb_pct_b = max(0.0, min(1.0, setup_bb_pct_b))
        except Exception:
            continue

    print(f"[DEBUG] [{symbol}] Setup features - rsi: {setup_rsi_14}, macd: {setup_macd_line}")
    return {
        "symbol": symbol,
        "patch": dict(
            setup_rsi_14=setup_rsi_14,
            setup_macd_line=setup_macd_line,
            setup_macd_signal=setup_macd_signal,
            setup_bb_upper=setup_bb_upper,
            setup_bb_lower=setup_bb_lower,
            setup_bb_pct_b=setup_bb_pct_b,
        ),
        "steps": steps,
        "cot": cot,
        "content": content,
    }


@timed_async("Node: FeatureEngineering-Setup")
async def features_setup_node(state: QuantAgentState) -> dict:
    """Setup layer (4H) — fan-out per symbol."""
    print(f"[DEBUG] ===== FEATURES SETUP NODE START (4H) =====")
    symbols = _resolve_symbols(state)
    print(f"[DEBUG] Input symbols: {symbols}")

    if not symbols:
        return {
            **state.model_dump(),
            "next_action": NextAction.FEATURES_EXEC,
            "intermediate_steps_agent": [],
            "cot": "",
        }

    results = await asyncio.gather(
        *[_run_features_setup_for_symbol(state, s) for s in symbols]
    )

    new_assets = dict(state.assets)
    all_steps: list = []
    primary_cot = ""
    primary_content = ""
    for r in results:
        sym = r["symbol"]
        current = new_assets.get(sym) or AssetState(symbol=sym)
        new_assets[sym] = current.model_copy(update=r["patch"])
        all_steps.extend(_tag_steps(r["steps"], sym))
        if sym == state.symbol:
            primary_cot = r["cot"]
            primary_content = r["content"]

    print(f"[DEBUG] ===== FEATURES SETUP NODE END =====")
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=primary_content)],
        "next_action": NextAction.FEATURES_EXEC,
        "intermediate_steps_global": state.intermediate_steps_global + all_steps,
        "intermediate_steps_agent": all_steps,
        "assets": new_assets,
        "cot": primary_cot,
        "reasoning_trail": _append_reasoning(state.reasoning_trail, "features_setup", primary_cot),
    }


async def _run_features_exec_for_symbol(
    state: QuantAgentState, symbol: str
) -> dict:
    """Exec features (1H) — execution timing — for a single symbol."""
    exec_interval = _get_multi_tf_interval(TimeframeLayer.EXECUTION)
    context = (
        f"Symbol: {symbol} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{exec_interval}\" — 1H timeframe for execution timing"
    )
    node_config = NODE_CONFIG["feature_engineering"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("feature_engineering"), _FEATURE_SYSTEM, context,
        force_interval=exec_interval,
        node_name=f"FeatureEngineering-Exec[{symbol}]",
        enable_cot=node_config["cot"],
    )

    asset = state.get_asset(symbol)
    exec_rsi_14 = asset.exec_rsi_14
    exec_macd_line = asset.exec_macd_line
    exec_macd_signal = asset.exec_macd_signal

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

    print(f"[DEBUG] [{symbol}] Exec features - rsi: {exec_rsi_14}")
    return {
        "symbol": symbol,
        "patch": dict(
            exec_rsi_14=exec_rsi_14,
            exec_macd_line=exec_macd_line,
            exec_macd_signal=exec_macd_signal,
        ),
        "steps": steps,
        "cot": cot,
        "content": content,
    }


@timed_async("Node: FeatureEngineering-Exec")
async def features_exec_node(state: QuantAgentState) -> dict:
    """Execution layer (1H) — fan-out per symbol."""
    print(f"[DEBUG] ===== FEATURES EXEC NODE START (1H) =====")
    symbols = _resolve_symbols(state)
    print(f"[DEBUG] Input symbols: {symbols}")

    if not symbols:
        return {
            **state.model_dump(),
            "next_action": NextAction.FORECAST,
            "intermediate_steps_agent": [],
            "cot": "",
        }

    results = await asyncio.gather(
        *[_run_features_exec_for_symbol(state, s) for s in symbols]
    )

    new_assets = dict(state.assets)
    all_steps: list = []
    primary_cot = ""
    primary_content = ""
    for r in results:
        sym = r["symbol"]
        current = new_assets.get(sym) or AssetState(symbol=sym)
        new_assets[sym] = current.model_copy(update=r["patch"])
        all_steps.extend(_tag_steps(r["steps"], sym))
        if sym == state.symbol:
            primary_cot = r["cot"]
            primary_content = r["content"]

    print(f"[DEBUG] ===== FEATURES EXEC NODE END =====")
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=primary_content)],
        "next_action": NextAction.FORECAST,
        "intermediate_steps_global": state.intermediate_steps_global + all_steps,
        "intermediate_steps_agent": all_steps,
        "assets": new_assets,
        "cot": primary_cot,
        "reasoning_trail": _append_reasoning(state.reasoning_trail, "features_exec", primary_cot),
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


async def _run_forecast_for_symbol(
    state: QuantAgentState, symbol: str
) -> dict:
    """Apollo forecast for a single symbol with safe training/backtest fallback.

    Returns a dict with keys: ``symbol``, ``patch`` (AssetState fields),
    ``audit_step`` (label, message) for the global trail, ``message``
    (AIMessage content for the primary symbol).
    """
    import logging
    logger = logging.getLogger(__name__)

    print(f"[FORECAST] [{symbol}] Iniciando", flush=True)
    settings = get_settings()
    client = get_apollo_client()
    window = build_prediction_window(
        lookback_days=settings.apollo_prediction_lookback_days,
    )

    asset = state.get_asset(symbol)
    raw_symbol = (symbol or "").strip().upper()
    if not raw_symbol:
        warning = "forecast pulado: símbolo não identificado"
        print(f"[FORECAST] ⚠️  {warning}", flush=True)
        logger.warning(f"[FORECAST] {warning}")
        return {
            "symbol": symbol,
            "patch": dict(
                forecast_period_start=window.start_date,
                forecast_period_end=window.end_date,
                forecast_status="unavailable",
                forecast_warnings=list(asset.forecast_warnings) + [warning],
                forecast_actionable=False,
            ),
            "audit_step": ("apollo_predict", warning),
            "message": "Apollo forecast indisponível: símbolo não identificado.",
        }

    apollo_symbol = raw_symbol if raw_symbol.endswith("USDT") else f"{raw_symbol}USDT"
    print(f"[FORECAST] [{symbol}] Apollo predict: symbol={apollo_symbol} | start={window.start_date} | end={window.end_date}", flush=True)

    warnings = list(asset.forecast_warnings)
    training_attempts = 0
    backtest_error_pct: float | None = None
    prediction: ApolloPredictionOutput | None = None
    status = "ready"
    action_step = "apollo_predict"

    try:
        logger.info(f"[FORECAST] [{symbol}] Chamando Apollo predict para {apollo_symbol}")
        prediction = await client.predict(
            symbol=apollo_symbol,
            start_date=window.start_date,
            end_date=window.end_date,
        )
        logger.info(f"[FORECAST] [{symbol}] ✅ Predict retornou: {prediction.direction} com {prediction.confidence:.1%} confiança")
    except ApolloApiError as exc:
        logger.error(f"[FORECAST] [{symbol}] ❌ Erro Apollo: {str(exc)} (status: {exc.status_code}, missing_model: {exc.missing_model})")
        if not exc.missing_model:
            warnings.append(f"forecast indisponível: {exc}")
            status = "unavailable"
        else:
            status = "training_required"
            action_step = "apollo_train"
            logger.warning(f"[FORECAST] [{symbol}] Modelo não encontrado, iniciando treinamento...")
            for attempt in range(1, settings.apollo_train_max_attempts + 1):
                try:
                    training_attempts = attempt
                    logger.info(f"[FORECAST] [{symbol}] Tentativa de treinamento {attempt}/{settings.apollo_train_max_attempts}")
                    train_result = await client.train(
                        symbol=apollo_symbol,
                        lookback_days=settings.apollo_train_lookback_days,
                    )
                    warnings.append(
                        f"treino Apollo iniciado (tentativa {attempt}/{settings.apollo_train_max_attempts}): {train_result.status}"
                    )
                    backtest_result, polls = await _run_apollo_backtest_with_polling(apollo_symbol)
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
                            symbol=apollo_symbol,
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
            "symbol": symbol,
            "patch": dict(
                forecast_period_start=window.start_date,
                forecast_period_end=window.end_date,
                forecast_training_attempts=training_attempts,
                forecast_backtest_error_pct=backtest_error_pct,
                forecast_status=status,
                forecast_warnings=warnings,
                forecast_actionable=False,
            ),
            "audit_step": (
                action_step,
                status if not warnings else " | ".join(warnings[-3:]),
            ),
            "message": "Apollo forecast indisponível ou não confiável.",
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
    logger.info(
        f"[FORECAST] [{symbol}] ✅ status: {status} | actionable: {quality.actionable} | warnings: {len(warnings)}"
    )

    return {
        "symbol": symbol,
        "patch": dict(
            forecast_period_start=prediction.period_start,
            forecast_period_end=prediction.period_end,
            forecast_data_points=prediction.data_points,
            forecast_current_price=prediction.current_price,
            forecast_predicted_price=prediction.predicted_price,
            forecast_direction=prediction.direction,
            forecast_confidence=prediction.confidence,
            forecast_model_mape=model_mape,
            forecast_period_volatility=prediction.confidence_explanation.period_volatility,
            forecast_data_quality=data_quality,
            forecast_return_pct=forecast_return_pct,
            forecast_backtest_error_pct=backtest_error_pct,
            forecast_training_attempts=training_attempts,
            forecast_status=status,
            forecast_actionable=quality.actionable,
            forecast_warnings=warnings,
        ),
        "audit_step": (action_step, summary),
        "message": summary,
    }


@timed_async("Node: Forecast")
async def forecast_node(state: QuantAgentState) -> dict:
    """Apollo forecast — fan-out per symbol.

    Per user choice (option 2 in the plan), the forecast is run for ALL
    symbols (primary + comparatives), in parallel via asyncio.gather. Cost
    and latency scale linearly with the number of symbols.
    """
    print(f"[FORECAST] ===== FORECAST NODE START =====")
    symbols = _resolve_symbols(state)
    print(f"[FORECAST] Symbols: {symbols}")

    if not symbols:
        return {
            **state.model_dump(),
            "next_action": NextAction.TREND_INTERPRET,
            "intermediate_steps_agent": [],
            "cot": "",
        }

    results = await asyncio.gather(
        *[_run_forecast_for_symbol(state, s) for s in symbols],
        return_exceptions=True,
    )

    new_assets = dict(state.assets)
    audit_steps: list = []
    primary_message = ""
    for s, r in zip(symbols, results):
        if isinstance(r, Exception):
            # Defensive: never let a forecast error crash the whole pipeline.
            err_msg = f"forecast crashed: {type(r).__name__}: {r}"
            print(f"[FORECAST] [{s}] ⚠️  {err_msg}")
            current = new_assets.get(s) or AssetState(symbol=s)
            new_assets[s] = current.model_copy(update=dict(
                forecast_status="error",
                forecast_warnings=list(current.forecast_warnings) + [err_msg],
                forecast_actionable=False,
            ))
            audit_steps.append((f"apollo_predict@{s}", err_msg))
            continue

        sym = r["symbol"]
        current = new_assets.get(sym) or AssetState(symbol=sym)
        new_assets[sym] = current.model_copy(update=r["patch"])
        label, msg = r["audit_step"]
        audit_steps.append((f"{label}@{sym}", msg))
        if sym == state.symbol:
            primary_message = r["message"]

    print(f"[FORECAST] ===== FORECAST NODE END =====")
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=primary_message)],
        "next_action": NextAction.TREND_INTERPRET,
        "intermediate_steps_global": state.intermediate_steps_global + audit_steps,
        "assets": new_assets,
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
    
    new_assets = _patch_asset(
        state.assets, state.symbol,
        trend_state=interpretation['trend_state'],
        trend_direction=interpretation['signal_direction'],
        signal_type=interpretation['signal_type'],
        execution_timing=interpretation['execution_timing'],
        signal_direction=interpretation['signal_direction'],  # Legacy field for compatibility
        signal_confidence=interpretation['confidence'],  # Legacy field for compatibility
    )

    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=interpretation['reasoning'])],
        "next_action": NextAction.SIGNAL,  # Route to signal_agent (then moe → decision_engine)
        "intermediate_steps_global": state.intermediate_steps_global + [("trend_interpret", interpretation['reasoning'])],
        "assets": new_assets,
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
    
    new_assets = _patch_asset(
        state.assets, state.symbol,
        final_signal=decision.signal.value,
        final_signal_type=decision.signal_type.value,
        final_confidence=decision.confidence,
        final_reasoning=decision.reasoning,
        # Override legacy fields with final decision
        signal_direction=decision.signal.value,
        signal_confidence=decision.confidence,
    )

    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=decision.reasoning)],
        "next_action": NextAction.RISK,  # Route to risk after decision
        "intermediate_steps_global": state.intermediate_steps_global + [("decision_engine", decision.reasoning)],
        "assets": new_assets,
        "cot": decision.reasoning,
    }


@timed_async("Node: ForecastQuestion")
async def forecast_question_node(state: QuantAgentState) -> dict:
    """Responde perguntas simples sobre previsões futuras do Apollo."""
    user_question = (state.messages[-1].content if state.messages else "").lower()

    # Validação: rejeita perguntas sobre preço ATUAL
    current_price_keywords = [
        "agora", "atualmente", "neste momento", "no momento", "está em",
        "qual é o preço", "como está", "currently", "now", "qual o preço atual"
    ]
    if any(kw in user_question for kw in current_price_keywords):
        return {
            **state.model_dump(),
            "final_answer": "Essa é uma pergunta sobre preço atual. O Apollo é especializado em previsões futuras (amanhã, próximos dias/semanas). Para preço em tempo real, use dados de mercado diretos.",
            "messages": state.messages + [AIMessage(content="Essa é uma pergunta sobre preço atual. O Apollo é especializado em previsões futuras (amanhã, próximos dias/semanas). Para preço em tempo real, use dados de mercado diretos.")],
        }

    if not state.forecast_confidence and state.forecast_status not in ["ready_existing_model", "trained_and_validated"]:
        fallback = f"A previsão do Apollo não está disponível: {state.forecast_status}. "
        fallback += "Verifique a disponibilidade do serviço Apollo e tente novamente."
        return {
            **state.model_dump(),
            "final_answer": fallback,
        }

    settings = get_settings()

    def _f(value, fmt: str = "{:.2f}", suffix: str = "") -> str:
        if value is None:
            return "(indisponível)"
        try:
            return fmt.format(value) + suffix
        except (ValueError, TypeError):
            return str(value)

    confidence_str = (
        f"{state.forecast_confidence * 100:.1f}%"
        if state.forecast_confidence is not None else "(indisponível)"
    )
    backtest_err_str = _f(state.forecast_backtest_error_pct, "{:.2f}", "%")

    forecast_summary = f"""
PREVISÃO APOLLO:
- Símbolo: {state.symbol}
- Janela: {state.forecast_period_start or '(indisponível)'} → {state.forecast_period_end or '(indisponível)'}
- Direção: {state.forecast_direction or '(indisponível)'}
- Preço Atual: {_f(state.forecast_current_price, '${:,.2f}')}
- Preço Previsto: {_f(state.forecast_predicted_price, '${:,.2f}')}
- Retorno Esperado: {_f(state.forecast_return_pct, '{:.2f}', '%')}
- Confiança: {confidence_str}
- MAPE do Modelo: {_f(state.forecast_model_mape, '{:.2f}', '%')}
- Erro do Backtest (p5): {backtest_err_str}
- Qualidade dos Dados: {state.forecast_data_quality or '(indisponível)'}
- Volatilidade do Período: {_f(state.forecast_period_volatility, '{:.2f}', '%')}
- Data Points: {state.forecast_data_points if state.forecast_data_points is not None else '(indisponível)'}
- Acionável: {state.forecast_actionable}
- Status: {state.forecast_status}
- Avisos: {'; '.join(state.forecast_warnings) if state.forecast_warnings else 'Nenhum'}

CRITÉRIOS DE QUALIDADE (referência):
- Threshold Confiança: >= {settings.apollo_confidence_threshold:.0%}
- Threshold MAPE: <= {settings.apollo_mape_threshold:.2f}%
"""

    user_question_original = state.messages[-1].content if state.messages else ""
    llm = _make_llm(model="gpt-4o-mini", temperature=0.1)

    cot = ""
    try:
        response = await llm.ainvoke([
            {"role": "system", "content": _FORECAST_RESPONSE_SYSTEM},
            {"role": "user", "content": f"{user_question_original}\n\n{forecast_summary}"}
        ])
        raw_content = response.content if hasattr(response, 'content') else str(response)
        cot, final_answer = _extract_cot_and_answer(raw_content)
    except Exception as e:
        final_answer = f"Erro ao processar previsão: {str(e)}"

    return {
        **state.model_dump(),
        "final_answer": final_answer,
        "cot": cot,
        "reasoning_trail": _append_reasoning(state.reasoning_trail, "forecast_question", cot),
        "messages": state.messages + [AIMessage(content=final_answer)],
    }

async def _run_risk_for_symbol(
    state: QuantAgentState, symbol: str
) -> dict:
    """Risk metrics (CVaR, Sharpe, drawdown, volatility) for a single symbol.
    Uses execution layer (1H) — multi-timeframe consistency."""
    asset = state.get_asset(symbol)
    exec_interval = _get_multi_tf_interval(TimeframeLayer.EXECUTION)
    context = (
        f"Symbol: {symbol} | Preço: {asset.live_price} | "
        f"INTERVALO OBRIGATÓRIO: interval=\"{exec_interval}\" — 1H timeframe for risk metrics (execution layer) | "
        f"Vol 24h: {asset.volume_24h} | Variação: {asset.price_change_pct}%\n"
        f"DADOS JÁ COLETADOS: RSI={asset.exec_rsi_14}\n"
        f"SUA TAREFA: Calcular métricas de risco usando as ferramentas disponíveis."
    )
    node_config = NODE_CONFIG["risk_agent"]
    content, cot, tool_msgs, steps = await _run_agent_loop(
        state, _get_llm_for_node("risk_agent"), _RISK_SYSTEM, context, clear_steps=True,
        force_interval=exec_interval,
        node_name=f"RiskAgent[{symbol}]",
        enable_cot=node_config["cot"],
    )

    var_95 = asset.var_95
    cvar_95 = asset.cvar_95
    sharpe = asset.sharpe
    max_drawdown = asset.max_drawdown
    volatility_annualized = asset.volatility_annualized
    volatility_raw = None
    volatility_interval = None
    risk_interval = None

    for tool_name, result in steps:
        try:
            if "error" in result:
                continue
            if tool_name == "calculate_risk":
                parsed = RiskMetricsOutput.model_validate_json(result)
                cvar_95 = parsed.cvar_95
                sharpe = parsed.sharpe
                max_drawdown = parsed.max_drawdown
                volatility_raw = parsed.volatility_20d if parsed.volatility_20d is not None else parsed.volatility_21
                volatility_interval = parsed.interval
                risk_interval = parsed.interval
            elif tool_name == "get_feature_cvar":
                parsed = CVaROutput.model_validate_json(result)
                cvar_95 = parsed.cvar_95
                risk_interval = parsed.interval
            elif tool_name == "get_feature_sharpe":
                parsed = SharpeOutput.model_validate_json(result)
                sharpe = parsed.sharpe
                risk_interval = parsed.interval
            elif tool_name == "get_feature_max_drawdown":
                parsed = MaxDrawdownOutput.model_validate_json(result)
                max_drawdown = parsed.max_drawdown
                risk_interval = parsed.interval
            elif tool_name == "get_feature_volatility":
                parsed = VolatilityOutput.model_validate_json(result)
                volatility_raw = parsed.volatility_raw
                volatility_annualized = parsed.volatility_annualized
                volatility_interval = parsed.data_interval
                risk_interval = parsed.data_interval
        except (json.JSONDecodeError, KeyError, TypeError, Exception):
            continue

    # Timeframe consistency check: risk metrics must match execution layer (1H)
    expected_interval = _get_multi_tf_interval(TimeframeLayer.EXECUTION)
    if risk_interval and risk_interval != expected_interval:
        print(f"[DEBUG] [{symbol}] WARNING: Risk interval {risk_interval} != expected {expected_interval}")
        cvar_95 = None
        sharpe = None
        max_drawdown = None
        volatility_annualized = None

    # Store execution layer metrics separately
    exec_sharpe = sharpe
    exec_volatility = volatility_annualized

    # ─── SANITY AUDIT ───
    risk_audit_anomalies: list[str] = []

    if max_drawdown is not None:
        if max_drawdown < 0 or max_drawdown > 1:
            risk_audit_anomalies.append(
                f"max_drawdown fora do range [0,1]: {max_drawdown:.4f} — descartado"
            )
            max_drawdown = None
        elif asset.recent_high and asset.recent_low and asset.recent_high > 0:
            observed_range_dd = (asset.recent_high - asset.recent_low) / asset.recent_high
            if max_drawdown > observed_range_dd * 1.5 and max_drawdown > 0.10:
                risk_audit_anomalies.append(
                    f"max_drawdown backend ({max_drawdown:.2%}) >> range observado "
                    f"({observed_range_dd:.2%}) × 1.5 — usando fallback observado"
                )
                max_drawdown = observed_range_dd

    if cvar_95 is not None and (cvar_95 < -1.0 or cvar_95 > 1.0):
        risk_audit_anomalies.append(
            f"cvar_95 fora do range [-1,1]: {cvar_95:.4f} — descartado"
        )
        cvar_95 = None

    if sharpe is not None and abs(sharpe) > 10:
        risk_audit_anomalies.append(
            f"sharpe implausível: {sharpe:.4f} — descartado"
        )
        sharpe = None

    if volatility_annualized is not None and volatility_annualized > 5.0:
        risk_audit_anomalies.append(
            f"volatility_annualized implausível: {volatility_annualized:.2%} — descartado"
        )
        volatility_annualized = None

    risk_level_str = determine_risk_level(
        cvar_95=cvar_95,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        volatility_annualized=volatility_annualized,
    )
    risk_level = RiskLevel(risk_level_str)

    # Annualizar volatilidade quando vier raw
    if volatility_raw is not None and volatility_annualized is None:
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
        if volatility_interval == "1m":
            volatility_annualized = volatility_raw * math.sqrt(252 * 1440)
        elif volatility_interval == "1h":
            volatility_annualized = volatility_raw * math.sqrt(252 * 24)
        elif volatility_interval == "1d":
            volatility_annualized = volatility_raw * math.sqrt(252)
        elif volatility_interval in ["1D", "1w", "1W"]:
            volatility_annualized = volatility_raw * math.sqrt(52)

    print(
        f"[DEBUG] [{symbol}] Risk - cvar: {cvar_95}, sharpe: {sharpe}, "
        f"drawdown: {max_drawdown}, vol: {volatility_annualized}, level: {risk_level}"
    )
    return {
        "symbol": symbol,
        "patch": dict(
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            volatility_annualized=volatility_annualized,
            risk_level=risk_level,
            exec_sharpe=exec_sharpe,
            exec_volatility=exec_volatility,
            anomalies_detected=list(asset.anomalies_detected) + risk_audit_anomalies,
        ),
        "steps": steps,
        "cot": cot,
        "content": content,
    }


@timed_async("Node: RiskAgent")
async def risk_agent_node(state: QuantAgentState) -> dict:
    """Risk metrics — fan-out per symbol. Comparados precisam dessas métricas
    para o execution_node renderizar a tabela comparativa."""
    print(f"[DEBUG] ===== RISK AGENT NODE START =====")
    symbols = _resolve_symbols(state)
    print(f"[DEBUG] Input symbols: {symbols}")

    if not symbols:
        return {
            **state.model_dump(),
            "next_action": NextAction.RISK_GATE,
            "intermediate_steps_agent": [],
            "cot": "",
        }

    results = await asyncio.gather(
        *[_run_risk_for_symbol(state, s) for s in symbols]
    )

    new_assets = dict(state.assets)
    all_steps: list = []
    primary_cot = ""
    primary_content = ""
    for r in results:
        sym = r["symbol"]
        current = new_assets.get(sym) or AssetState(symbol=sym)
        new_assets[sym] = current.model_copy(update=r["patch"])
        all_steps.extend(_tag_steps(r["steps"], sym))
        if sym == state.symbol:
            primary_cot = r["cot"]
            primary_content = r["content"]

    print(f"[DEBUG] ===== RISK AGENT NODE END =====")
    return {
        **state.model_dump(),
        "messages":           state.messages + [AIMessage(content=primary_content)],
        "next_action":        NextAction.RISK_GATE,
        "intermediate_steps_global": state.intermediate_steps_global + all_steps,
        "intermediate_steps_agent": all_steps,
        "assets":             new_assets,
        "cot":                primary_cot,
        "reasoning_trail":    _append_reasoning(state.reasoning_trail, "risk_agent", primary_cot),
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
    new_assets = _patch_asset(
        state.assets, state.symbol,
        signal_direction=direction,
        regime=regime,
        signal_confidence=signal_confidence,
    )
    return {
        **state.model_dump(),
        "messages":           state.messages + [AIMessage(content=content)],
        "next_action":        NextAction.MOE,
        "assets":             new_assets,
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "intermediate_steps_agent": steps,
        "cot":                cot,
        "reasoning_trail":    _append_reasoning(state.reasoning_trail, "signal_agent", cot),
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
    
    new_assets = _patch_asset(
        state.assets, state.symbol,
        moe_final_signal=moe_output.final_signal,
        moe_final_confidence=moe_output.final_confidence,
        moe_selected_experts=[e.value for e in moe_output.selected_experts],
        moe_expert_weights={k.value: v for k, v in moe_output.expert_weights.items()},
        moe_gating_reason=moe_output.gating_reason,
        moe_position_size=moe_output.position_size,
        moe_risk_adjusted_signal=moe_output.risk_adjusted_signal,
    )
    return {
        **state.model_dump(),
        "messages": state.messages + [AIMessage(content=f"MoE signal: {moe_output.final_signal:.3f}, confidence: {moe_output.final_confidence:.3f}")],
        "next_action": NextAction.DECISION_ENGINE,  # MoE feeds into decision_engine (FINAL authority)
        "assets": new_assets,
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


def _coverage_summary_for_asset(asset: AssetState) -> tuple[int, int, list[str]]:
    """Conta quantos campos críticos estão preenchidos para um ativo;
    retorna (ok, total, missing). Usado para anti-alucinação no execution_node."""
    fields = {
        "live_price": asset.live_price,
        "price_change_pct": asset.price_change_pct,
        "rsi_14": asset.rsi_14,
        "macd_line": asset.macd_line,
        "bb_upper": asset.bb_upper,
        "sharpe": asset.sharpe,
        "cvar_95": asset.cvar_95,
        "max_drawdown": asset.max_drawdown,
        "volatility_annualized": asset.volatility_annualized,
    }
    missing = [k for k, v in fields.items() if v is None]
    return len(fields) - len(missing), len(fields), missing


def _render_asset_block(asset: AssetState, *, label: str = "") -> str:
    """Renderiza o bloco completo de dados de um ativo para o contexto do
    execution_node. Inclui market data, indicadores multi-TF, risco e forecast."""
    header = f"━━━ {label or asset.symbol} ━━━"
    return f"""
{header}
DADOS DE MERCADO
  Symbol:          {asset.symbol}
  Preço atual:     {_fmt(asset.live_price, prec=2)}
  Variação 24h:    {_fmt(asset.price_change_pct, prec=4)}%
  Volume 24h:      {_fmt(asset.volume_24h, prec=2)}
  Range recente:   high={_fmt(asset.recent_high, prec=2)} low={_fmt(asset.recent_low, prec=2)}

INDICADORES — MACRO (1D)
  RSI(14):         {_fmt(asset.macro_rsi_14, prec=2)}
  MACD line/signal:{_fmt(asset.macro_macd_line)} / {_fmt(asset.macro_macd_signal)}
  Bollinger %B:    {_fmt(asset.macro_bb_pct_b, prec=2)}
  Regime:          {asset.macro_regime or '(não coletado)'}
  Bias:            {asset.macro_bias or '(não coletado)'}

INDICADORES — SETUP (4H)
  RSI(14):         {_fmt(asset.setup_rsi_14, prec=2)}
  MACD line/signal:{_fmt(asset.setup_macd_line)} / {_fmt(asset.setup_macd_signal)}
  Bollinger %B:    {_fmt(asset.setup_bb_pct_b, prec=2)}

INDICADORES — EXEC (1H)
  RSI(14):         {_fmt(asset.exec_rsi_14, prec=2)}
  MACD line/signal:{_fmt(asset.exec_macd_line)} / {_fmt(asset.exec_macd_signal)}

MÉTRICAS DE RISCO
  Risk level:      {asset.risk_level}
  CVaR 95%:        {_fmt(asset.cvar_95, pct=True)}
  Sharpe:          {_fmt(asset.sharpe, prec=3)}
  Max Drawdown:    {_fmt(asset.max_drawdown, pct=True)}
  Vol. anualizada: {_fmt(asset.volatility_annualized, pct=True)}

SINAL & DECISÃO
  Regime:          {asset.regime or '(não coletado)'}
  Direção:         {asset.signal_direction or '(não coletado)'}
  Confiança:       {_fmt(asset.signal_confidence, pct=True)}
  Trend state:     {asset.trend_state or '(não coletado)'}
  Final signal:    {asset.final_signal or '(não coletado)'} ({asset.final_signal_type or '—'})
  Final reasoning: {asset.final_reasoning or '(não coletado)'}

FORECAST APOLLO
  Status:          {asset.forecast_status}
  Janela:          {asset.forecast_period_start or '(não coletado)'} → {asset.forecast_period_end or '(não coletado)'}
  Current price:   {_fmt(asset.forecast_current_price, prec=2)}
  Predicted price: {_fmt(asset.forecast_predicted_price, prec=2)}
  Direção ML:      {asset.forecast_direction or '(não coletado)'}
  Retorno esperado:{_fmt(asset.forecast_return_pct, prec=2)}%
  Confiança ML:    {_fmt(asset.forecast_confidence, pct=True)}
  MAPE modelo:     {_fmt(asset.forecast_model_mape, prec=2)}%
  Acionável:       {asset.forecast_actionable}
  Avisos:          {asset.forecast_warnings or 'Nenhum'}

QUALIDADE DOS DADOS
  Anomalias:       {asset.anomalies_detected or 'Nenhuma'}
""".rstrip()


@timed_async("Node: Execution")
async def execution_node(state: QuantAgentState) -> dict:
    """
    Consolida a análise e gera a resposta final orientada à PERGUNTA do usuário.

    Multi-asset aware:
      • Renderiza dados de TODOS os ativos em `state.assets` (primário + comparativos).
      • Inclui a pergunta original do usuário no topo do contexto.
      • Promove o `reasoning_trail` a espinha dorsal antes dos dados estruturados.
      • Deixa o LLM escolher entre modo comparativo / single / pergunta específica
        com base na pergunta — sem template forçado.
    """
    # VALIDAÇÃO FINAL OBRIGATÓRIA — gate é sobre o primário
    if not state.gate_approved:
        raise Exception(f"Execution node chamado sem aprovação do risk gate. Motivo: {state.gate_reason}")
    if not state.symbol:
        raise Exception("Symbol não definido no estado.")
    if state.signal_direction is None:
        raise Exception("Signal direction não definido no estado.")
    if state.risk_level is None:
        raise Exception("Risk level não definido no estado.")

    # ──────────── 1) Pergunta original do usuário (bússola da resposta) ────────────
    original_query = next(
        (m for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        None,
    )
    user_question_text = original_query.content if original_query else "(não disponível)"

    # ──────────── 2) Reasoning trail — espinha dorsal ────────────
    if state.reasoning_trail:
        trail_block = "\n".join(
            f"  [{name}] {thought}" for name, thought in state.reasoning_trail
        )
    else:
        trail_block = "  (nenhum)"

    # ──────────── 3) Dados estruturados — 1 bloco por ativo ────────────
    symbols = _resolve_symbols(state)
    asset_blocks: list[str] = []
    for sym in symbols:
        asset = state.get_asset(sym)
        label = f"{sym} (PRIMÁRIO — foco da decisão)" if sym == state.symbol else f"{sym} (comparativo)"
        asset_blocks.append(_render_asset_block(asset, label=label))
    asset_data_section = "\n\n".join(asset_blocks)

    # Cobertura é avaliada sobre o primário
    primary_asset = state.primary
    ok, total, missing = _coverage_summary_for_asset(primary_asset)
    coverage_line = f"Cobertura ({state.symbol}): {ok}/{total} campos preenchidos"
    if missing:
        coverage_line += f" | AUSENTES: {', '.join(missing)}"

    multi_asset_hint = ""
    if len(symbols) > 1:
        multi_asset_hint = (
            f"\n⚠️  CONTEXTO MULTI-ATIVO: a pergunta envolve {len(symbols)} ativos "
            f"({', '.join(symbols)}). Se a pergunta é comparativa, use o MODO COMPARATIVO."
        )

    context = f"""
PERGUNTA ORIGINAL DO USUÁRIO (responda esta pergunta — não force template):
  "{user_question_text}"
{multi_asset_hint}

RACIOCÍNIO ACUMULADO DOS AGENTES (use como espinha dorsal, sintetize — não copie literalmente):
{trail_block}

═══════════════════════════════════════════════════════════
DADOS POR ATIVO
═══════════════════════════════════════════════════════════
{asset_data_section}

═══════════════════════════════════════════════════════════
COBERTURA & RISK GATE (sobre o primário {state.symbol})
═══════════════════════════════════════════════════════════
  {coverage_line}
  Risk gate aprovado:  {state.gate_approved}
  Risk gate motivo:    {state.gate_reason}
  Timeframe da análise:{state.timeframe}

HISTÓRICO DE ANÁLISE (últimos 12 steps, para auditoria):
{chr(10).join(f'  [{k}] {v[:120]}' for k, v in state.intermediate_steps_global[-12:])}
""".strip()

    msgs = [
        SystemMessage(content=_EXECUTION_SYSTEM),
        HumanMessage(content=context),
    ]
    node_config = NODE_CONFIG["execution"]
    llm = _get_llm_for_node("execution")

    if node_config["cot"]:
        msgs.append(HumanMessage(content=(
            "\n\nFORMATO DE RESPOSTA OBRIGATÓRIO:\n"
            "<thought>Resumo CONCISO do seu raciocínio (max 2-3 frases). "
            "Ex: 'Pergunta comparativa entre X e Y; X tem Sharpe maior, Y tem momentum melhor; vereditc X.'</thought>\n"
            "<answer>Sua resposta final aqui</answer>"
        )))

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
    cost_summary = cost_tracker.get_summary()

    cot, answer = _extract_cot_and_answer(final_response.content) if node_config["cot"] else ("", final_response.content)

    return {
        **state.model_dump(),
        "messages":     state.messages + [final_response],
        "next_action":  NextAction.FINALIZE,
        "final_answer": answer,
        "cost_summary": cost_summary,
        "cot":          cot,
        "reasoning_trail": _append_reasoning(state.reasoning_trail, "execution", cot),
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

# Roteamento condicional após o nó `forecast`. Mantido em nível de módulo
# (não como closure dentro de `build_quant_graph`) para permitir testes
# diretos e travar a regressão em que o pipeline lia `messages[-1]`
# (uma AIMessage upstream) em vez da pergunta original do usuário.
_FORECAST_FUTURE_KEYWORDS: tuple[str, ...] = (
    "como vai", "como estara", "preve", "forecast", "quando", "proximo",
    "amanha", "tomorrow", "predict", "previsao", "vai estar", "estara",
    "qual e a previsao", "proximas", "semana que vem", "proximo mes",
    "futuro", "future",
)

_FORECAST_CURRENT_PRICE_KEYWORDS: tuple[str, ...] = (
    "agora", "atualmente", "neste momento", "no momento", "esta em",
    "qual e o preco", "como esta", "currently", "now",
)


def route_after_forecast(state: QuantAgentState) -> str:
    """Decide se a pergunta corrente do usuário pede previsão futura
    (`forecast_question`) ou contexto/preço atual (`trend_interpret`).

    Olha a ÚLTIMA `HumanMessage` em `state.messages` (não a última
    mensagem absoluta — esta já é uma `AIMessage` produzida pelos
    nós upstream do pipeline; nem a primeira `HumanMessage` —
    em conversas multi-turno o `chat.py` injeta histórico/contexto
    como `HumanMessage` antes da pergunta corrente).

    Texto é normalizado (lowercase, sem acentos, sem pontuação)
    antes de comparar com as keywords, para tolerar variações como
    "amanhã"/"amanha", "previsão"/"previsao".
    """
    original = next(
        (m for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        None,
    )
    msg = _normalize_text(original.content if original else "")

    if any(kw in msg for kw in _FORECAST_CURRENT_PRICE_KEYWORDS):
        return "trend_interpret"

    if any(kw in msg for kw in _FORECAST_FUTURE_KEYWORDS):
        return "forecast_question"

    return "trend_interpret"


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

    # Conditional: orchestrator pode finalizar cedo (saudação / out-of-scope / educação)
    # ou prosseguir para o pipeline completo. RESPEITA o next_action setado pelo nó.
    def route_after_orchestrator(state: QuantAgentState) -> str:
        # Se o orchestrator já produziu uma resposta final (saudação/escopo/educação),
        # vai direto para finalize — não roda o pipeline de análise.
        if state.next_action == NextAction.FINALIZE:
            return "finalize"
        return "market_data"

    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "finalize": "finalize",
            "market_data": "market_data",
        },
    )

    # Fixed edges (multi-timeframe pipeline with decision engine as FINAL authority)
    workflow.add_edge("market_data",  "features_macro")   # 1D for regime
    workflow.add_edge("features_macro", "features_setup")  # 4H for signal
    workflow.add_edge("features_setup", "features_exec")   # 1H for execution
    workflow.add_edge("features_exec",  "forecast")        # Apollo forecast

    # Conditional: forecast → question answering or full analysis
    workflow.add_conditional_edges(
        "forecast",
        route_after_forecast,
        {
            "forecast_question": "forecast_question",
            "trend_interpret": "trend_interpret",
        },
    )

    # Pipeline: trend_interpret → signal → moe → decision_engine → risk → risk_gate
    # Rationale: signal/moe alimentam o decision_engine (FINAL authority) com
    # moe_final_signal/confidence ANTES dele decidir. Assim o decision_engine
    # realmente tem a palavra final e não é sobrescrito por nós downstream.
    workflow.add_edge("trend_interpret", "signal")        # Signal agent: alpha generation
    workflow.add_edge("signal",         "moe")            # Mixture of Experts: combine signals
    workflow.add_edge("moe",            "decision_engine") # Decision engine (FINAL authority)
    workflow.add_edge("decision_engine","risk")           # Risk uses final decision
    workflow.add_edge("risk",           "risk_gate")       # Risk gate: pre-trade checks

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
