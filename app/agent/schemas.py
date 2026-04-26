"""
Structured Output Schemas for Quant Agents
──────────────────────────────────────────
Pydantic models for type-safe, deterministic agent outputs.
Eliminates regex parsing and natural language dependencies.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


# ─────────────────────────────────────────────
# Tool Output Schemas
# ─────────────────────────────────────────────

class LivePriceOutput(BaseModel):
    """Structured output for get_live_price tool."""
    symbol: str
    interval: str
    close: float
    high: float
    low: float
    volume: float
    timestamp: int
    _timing: Optional[dict] = None


class IndicatorsOutput(BaseModel):
    """Structured output for get_indicators tool."""
    symbol: str
    interval: str
    candle_count: int
    current_price: float
    pct_change: float
    total_volume: float
    high: float
    low: float
    first_timestamp: int
    last_timestamp: int
    _timing: Optional[dict] = None


class RSIOutput(BaseModel):
    """Structured output for RSI feature."""
    symbol: str
    interval: str
    rsi_14: float
    regime: Literal["overbought", "oversold", "neutral"]
    _timing: Optional[dict] = None


class MACDOutput(BaseModel):
    """Structured output for MACD feature."""
    symbol: str
    interval: str
    macd_line: float
    signal: float
    histogram: float
    crossover: Optional[Literal["bullish", "bearish", "none"]] = None
    _timing: Optional[dict] = None


class BollingerBandsOutput(BaseModel):
    """Structured output for Bollinger Bands feature."""
    symbol: str
    interval: str
    upper: float
    middle: float
    lower: float
    pct_b: float  # Position within bands (0-1)
    width: Optional[float] = None
    breakout: Optional[bool] = None  # True if price is outside the bands
    available: Optional[bool] = None
    anomalies: Optional[list] = None
    _timing: Optional[dict] = None


class VolatilityOutput(BaseModel):
    """Structured output for volatility feature."""
    symbol: str
    interval: str
    volatility_raw: float
    volatility_annualized: float
    data_interval: str  # "1m", "1h", "1d", etc.
    _timing: Optional[dict] = None


class RiskMetricsOutput(BaseModel):
    """Structured output for risk metrics."""
    symbol: str
    interval: str
    cvar_95: float
    sharpe: float
    max_drawdown: float
    volatility_20d: Optional[float] = None
    volatility_21: Optional[float] = None
    _timing: Optional[dict] = None


class SharpeOutput(BaseModel):
    """Structured output for Sharpe ratio."""
    symbol: str
    interval: str
    sharpe: float
    _timing: Optional[dict] = None


class CVaROutput(BaseModel):
    """Structured output for CVaR 95%."""
    symbol: str
    interval: str
    cvar_95: float
    _timing: Optional[dict] = None


class MaxDrawdownOutput(BaseModel):
    """Structured output for max drawdown."""
    symbol: str
    interval: str
    max_drawdown: float
    _timing: Optional[dict] = None


class SMAOutput(BaseModel):
    """Structured output for SMA."""
    symbol: str
    interval: str
    period: int
    sma: float
    _timing: Optional[dict] = None


class EMAReturnOutput(BaseModel):
    """Structured output for EMA of returns."""
    symbol: str
    interval: str
    ema_return_60: float
    _timing: Optional[dict] = None


class ApolloConfidenceExplanation(BaseModel):
    """Structured output for Apollo confidence explanation."""
    model_mape: Optional[float] = None
    period_volatility: Optional[float] = None
    data_quality: Optional[str] = None


class ApolloPredictionOutput(BaseModel):
    """Structured output for Apollo prediction endpoint."""
    symbol: str
    period_start: str
    period_end: str
    data_points: int
    current_price: float
    predicted_price: float
    direction: Literal["UP", "DOWN", "FLAT"]
    confidence: float
    confidence_explanation: ApolloConfidenceExplanation = Field(default_factory=ApolloConfidenceExplanation)


class ApolloTrainingOutput(BaseModel):
    """Structured output for Apollo training endpoint."""
    message: str
    symbol: str
    lookback_days: int
    status: str
    note: Optional[str] = None


class ApolloBacktestRange(BaseModel):
    """Structured output for Apollo backtest date range."""
    min: str
    max: str


class ApolloBacktestPeriod(BaseModel):
    """Structured output for a single Apollo backtest period."""
    period: str
    start_date: str
    end_date: str
    data_points: int
    current_price: float
    predicted_price: float


class ApolloBacktestOutput(BaseModel):
    """Structured output for Apollo backtest endpoint."""
    symbol: str
    date_range: ApolloBacktestRange
    periods_tested: int
    results: list[ApolloBacktestPeriod] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Agent Output Schemas
# ─────────────────────────────────────────────

class OrchestratorOutput(BaseModel):
    """Structured output for Orchestrator agent."""
    next_agent: Literal["market_data", "features", "risk", "signal", "risk_gate", "execution", "finalize", "blocked"]
    symbol: str
    timeframe: Literal["intraday", "daily", "weekly"]
    context: str = ""


class MarketDataOutput(BaseModel):
    """Structured output for Market Data agent."""
    symbol: str
    timeframe: str
    live_price: float
    volume_24h: float
    price_change_pct: float
    anomalies: list[str] = Field(default_factory=list)
    analysis: str = ""  # LLM narrative analysis


class FeatureEngineeringOutput(BaseModel):
    """Structured output for Feature Engineering agent."""
    symbol: str
    timeframe: str
    rsi_14: float
    rsi_regime: Literal["overbought", "oversold", "neutral"]
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_crossover: Optional[Literal["bullish", "bearish", "none"]] = None
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_pct_b: float
    analysis: str = ""  # LLM narrative analysis


class RiskAgentOutput(BaseModel):
    """Structured output for Risk agent."""
    symbol: str
    risk_level: Literal["low", "moderate", "high", "extreme"]
    var_95: Optional[float] = None
    cvar_95: float
    sharpe: float
    max_drawdown: float
    volatility_annualized: float
    reasoning: str = ""  # LLM narrative analysis


class SignalAgentOutput(BaseModel):
    """Structured output for Signal agent."""
    symbol: str
    regime: Literal["trending", "ranging", "breakout"]
    signal_direction: Literal["long", "short", "neutral"]
    signal_confidence: float  # 0.0 to 1.0
    indicator_convergence: int  # Number of convergent indicators
    reasoning: str = ""  # LLM narrative analysis


class RiskGateOutput(BaseModel):
    """Structured output for Risk Gate."""
    approved: bool
    reason: str
    blocked_by: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExecutionOutput(BaseModel):
    """Structured output for Execution layer."""
    symbol: str
    timeframe: str
    executive_summary: str
    technical_analysis: str
    risk_metrics: str
    signal: str
    sizing_suggestion: Optional[str] = None
    alerts: list[str] = Field(default_factory=list)
    disclaimer: str = "Análise quantitativa — não é recomendação de investimento"
