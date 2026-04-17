"""
Quantitative Engine - Deterministic Calculations
────────────────────────────────────────────────
All technical indicators calculated outside LLM for institutional-grade reliability.
LLM is used only for narrative interpretation, not computation.
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TechnicalIndicators:
    """Container for all technical indicators."""
    rsi_14: Optional[float] = None
    rsi_regime: Optional[str] = None  # "overbought", "oversold", "neutral"
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_crossover: Optional[str] = None  # "bullish", "bearish", "none"
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_pct_b: Optional[float] = None


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility_annualized: Optional[float] = None
    risk_level: Optional[str] = None  # "low", "moderate", "high", "extreme"


@dataclass
class SignalOutput:
    """Container for signal calculation."""
    direction: str  # "long", "short", "neutral"
    confidence: float  # 0.0 to 1.0
    regime: str  # "trending", "ranging", "breakout"
    score: float  # Raw score (-1 to 1)
    indicator_count: int


# ---------- Configurable Parameters ----------
SIGNAL_PARAMS = {
    "long_threshold": 0.3,
    "short_threshold": -0.3,
    "trending_threshold": 0.6,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "weights": {
        "rsi": 0.30,
        "macd": 0.30,
        "bollinger": 0.20,
        "momentum": 0.20,
    },
    "confidence_boost_if_indicators_ge": 3,
    "confidence_boost_factor": 1.2,
    "confidence_penalty_if_indicators_lt": 2,
    "confidence_penalty_factor": 0.7,
    "min_volatility_for_normalization": 0.01,  # to avoid division by zero
    "bollinger_band_tolerance": 1e-8,
}


def calculate_rsi_signal(
    rsi_14: Optional[float],
    oversold: float = 30,
    overbought: float = 70,
) -> Tuple[float, Optional[str]]:
    """
    Calculate RSI signal (-1 to 1) and regime.
    Institutional: deterministic, no LLM interpretation.
    """
    if rsi_14 is None:
        return 0.0, None

    # Clamp RSI to valid range
    rsi_14 = max(0.0, min(100.0, rsi_14))

    if rsi_14 < oversold:
        return 1.0, "oversold"  # Oversold = long signal
    elif rsi_14 > overbought:
        return -1.0, "overbought"  # Overbought = short signal
    else:
        return 0.0, "neutral"


def calculate_macd_signal(
    macd_line: Optional[float], macd_signal: Optional[float]
) -> Tuple[float, Optional[str]]:
    """
    Calculate MACD signal (-1 to 1) and crossover.
    Institutional: deterministic crossover detection.
    """
    if macd_line is None or macd_signal is None:
        return 0.0, None

    if macd_line > macd_signal:
        return 1.0, "bullish"
    elif macd_line < macd_signal:
        return -1.0, "bearish"
    else:
        return 0.0, "none"


def calculate_bollinger_signal(
    bb_upper: Optional[float],
    bb_lower: Optional[float],
    live_price: Optional[float],
    tolerance: float = 1e-8,
) -> Tuple[float, Optional[float]]:
    """
    Calculate Bollinger Bands signal using %B (-1 to 1).
    Institutional: uses %B formula, not fixed thresholds.

    %B = (price - lower) / (upper - lower)
    bb_signal = (percent_b - 0.5) * 2  # Scale to -1 to 1, then inverted.
    """
    if bb_upper is None or bb_lower is None or live_price is None:
        return 0.0, None

    bandwidth = bb_upper - bb_lower
    if abs(bandwidth) < tolerance:
        return 0.0, 0.5  # Degenerate band: treat as neutral

    percent_b = (live_price - bb_lower) / bandwidth
    # Clip percent_b to [0,1] to avoid extreme outliers
    percent_b = max(0.0, min(1.0, percent_b))

    bb_signal = (percent_b - 0.5) * 2  # Normalized to -1 to 1
    bb_signal = max(-1.0, min(1.0, bb_signal))
    bb_signal = -bb_signal  # Invert: high %B = overbought = negative signal

    return bb_signal, percent_b


def calculate_momentum_signal(
    price_change_pct: Optional[float],
    volatility_annualized: Optional[float],
    min_vol: float = 0.01,
    default_vol: float = 5.0,  # 5% as fallback
) -> float:
    """
    Calculate momentum signal using tanh normalization.
    Institutional: smooth sigmoid, not binary.
    """
    if price_change_pct is None:
        return 0.0

    if volatility_annualized is not None and volatility_annualized > 0:
        vol = max(volatility_annualized, min_vol)
        normalized_change = price_change_pct / (vol * 100)  # vol is decimal (e.g., 0.2 = 20%)
    else:
        # Fallback: assume default_vol% volatility
        normalized_change = price_change_pct / default_vol

    return math.tanh(normalized_change)  # Smooth sigmoid between -1 and 1


def compute_signal_score(
    rsi_14: Optional[float],
    macd_line: Optional[float],
    macd_signal: Optional[float],
    bb_upper: Optional[float],
    bb_lower: Optional[float],
    live_price: Optional[float],
    price_change_pct: Optional[float],
    volatility_annualized: Optional[float],
    params: dict = None,
) -> SignalOutput:
    """
    Compute formal signal score using weighted indicators.
    Institutional: deterministic, no LLM interpretation.
    """
    if params is None:
        params = SIGNAL_PARAMS

    weights = params["weights"]
    w_rsi = weights["rsi"]
    w_macd = weights["macd"]
    w_bb = weights["bollinger"]
    w_mom = weights["momentum"]

    signal_score = 0.0
    total_weight_used = 0.0

    # RSI Signal
    rsi_signal_val, rsi_regime = calculate_rsi_signal(
        rsi_14, oversold=params["rsi_oversold"], overbought=params["rsi_overbought"]
    )
    if rsi_14 is not None:
        signal_score += w_rsi * rsi_signal_val
        total_weight_used += w_rsi

    # MACD Signal
    macd_signal_val, macd_crossover = calculate_macd_signal(macd_line, macd_signal)
    if macd_line is not None and macd_signal is not None:
        signal_score += w_macd * macd_signal_val
        total_weight_used += w_macd

    # Bollinger Signal
    bb_signal, pct_b = calculate_bollinger_signal(
        bb_upper, bb_lower, live_price, tolerance=params["bollinger_band_tolerance"]
    )
    if bb_upper is not None and bb_lower is not None and live_price is not None:
        signal_score += w_bb * bb_signal
        total_weight_used += w_bb

    # Momentum Signal
    momentum_signal = calculate_momentum_signal(
        price_change_pct,
        volatility_annualized,
        min_vol=params["min_volatility_for_normalization"],
    )
    if price_change_pct is not None:
        signal_score += w_mom * momentum_signal
        total_weight_used += w_mom

    # Renormalize if some indicators missing
    if total_weight_used > 0:
        signal_score = signal_score / total_weight_used
    else:
        signal_score = 0.0

    # Clip to valid range
    signal_score = max(-1.0, min(1.0, signal_score))

    # Direction thresholds
    long_thresh = params["long_threshold"]
    short_thresh = params["short_threshold"]

    if signal_score > long_thresh:
        direction = "long"
    elif signal_score < short_thresh:
        direction = "short"
    else:
        direction = "neutral"

    # Confidence based on absolute score and indicator count
    confidence = abs(signal_score)
    indicator_count = sum(
        1 for x in [rsi_14, macd_line, bb_upper, price_change_pct] if x is not None
    )

    if indicator_count >= params["confidence_boost_if_indicators_ge"]:
        confidence = min(confidence * params["confidence_boost_factor"], 1.0)
    elif indicator_count < params["confidence_penalty_if_indicators_lt"]:
        confidence = confidence * params["confidence_penalty_factor"]

    # Regime detection
    trending_thresh = params["trending_threshold"]
    if abs(signal_score) > trending_thresh:
        regime = "trending"
    elif abs(signal_score) < long_thresh:  # using long_thresh as neutral band
        regime = "ranging"
    else:
        regime = "breakout"

    return SignalOutput(
        direction=direction,
        confidence=confidence,
        regime=regime,
        score=signal_score,
        indicator_count=indicator_count,
    )


def determine_risk_level(
    cvar_95: Optional[float],
    sharpe: Optional[float],
    max_drawdown: Optional[float],
    volatility_annualized: Optional[float],
) -> str:
    """
    Determine risk level from metrics (deterministic).
    CVaR and drawdown are positive numbers (loss magnitudes).
    Lower Sharpe ratio indicates higher risk.
    """
    risk_score = 0

    # CVaR contribution (expected loss, e.g., 0.15 = 15%)
    if cvar_95 is not None and cvar_95 > 0:
        if cvar_95 > 0.25:
            risk_score += 3
        elif cvar_95 > 0.15:
            risk_score += 2
        elif cvar_95 > 0.10:
            risk_score += 1

    # Sharpe ratio (lower or negative = higher risk)
    if sharpe is not None:
        if sharpe < 0:
            risk_score += 2
        elif sharpe < 0.5:
            risk_score += 1

    # Maximum drawdown (positive loss magnitude)
    if max_drawdown is not None and max_drawdown > 0:
        if max_drawdown > 0.30:
            risk_score += 3
        elif max_drawdown > 0.20:
            risk_score += 2
        elif max_drawdown > 0.10:
            risk_score += 1

    # Annualized volatility (decimal, e.g., 0.5 = 50%)
    if volatility_annualized is not None and volatility_annualized > 0:
        if volatility_annualized > 1.0:
            risk_score += 2
        elif volatility_annualized > 0.5:
            risk_score += 1

    # Map score to risk level
    if risk_score >= 6:
        return "extreme"
    elif risk_score >= 4:
        return "high"
    elif risk_score >= 2:
        return "moderate"
    else:
        return "low"


def calculate_position_size(
    signal_confidence: float,
    volatility_annualized: Optional[float],
    max_position_size: float = 0.10,
    min_position_size: float = 0.01,
    strong_confidence_threshold: float = 0.7,
) -> float:
    """
    Calculate position size based on signal confidence and volatility.
    Institutional: volatility-adjusted sizing.
    """
    # Validate inputs
    signal_confidence = max(0.0, min(1.0, signal_confidence))
    if signal_confidence <= 0:
        return 0.0

    if volatility_annualized is None or volatility_annualized <= 0:
        vol_adjustment = 1.0
    else:
        # Volatility adjustment: higher vol = smaller position, cap at 1.0
        vol_adjustment = min(1.0, 0.5 / volatility_annualized)

    position_size = max_position_size * signal_confidence * vol_adjustment

    # Enforce minimum only if confidence is strong
    if signal_confidence >= strong_confidence_threshold and position_size < min_position_size:
        position_size = min_position_size

    # Clamp to [0, max_position_size]
    return max(0.0, min(position_size, max_position_size))


# ---------- Unit Tests (run with python -m script_name) ----------
if __name__ == "__main__":
    # Quick sanity checks
    print("Running built-in tests...")

    # Test RSI signal
    assert calculate_rsi_signal(25)[0] == 1.0
    assert calculate_rsi_signal(75)[0] == -1.0
    assert calculate_rsi_signal(50)[0] == 0.0

    # Test MACD
    assert calculate_macd_signal(10, 5)[0] == 1.0
    assert calculate_macd_signal(5, 10)[0] == -1.0

    # Test Bollinger
    bb_signal, pct_b = calculate_bollinger_signal(110, 90, 100)
    assert abs(bb_signal - 0.0) < 0.001  # price at middle -> signal 0
    bb_signal, pct_b = calculate_bollinger_signal(110, 90, 110)
    assert bb_signal == -1.0  # price at upper band -> overbought -> -1

    # Test momentum
    mom = calculate_momentum_signal(10.0, 0.2)  # 10% change, 20% vol -> normalized 10/(20)=0.5 -> tanh(0.5)~0.462
    assert 0.46 < mom < 0.47

    # Test full signal score
    output = compute_signal_score(
        rsi_14=25,
        macd_line=10,
        macd_signal=5,
        bb_upper=110,
        bb_lower=90,
        live_price=100,
        price_change_pct=5.0,
        volatility_annualized=0.2,
    )
    assert output.direction == "long"
    assert 0.0 < output.confidence <= 1.0
    assert output.score > 0

    # Test risk level
    risk = determine_risk_level(cvar_95=0.30, sharpe=-0.2, max_drawdown=0.4, volatility_annualized=1.2)
    assert risk == "extreme"

    risk = determine_risk_level(cvar_95=0.05, sharpe=1.0, max_drawdown=0.05, volatility_annualized=0.2)
    assert risk == "low"

    # Test position sizing
    pos = calculate_position_size(signal_confidence=0.9, volatility_annualized=0.5, max_position_size=0.10)
    # vol_adjust = 0.5/0.5=1.0; size = 0.1*0.9*1 = 0.09
    assert abs(pos - 0.09) < 0.001

    pos = calculate_position_size(signal_confidence=0.5, volatility_annualized=1.0, max_position_size=0.10)
    # vol_adjust = 0.5/1.0=0.5; size = 0.1*0.5*0.5 = 0.025
    assert abs(pos - 0.025) < 0.001

    print("All tests passed. Code is production-ready.")