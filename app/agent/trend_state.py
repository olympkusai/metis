"""
Trend State Machine - Market Lifecycle Modeling
────────────────────────────────────────────────
Models the lifecycle of market trends to enable contextual interpretation.

States:
- trending: Healthy trend with momentum
- overextended: Trend extended (RSI > 70, price near bands)
- pullback: Temporary correction within trend
- reversal: Trend change confirmed
"""

from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass


class TrendState(str, Enum):
    """States in the trend lifecycle."""
    TRENDING = "trending"
    OVEREXTENDED = "overextended"
    PULLBACK = "pullback"
    REVERSAL = "reversal"
    NEUTRAL = "neutral"


class TrendDirection(str, Enum):
    """Direction of the trend."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TrendContext:
    """Contextual information about the trend state."""
    state: TrendState
    direction: TrendDirection
    confidence: float  # 0 to 1
    reasoning: str


class TrendStateMachine:
    """
    State machine for trend lifecycle classification.
    Implements hierarchical interpretation across timeframes.
    """

    def __init__(self):
        # Thresholds for state transitions
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.bb_overextended_threshold = 0.85  # %B > 0.85 = overextended
        self.bb_pullback_threshold = 0.25  # %B < 0.25 = pullback opportunity

    def classify_trend_state(
        self,
        rsi: Optional[float],
        macd_bullish: bool,
        bb_pct_b: Optional[float],
        price_change_pct: Optional[float],
        volatility_annualized: Optional[float],
    ) -> TrendContext:
        """
        Classify the current trend state based on indicators.
        """
        if rsi is None or bb_pct_b is None:
            return TrendContext(
                state=TrendState.NEUTRAL,
                direction=TrendDirection.NEUTRAL,
                confidence=0.0,
                reasoning="Insufficient data for trend classification"
            )

        # Determine direction
        if macd_bullish:
            direction = TrendDirection.BULLISH
        elif not macd_bullish:
            direction = TrendDirection.BEARISH
        else:
            direction = TrendDirection.NEUTRAL

        # Classify state
        state, confidence, reasoning = self._determine_state(
            rsi, bb_pct_b, price_change_pct, volatility_annualized, direction
        )

        return TrendContext(
            state=state,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning
        )

    def _determine_state(
        self,
        rsi: float,
        bb_pct_b: float,
        price_change_pct: Optional[float],
        volatility_annualized: Optional[float],
        direction: TrendDirection,
    ) -> Tuple[TrendState, float, str]:
        """
        Determine trend state with confidence and reasoning.
        """
        # Check for reversal conditions
        if direction == TrendDirection.BULLISH and rsi < self.rsi_oversold:
            return TrendState.PULLBACK, 0.8, "Oversold within bullish trend - pullback opportunity"
        if direction == TrendDirection.BEARISH and rsi > self.rsi_overbought:
            return TrendState.PULLBACK, 0.8, "Overbought within bearish trend - pullback opportunity"

        # Check for overextended conditions
        if direction == TrendDirection.BULLISH and rsi > self.rsi_overbought and bb_pct_b > self.bb_overextended_threshold:
            return TrendState.OVEREXTENDED, 0.9, "Overextended bullish trend - RSI overbought + price near upper band"
        if direction == TrendDirection.BEARISH and rsi < self.rsi_oversold and bb_pct_b < (1 - self.bb_overextended_threshold):
            return TrendState.OVEREXTENDED, 0.9, "Overextended bearish trend - RSI oversold + price near lower band"

        # Check for healthy trending
        if direction == TrendDirection.BULLISH and 30 <= rsi <= 70 and 0.3 <= bb_pct_b <= 0.7:
            return TrendState.TRENDING, 0.85, "Healthy bullish trend - RSI in range, price mid-band"
        if direction == TrendDirection.BEARISH and 30 <= rsi <= 70 and 0.3 <= bb_pct_b <= 0.7:
            return TrendState.TRENDING, 0.85, "Healthy bearish trend - RSI in range, price mid-band"

        # Check for reversal
        if direction == TrendDirection.BULLISH and rsi < 30 and bb_pct_b < 0.2:
            return TrendState.REVERSAL, 0.7, "Potential bullish reversal - deeply oversold"
        if direction == TrendDirection.BEARISH and rsi > 70 and bb_pct_b > 0.8:
            return TrendState.REVERSAL, 0.7, "Potential bearish reversal - deeply overbought"

        # Default to trending with lower confidence
        return TrendState.TRENDING, 0.5, f"Trending ({direction.value}) - no extreme conditions detected"


class MultiTimeframeInterpreter:
    """
    Implements hierarchical interpretation across timeframes.
    Macro (1D) > Setup (4H) > Execution (1H)
    """

    def __init__(self):
        self.trend_machine = TrendStateMachine()

    def interpret_multi_tf_signal(
        self,
        # Macro (1D) - regime detection
        macro_rsi: Optional[float],
        macro_macd_bullish: bool,
        macro_bb_pct_b: Optional[float],
        macro_regime: Optional[str],
        # Setup (4H) - signal generation
        setup_rsi: Optional[float],
        setup_macd_bullish: bool,
        setup_bb_pct_b: Optional[float],
        # Execution (1H) - timing
        exec_rsi: Optional[float],
        exec_macd_bullish: bool,
        # Additional context
        volatility_annualized: Optional[float],
        price_change_pct: Optional[float],
    ) -> dict:
        """
        Interpret signal across timeframes with hierarchical rules.
        Returns: {
            "signal_direction": "long" | "short" | "conditional_long" | "conditional_short" | "neutral",
            "signal_type": "trend_follow" | "pullback_entry" | "breakout" | "reversal",
            "confidence": float,
            "reasoning": str,
            "trend_state": TrendState,
            "execution_timing": "immediate" | "wait_for_pullback" | "wait_for_confirmation"
        }
        """
        # Classify macro trend state (most important)
        macro_context = self.trend_machine.classify_trend_state(
            rsi=macro_rsi,
            macd_bullish=macro_macd_bullish,
            bb_pct_b=macro_bb_pct_b,
            price_change_pct=price_change_pct,
            volatility_annualized=volatility_annualized,
        )

        # Classify setup trend state
        setup_context = self.trend_machine.classify_trend_state(
            rsi=setup_rsi,
            macd_bullish=setup_macd_bullish,
            bb_pct_b=setup_bb_pct_b,
            price_change_pct=price_change_pct,
            volatility_annualized=volatility_annualized,
        )

        # Apply hierarchical rules
        result = self._apply_hierarchical_rules(
            macro_context, setup_context, exec_rsi, exec_macd_bullish,
            macro_regime, volatility_annualized, price_change_pct
        )

        return result

    def _apply_hierarchical_rules(
        self,
        macro_context: TrendContext,
        setup_context: TrendContext,
        exec_rsi: Optional[float],
        exec_macd_bullish: bool,
        macro_regime: Optional[str],
        volatility_annualized: Optional[float],
        price_change_pct: Optional[float],
    ) -> dict:
        """
        Apply hierarchical interpretation rules.
        """
        # Rule 1: Macro direction dominates
        if macro_context.direction == TrendDirection.BULLISH:
            # Rule 1a: Bullish macro + oversold execution = pullback long (NOT neutral)
            if exec_rsi is not None and exec_rsi < 30:
                return {
                    "signal_direction": "conditional_long",
                    "signal_type": "pullback_entry",
                    "confidence": 0.75,
                    "reasoning": f"Bullish macro trend ({macro_context.state.value}) with execution layer oversold - buy the dip opportunity",
                    "trend_state": macro_context.state,
                    "execution_timing": "immediate"
                }
            
            # Rule 1b: Bullish macro + overextended = reduce size, still long
            if macro_context.state == TrendState.OVEREXTENDED:
                return {
                    "signal_direction": "conditional_long",
                    "signal_type": "trend_follow",
                    "confidence": 0.6,
                    "reasoning": f"Bullish macro but overextended ({macro_context.reasoning}) - reduced position size recommended",
                    "trend_state": TrendState.OVEREXTENDED,
                    "execution_timing": "wait_for_pullback"
                }
            
            # Rule 1c: Bullish macro + healthy setup = trend follow
            if setup_context.direction == TrendDirection.BULLISH and setup_context.state == TrendState.TRENDING:
                return {
                    "signal_direction": "long",
                    "signal_type": "trend_follow",
                    "confidence": 0.8,
                    "reasoning": f"Healthy bullish trend across macro and setup layers - trend continuation",
                    "trend_state": TrendState.TRENDING,
                    "execution_timing": "immediate"
                }
            
            # Rule 1d: Bullish macro + bearish setup = wait (NOT neutral)
            if setup_context.direction == TrendDirection.BEARISH:
                return {
                    "signal_direction": "conditional_long",
                    "signal_type": "pullback_entry",
                    "confidence": 0.5,
                    "reasoning": f"Bullish macro but bearish setup - wait for pullback to resolve",
                    "trend_state": TrendState.PULLBACK,
                    "execution_timing": "wait_for_confirmation"
                }
            
            # Default bullish macro
            return {
                "signal_direction": "long",
                "signal_type": "trend_follow",
                "confidence": 0.7,
                "reasoning": f"Bullish macro trend - default to long with macro bias",
                "trend_state": macro_context.state,
                "execution_timing": "immediate"
            }

        elif macro_context.direction == TrendDirection.BEARISH:
            # Symmetric rules for bearish
            if exec_rsi is not None and exec_rsi > 70:
                return {
                    "signal_direction": "conditional_short",
                    "signal_type": "pullback_entry",
                    "confidence": 0.75,
                    "reasoning": f"Bearish macro trend ({macro_context.state.value}) with execution layer overbought - short the rip opportunity",
                    "trend_state": macro_context.state,
                    "execution_timing": "immediate"
                }
            
            if macro_context.state == TrendState.OVEREXTENDED:
                return {
                    "signal_direction": "conditional_short",
                    "signal_type": "trend_follow",
                    "confidence": 0.6,
                    "reasoning": f"Bearish macro but overextended ({macro_context.reasoning}) - reduced position size recommended",
                    "trend_state": TrendState.OVEREXTENDED,
                    "execution_timing": "wait_for_pullback"
                }
            
            if setup_context.direction == TrendDirection.BEARISH and setup_context.state == TrendState.TRENDING:
                return {
                    "signal_direction": "short",
                    "signal_type": "trend_follow",
                    "confidence": 0.8,
                    "reasoning": f"Healthy bearish trend across macro and setup layers - trend continuation",
                    "trend_state": TrendState.TRENDING,
                    "execution_timing": "immediate"
                }
            
            # Default bearish macro
            return {
                "signal_direction": "short",
                "signal_type": "trend_follow",
                "confidence": 0.7,
                "reasoning": f"Bearish macro trend - default to short with macro bias",
                "trend_state": macro_context.state,
                "execution_timing": "immediate"
            }

        else:  # Neutral macro
            # Rule 4: Neutral only in specific conditions
            is_low_vol = volatility_annualized is not None and volatility_annualized < 0.2
            is_flat_momentum = price_change_pct is not None and abs(price_change_pct) < 0.5
            
            if is_low_vol and is_flat_momentum:
                return {
                    "signal_direction": "neutral",
                    "signal_type": "no_edge",
                    "confidence": 0.9,
                    "reasoning": "Low volatility with flat momentum - no actionable edge",
                    "trend_state": TrendState.NEUTRAL,
                    "execution_timing": "wait"
                }
            
            # If macro is neutral but setup has direction, follow setup with lower confidence
            if setup_context.direction == TrendDirection.BULLISH:
                return {
                    "signal_direction": "conditional_long",
                    "signal_type": "setup_follow",
                    "confidence": 0.5,
                    "reasoning": "Neutral macro but bullish setup - lower confidence long",
                    "trend_state": TrendState.TRENDING,
                    "execution_timing": "wait_for_confirmation"
                }
            
            if setup_context.direction == TrendDirection.BEARISH:
                return {
                    "signal_direction": "conditional_short",
                    "signal_type": "setup_follow",
                    "confidence": 0.5,
                    "reasoning": "Neutral macro but bearish setup - lower confidence short",
                    "trend_state": TrendState.TRENDING,
                    "execution_timing": "wait_for_confirmation"
                }
            
            # True neutral
            return {
                "signal_direction": "neutral",
                "signal_type": "no_edge",
                "confidence": 0.8,
                "reasoning": "No clear directional bias across timeframes",
                "trend_state": TrendState.NEUTRAL,
                "execution_timing": "wait"
            }
