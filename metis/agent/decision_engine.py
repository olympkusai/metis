"""
Decision Engine - Deterministic Multi-Timeframe Decision Layer
──────────────────────────────────────────────────────────────
This layer provides the FINAL decision authority, overriding MoE when there's
clear hierarchical conflict resolution needed.

Rules:
1. Macro (1D) trend dominates Setup (4H) and Execution (1H)
2. Pullback within trend = directional opportunity, NOT neutral
3. Neutral only in specific low-volatility, flat-momentum cases
4. MoE is auxiliary input, not final authority
"""

from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum


class MacroTrend(str, Enum):
    """Macro trend direction from 1D timeframe."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ExecutionState(str, Enum):
    """Execution layer state from 1H timeframe."""
    OVERSOLD = "oversold"      # RSI < 30
    OVERBOUGHT = "overbought"  # RSI > 70
    NEUTRAL = "neutral"        # 30 <= RSI <= 70


class FinalSignal(str, Enum):
    """Final signal decision."""
    LONG = "long"
    SHORT = "short"
    CONDITIONAL_LONG = "conditional_long"      # Wait for entry
    CONDITIONAL_SHORT = "conditional_short"    # Wait for entry
    WAIT = "wait"                              # No clear edge
    NEUTRAL = "neutral"                        # Only in specific cases


class SignalType(str, Enum):
    """Type of signal for context."""
    TREND_FOLLOW = "trend_follow"
    PULLBACK_ENTRY = "pullback_entry"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    NO_EDGE = "no_edge"


@dataclass
class DecisionOutput:
    """Final decision from the decision engine."""
    signal: FinalSignal
    signal_type: SignalType
    confidence: float  # 0 to 1
    reasoning: str
    execution_timing: str  # "immediate", "wait_for_pullback", "wait_for_confirmation"


class DecisionEngine:
    """
    Deterministic decision engine with hierarchical multi-timeframe rules.
    This engine has FINAL authority over MoE output.
    """

    def __init__(self):
        # Thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.volatility_low_threshold = 0.2  # 20% annualized
        self.momentum_flat_threshold = 0.5   # 0.5% change

    def decide(
        self,
        # Macro (1D) - regime detection
        macro_trend: MacroTrend,
        macro_rsi: Optional[float],
        macro_bb_pct_b: Optional[float],
        # Setup (4H) - signal generation
        setup_rsi: Optional[float],
        setup_macd_bullish: bool,
        # Execution (1H) - timing
        exec_rsi: Optional[float],
        exec_macd_bullish: bool,
        # Additional context
        volatility_annualized: Optional[float],
        price_change_pct: Optional[float],
        # MoE auxiliary input (optional)
        moe_signal: Optional[float] = None,
        moe_confidence: Optional[float] = None,
    ) -> DecisionOutput:
        """
        Make final decision based on hierarchical rules.
        MoE is auxiliary, not authoritative.
        """
        # Determine execution state
        exec_state = self._classify_execution_state(exec_rsi)

        # Apply hierarchical rules
        decision = self._apply_hierarchical_rules(
            macro_trend, exec_state, setup_rsi, setup_macd_bullish,
            volatility_annualized, price_change_pct,
            moe_signal, moe_confidence
        )

        return decision

    def _classify_execution_state(self, rsi: Optional[float]) -> ExecutionState:
        """Classify execution layer state based on RSI."""
        if rsi is None:
            return ExecutionState.NEUTRAL
        
        if rsi < self.rsi_oversold:
            return ExecutionState.OVERSOLD
        elif rsi > self.rsi_overbought:
            return ExecutionState.OVERBOUGHT
        else:
            return ExecutionState.NEUTRAL

    def _apply_hierarchical_rules(
        self,
        macro_trend: MacroTrend,
        exec_state: ExecutionState,
        setup_rsi: Optional[float],
        setup_macd_bullish: bool,
        volatility_annualized: Optional[float],
        price_change_pct: Optional[float],
        moe_signal: Optional[float],
        moe_confidence: Optional[float],
    ) -> DecisionOutput:
        """
        Apply hierarchical rules with macro dominance.
        This is the CORE decision logic.
        """
        # Rule 1: Macro BULLISH dominates
        if macro_trend == MacroTrend.BULLISH:
            # Rule 1a: Bullish macro + oversold execution = pullback long (NOT neutral)
            if exec_state == ExecutionState.OVERSOLD:
                return DecisionOutput(
                    signal=FinalSignal.CONDITIONAL_LONG,
                    signal_type=SignalType.PULLBACK_ENTRY,
                    confidence=0.75,
                    reasoning="Bullish macro trend with execution layer oversold - buy the dip opportunity (pullback entry)",
                    execution_timing="immediate"
                )
            
            # Rule 1b: Bullish macro + overbought execution = wait or reduce
            if exec_state == ExecutionState.OVERBOUGHT:
                return DecisionOutput(
                    signal=FinalSignal.WAIT,
                    signal_type=SignalType.TREND_FOLLOW,
                    confidence=0.5,
                    reasoning="Bullish macro but execution layer overbought - wait for pullback or reduce position size",
                    execution_timing="wait_for_pullback"
                )
            
            # Rule 1c: Bullish macro + neutral execution = trend follow
            if exec_state == ExecutionState.NEUTRAL:
                # Check setup layer for confirmation
                if setup_macd_bullish:
                    return DecisionOutput(
                        signal=FinalSignal.LONG,
                        signal_type=SignalType.TREND_FOLLOW,
                        confidence=0.8,
                        reasoning="Bullish macro confirmed by setup layer - trend continuation",
                        execution_timing="immediate"
                    )
                else:
                    return DecisionOutput(
                        signal=FinalSignal.CONDITIONAL_LONG,
                        signal_type=SignalType.TREND_FOLLOW,
                        confidence=0.6,
                        reasoning="Bullish macro but setup layer not confirming - proceed with caution",
                        execution_timing="wait_for_confirmation"
                    )

        # Rule 2: Macro BEARISH dominates
        elif macro_trend == MacroTrend.BEARISH:
            # Rule 2a: Bearish macro + overbought execution = pullback short (NOT neutral)
            if exec_state == ExecutionState.OVERBOUGHT:
                return DecisionOutput(
                    signal=FinalSignal.CONDITIONAL_SHORT,
                    signal_type=SignalType.PULLBACK_ENTRY,
                    confidence=0.75,
                    reasoning="Bearish macro trend with execution layer overbought - short the rip opportunity (pullback entry)",
                    execution_timing="immediate"
                )
            
            # Rule 2b: Bearish macro + oversold execution = wait or reduce
            if exec_state == ExecutionState.OVERSOLD:
                return DecisionOutput(
                    signal=FinalSignal.WAIT,
                    signal_type=SignalType.TREND_FOLLOW,
                    confidence=0.5,
                    reasoning="Bearish macro but execution layer oversold - wait for pullback or reduce position size",
                    execution_timing="wait_for_pullback"
                )
            
            # Rule 2c: Bearish macro + neutral execution = trend follow
            if exec_state == ExecutionState.NEUTRAL:
                if not setup_macd_bullish:
                    return DecisionOutput(
                        signal=FinalSignal.SHORT,
                        signal_type=SignalType.TREND_FOLLOW,
                        confidence=0.8,
                        reasoning="Bearish macro confirmed by setup layer - trend continuation",
                        execution_timing="immediate"
                    )
                else:
                    return DecisionOutput(
                        signal=FinalSignal.CONDITIONAL_SHORT,
                        signal_type=SignalType.TREND_FOLLOW,
                        confidence=0.6,
                        reasoning="Bearish macro but setup layer conflicting - proceed with caution",
                        execution_timing="wait_for_confirmation"
                    )

        # Rule 3: Macro NEUTRAL - only neutral in specific conditions
        elif macro_trend == MacroTrend.NEUTRAL:
            is_low_vol = volatility_annualized is not None and volatility_annualized < self.volatility_low_threshold
            is_flat_momentum = price_change_pct is not None and abs(price_change_pct) < self.momentum_flat_threshold
            
            if is_low_vol and is_flat_momentum:
                return DecisionOutput(
                    signal=FinalSignal.NEUTRAL,
                    signal_type=SignalType.NO_EDGE,
                    confidence=0.9,
                    reasoning="Low volatility with flat momentum - no actionable edge",
                    execution_timing="wait"
                )
            
            # If macro is neutral but setup has direction, follow setup with lower confidence
            if setup_macd_bullish:
                return DecisionOutput(
                    signal=FinalSignal.CONDITIONAL_LONG,
                    signal_type=SignalType.TREND_FOLLOW,
                    confidence=0.5,
                    reasoning="Neutral macro but bullish setup - lower confidence long",
                    execution_timing="wait_for_confirmation"
                )
            
            if not setup_macd_bullish and setup_rsi is not None:
                if setup_rsi < 40:
                    return DecisionOutput(
                        signal=FinalSignal.CONDITIONAL_SHORT,
                        signal_type=SignalType.TREND_FOLLOW,
                        confidence=0.5,
                        reasoning="Neutral macro but bearish setup - lower confidence short",
                        execution_timing="wait_for_confirmation"
                    )

        # Rule 4: Fallback - use MoE as auxiliary if available
        if moe_signal is not None and moe_confidence is not None:
            if moe_confidence > 0.6:
                if moe_signal > 0.2:
                    return DecisionOutput(
                        signal=FinalSignal.LONG,
                        signal_type=SignalType.TREND_FOLLOW,
                        confidence=moe_confidence,
                        reasoning=f"MoE indicates long with confidence {moe_confidence:.2f} (auxiliary)",
                        execution_timing="immediate"
                    )
                elif moe_signal < -0.2:
                    return DecisionOutput(
                        signal=FinalSignal.SHORT,
                        signal_type=SignalType.TREND_FOLLOW,
                        confidence=moe_confidence,
                        reasoning=f"MoE indicates short with confidence {moe_confidence:.2f} (auxiliary)",
                        execution_timing="immediate"
                    )

        # Default: no clear edge
        return DecisionOutput(
            signal=FinalSignal.WAIT,
            signal_type=SignalType.NO_EDGE,
            confidence=0.8,
            reasoning="No clear directional bias across timeframes - wait",
            execution_timing="wait"
        )
