"""
Mixture of Experts (MoE) for Signal Generation
────────────────────────────────────────────────
Combines multiple signal experts with a gating network.
Experts specialize in different market conditions.
"""

import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum


class ExpertType(str, Enum):
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER = "bollinger"
    MOMENTUM = "momentum"
    TREND = "trend"


@dataclass
class ExpertOutput:
    """Output from a single expert."""
    expert_type: ExpertType
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    reasoning: str


@dataclass
class MoEOutput:
    """Combined output from MoE layer."""
    final_signal: float  # -1 to 1
    final_confidence: float  # 0 to 1
    selected_experts: List[ExpertType]
    expert_weights: Dict[ExpertType, float]
    gating_reason: str
    position_size: float = 0.0  # 0 to 1 (fraction of capital)
    risk_adjusted_signal: float = 0.0  # signal * position_size


class MoEGatingNetwork:
    """
    Gating network that selects and weights experts based on market conditions.
    Can be rule-based or learned (ML-based).
    """
    
    def __init__(self, mode: str = "rule_based"):
        self.mode = mode
        
    def compute_gate_weights(
        self,
        trend_state: str,  # "trending", "overextended", "pullback", "reversal", "neutral"
        volatility_annualized: Optional[float],
        rsi_14: Optional[float],
        bb_breakout: bool,
    ) -> Dict[ExpertType, float]:
        """
        Compute weights for each expert based on market conditions.
        Returns normalized weights (sum to 1.0).
        Uses trend state instead of simple regime for hierarchical interpretation.
        """
        weights = {
            ExpertType.RSI: 0.0,
            ExpertType.MACD: 0.0,
            ExpertType.BOLLINGER: 0.0,
            ExpertType.MOMENTUM: 0.0,
            ExpertType.TREND: 0.0,
        }
        
        # Base weights by trend state (hierarchical interpretation)
        if trend_state == "trending":
            # Healthy trend - favor trend-following experts
            weights[ExpertType.TREND] = 0.35
            weights[ExpertType.MACD] = 0.35
            weights[ExpertType.MOMENTUM] = 0.2
            weights[ExpertType.RSI] = 0.1
            weights[ExpertType.BOLLINGER] = 0.0
            
        elif trend_state == "overextended":
            # Overextended trend - favor mean reversion (RSI) but keep momentum
            weights[ExpertType.RSI] = 0.35  # Higher for mean reversion
            weights[ExpertType.BOLLINGER] = 0.25
            weights[ExpertType.MACD] = 0.2
            weights[ExpertType.MOMENTUM] = 0.15
            weights[ExpertType.TREND] = 0.05
            
        elif trend_state == "pullback":
            # Pullback within trend - favor trend-following with RSI for timing
            weights[ExpertType.TREND] = 0.3
            weights[ExpertType.MACD] = 0.3
            weights[ExpertType.RSI] = 0.25  # Higher for pullback entry timing
            weights[ExpertType.MOMENTUM] = 0.15
            weights[ExpertType.BOLLINGER] = 0.0
            
        elif trend_state == "reversal":
            # Potential reversal - favor mean reversion
            weights[ExpertType.RSI] = 0.4
            weights[ExpertType.BOLLINGER] = 0.3
            weights[ExpertType.MACD] = 0.2
            weights[ExpertType.MOMENTUM] = 0.1
            weights[ExpertType.TREND] = 0.0
            
        else:  # neutral or unknown
            # Balanced weights when no clear trend state
            weights[ExpertType.RSI] = 0.25
            weights[ExpertType.MACD] = 0.25
            weights[ExpertType.BOLLINGER] = 0.25
            weights[ExpertType.MOMENTUM] = 0.25
            weights[ExpertType.TREND] = 0.0
        
        # Volatility adjustment
        if volatility_annualized is not None:
            if volatility_annualized > 0.8:  # High vol - favor momentum/bollinger
                weights[ExpertType.MOMENTUM] *= 1.3
                weights[ExpertType.BOLLINGER] *= 1.2
                weights[ExpertType.RSI] *= 0.7
            elif volatility_annualized < 0.3:  # Low vol - favor trend/macd
                weights[ExpertType.TREND] *= 1.2
                weights[ExpertType.MACD] *= 1.2
                weights[ExpertType.BOLLINGER] *= 0.8
        
        # RSI extreme values boost RSI expert
        if rsi_14 is not None:
            if rsi_14 < 30 or rsi_14 > 70:
                weights[ExpertType.RSI] *= 1.4
        
        # Bollinger breakout boost (use the detected breakout signal)
        if bb_breakout:
            weights[ExpertType.BOLLINGER] *= 1.5
            weights[ExpertType.MOMENTUM] *= 1.3
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights


class SignalExpert:
    """Base class for signal experts."""
    
    def __init__(self, expert_type: ExpertType):
        self.expert_type = expert_type
    
    def compute_signal(
        self,
        rsi_14: Optional[float],
        macd_line: Optional[float],
        macd_signal: Optional[float],
        bb_upper: Optional[float],
        bb_lower: Optional[float],
        live_price: Optional[float],
        price_change_pct: Optional[float],
        volatility_annualized: Optional[float],
    ) -> ExpertOutput:
        """Compute signal for this expert. Override in subclasses."""
        raise NotImplementedError


class RSIExpert(SignalExpert):
    """Expert specializing in RSI-based reversal signals."""
    
    def __init__(self):
        super().__init__(ExpertType.RSI)
        self.oversold = 30
        self.overbought = 70
    
    def compute_signal(
        self,
        rsi_14: Optional[float],
        macd_line: Optional[float] = None,
        macd_signal: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_lower: Optional[float] = None,
        live_price: Optional[float] = None,
        price_change_pct: Optional[float] = None,
        volatility_annualized: Optional[float] = None,
    ) -> ExpertOutput:
        if rsi_14 is None:
            return ExpertOutput(
                expert_type=self.expert_type,
                signal=0.0,
                confidence=0.0,
                reasoning="RSI data unavailable"
            )
        
        # Clamp
        rsi_14 = max(0.0, min(100.0, rsi_14))
        
        if rsi_14 < self.oversold:
            signal = 1.0
            confidence = 1.0 - (rsi_14 / self.oversold)  # Lower RSI = higher confidence
            reasoning = f"RSI {rsi_14:.1f} oversold - bullish reversal signal"
        elif rsi_14 > self.overbought:
            signal = -1.0
            confidence = (rsi_14 - self.overbought) / (100 - self.overbought)  # Higher RSI = higher confidence
            reasoning = f"RSI {rsi_14:.1f} overbought - bearish reversal signal"
        else:
            signal = 0.0
            confidence = 0.3  # Low confidence in neutral zone
            reasoning = f"RSI {rsi_14:.1f} neutral - no reversal signal"
        
        return ExpertOutput(
            expert_type=self.expert_type,
            signal=signal,
            confidence=min(confidence, 1.0),
            reasoning=reasoning
        )


class MACDExpert(SignalExpert):
    """Expert specializing in MACD momentum signals."""
    
    def __init__(self):
        super().__init__(ExpertType.MACD)
    
    def compute_signal(
        self,
        rsi_14: Optional[float] = None,
        macd_line: Optional[float] = None,
        macd_signal: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_lower: Optional[float] = None,
        live_price: Optional[float] = None,
        price_change_pct: Optional[float] = None,
        volatility_annualized: Optional[float] = None,
    ) -> ExpertOutput:
        if macd_line is None or macd_signal is None:
            return ExpertOutput(
                expert_type=self.expert_type,
                signal=0.0,
                confidence=0.0,
                reasoning="MACD data unavailable"
            )
        
        diff = macd_line - macd_signal
        histogram = diff  # Simplified
        
        # Normalize using sum of absolute values (more consistent than price scaling)
        scale = abs(macd_line) + abs(macd_signal) + 1e-6
        normalized_diff = diff / scale
        
        signal = math.tanh(normalized_diff * 3)  # Scale factor for sensitivity
        confidence = min(abs(normalized_diff) * 2, 1.0)  # Higher diff = higher confidence
        
        if diff > 0:
            reasoning = f"MACD bullish (line {macd_line:.2f} > signal {macd_signal:.2f})"
        else:
            reasoning = f"MACD bearish (line {macd_line:.2f} < signal {macd_signal:.2f})"
        
        return ExpertOutput(
            expert_type=self.expert_type,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning
        )


class BollingerExpert(SignalExpert):
    """Expert specializing in Bollinger Bands breakout signals."""
    
    def __init__(self):
        super().__init__(ExpertType.BOLLINGER)
    
    def compute_signal(
        self,
        rsi_14: Optional[float] = None,
        macd_line: Optional[float] = None,
        macd_signal: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_lower: Optional[float] = None,
        live_price: Optional[float] = None,
        price_change_pct: Optional[float] = None,
        volatility_annualized: Optional[float] = None,
    ) -> ExpertOutput:
        if bb_upper is None or bb_lower is None or live_price is None:
            return ExpertOutput(
                expert_type=self.expert_type,
                signal=0.0,
                confidence=0.0,
                reasoning="Bollinger data unavailable"
            )
        
        bandwidth = bb_upper - bb_lower
        if abs(bandwidth) < 1e-8:
            return ExpertOutput(
                expert_type=self.expert_type,
                signal=0.0,
                confidence=0.0,
                reasoning="Degenerate Bollinger bands"
            )
        
        # Compute raw %B BEFORE clamping for breakout detection
        raw_pct_b = (live_price - bb_lower) / bandwidth
        bb_breakout = raw_pct_b > 1.0 or raw_pct_b < 0.0
        pct_b = max(0.0, min(1.0, raw_pct_b))  # Clamp for signal calculation
        
        # %B-based signal (inverted: high %B = overbought = negative)
        signal = (0.5 - pct_b) * 2  # -1 to 1
        
        if bb_breakout:
            # Breakout = continuation (not reversal)
            if raw_pct_b > 1.0:
                signal = 1.0  # Bullish continuation
                confidence = 1.0
                reasoning = f"Price above upper band (raw %B {raw_pct_b:.2f}) - breakout long (continuation)"
            else:
                signal = -1.0  # Bearish continuation
                confidence = 1.0
                reasoning = f"Price below lower band (raw %B {raw_pct_b:.2f}) - breakout short (continuation)"
        elif pct_b > 0.8:
            # Near upper band = mean reversion (overbought)
            confidence = 0.7
            reasoning = f"Price near upper band (%B {pct_b:.2f}) - overbought (mean reversion)"
        elif pct_b < 0.2:
            # Near lower band = mean reversion (oversold)
            confidence = 0.7
            reasoning = f"Price near lower band (%B {pct_b:.2f}) - oversold (mean reversion)"
        else:
            confidence = 0.3
            reasoning = f"Price within bands (%B {pct_b:.2f}) - neutral"
        
        return ExpertOutput(
            expert_type=self.expert_type,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning
        )


class MomentumExpert(SignalExpert):
    """Expert specializing in momentum-based trend signals."""
    
    def __init__(self):
        super().__init__(ExpertType.MOMENTUM)
    
    def compute_signal(
        self,
        rsi_14: Optional[float] = None,
        macd_line: Optional[float] = None,
        macd_signal: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_lower: Optional[float] = None,
        live_price: Optional[float] = None,
        price_change_pct: Optional[float] = None,
        volatility_annualized: Optional[float] = None,
    ) -> ExpertOutput:
        if price_change_pct is None:
            return ExpertOutput(
                expert_type=self.expert_type,
                signal=0.0,
                confidence=0.0,
                reasoning="Price change data unavailable"
            )
        
        # Raw momentum signal (no volatility normalization here - done in risk layer)
        # This avoids double-correction of volatility
        return_decimal = price_change_pct / 100.0
        signal = math.tanh(return_decimal * 5)  # Less aggressive scaling (was 10)
        
        # Confidence using exponential decay (less aggressive)
        confidence = 1 - math.exp(-abs(return_decimal) * 20)  # Less aggressive (was 50)
        
        direction = "bullish" if signal > 0 else "bearish"
        reasoning = f"Momentum {direction} (change {price_change_pct:.2f}%, signal {signal:.2f})"
        
        return ExpertOutput(
            expert_type=self.expert_type,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning
        )


class RiskLayer:
    """
    Risk layer for position sizing and volatility targeting.
    Converts raw signal/confidence into position size with risk controls.
    """
    
    def __init__(
        self,
        max_position_size: float = 0.10,  # Max 10% of capital per position
        min_position_size: float = 0.01,  # Min 1% if signal is strong
        volatility_target: float = 0.15,  # 15% annualized vol target
        signal_strength_fraction: float = 0.25,  # Signal strength scaling (not true Kelly)
    ):
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.volatility_target = volatility_target
        self.signal_strength_fraction = signal_strength_fraction
    
    def compute_position_size(
        self,
        signal: float,
        confidence: float,
        volatility_annualized: Optional[float],
        sharpe: Optional[float] = None,
    ) -> tuple[float, float]:
        """
        Compute position size and risk-adjusted signal.
        Returns (position_size, risk_adjusted_signal).
        """
        # Volatility targeting: scale position inversely with volatility
        if volatility_annualized is not None and volatility_annualized > 0:
            vol_adjustment = self.volatility_target / max(volatility_annualized, 0.01)
            vol_adjustment = min(vol_adjustment, 2.0)  # Cap at 2x to avoid over-leverage
        else:
            vol_adjustment = 1.0
        
        # Sharpe-based cap: lower Sharpe = lower max position
        sharpe_cap = 1.0
        if sharpe is not None:
            if sharpe < 0:
                sharpe_cap = 0.3  # Negative Sharpe: cap at 30%
            elif sharpe < 0.5:
                sharpe_cap = 0.5
            elif sharpe < 1.0:
                sharpe_cap = 0.7
            elif sharpe < 2.0:
                sharpe_cap = 0.9
            else:
                sharpe_cap = 1.0  # Excellent Sharpe: no cap
        
        # Base position: signal * (0.5 + 0.5 * confidence) * signal strength scaling
        # Reduces double-dependence on confidence (confidence already used in aggregation)
        base_position = abs(signal) * (0.5 + 0.5 * confidence) * self.signal_strength_fraction
        
        # Apply adjustments
        position_size = base_position * vol_adjustment * sharpe_cap
        
        # Enforce bounds
        position_size = max(0.0, min(position_size, self.max_position_size))
        
        # Enforce minimum only if signal is strong
        if confidence >= 0.7 and position_size < self.min_position_size:
            position_size = self.min_position_size
        
        # Risk-adjusted signal: signal scaled by position size
        risk_adjusted_signal = signal * (position_size / self.max_position_size)
        
        return position_size, risk_adjusted_signal


class TrendExpert(SignalExpert):
    """Expert specializing in trend continuation signals."""
    
    def __init__(self):
        super().__init__(ExpertType.TREND)
    
    def compute_signal(
        self,
        rsi_14: Optional[float] = None,
        macd_line: Optional[float] = None,
        macd_signal: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_lower: Optional[float] = None,
        live_price: Optional[float] = None,
        price_change_pct: Optional[float] = None,
        volatility_annualized: Optional[float] = None,
    ) -> ExpertOutput:
        # Trend expert combines MACD and momentum
        signals = []
        confidences = []
        
        if macd_line is not None and macd_signal is not None:
            macd_sig = 1.0 if macd_line > macd_signal else -1.0
            signals.append(macd_sig)
            confidences.append(0.8)
        
        if price_change_pct is not None:
            mom_sig = 1.0 if price_change_pct > 0 else -1.0
            signals.append(mom_sig)
            confidences.append(0.7)
        
        if not signals:
            return ExpertOutput(
                expert_type=self.expert_type,
                signal=0.0,
                confidence=0.0,
                reasoning="Insufficient trend data"
            )
        
        # Weighted average
        total_conf = sum(confidences)
        signal = sum(s * c for s, c in zip(signals, confidences)) / total_conf
        confidence = total_conf / len(signals)
        
        direction = "bullish" if signal > 0 else "bearish"
        reasoning = f"Trend {direction} (continuation signal)"
        
        return ExpertOutput(
            expert_type=self.expert_type,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning
        )


class MoESignalLayer:
    """
    Mixture of Experts layer for signal generation.
    Combines multiple experts with gating network and risk layer.
    """
    
    def __init__(
        self,
        gating_mode: str = "rule_based",
        enable_risk_layer: bool = True,
        max_position_size: float = 0.10,
    ):
        self.gating_network = MoEGatingNetwork(mode=gating_mode)
        self.experts = {
            ExpertType.RSI: RSIExpert(),
            ExpertType.MACD: MACDExpert(),
            ExpertType.BOLLINGER: BollingerExpert(),
            ExpertType.MOMENTUM: MomentumExpert(),
            ExpertType.TREND: TrendExpert(),
        }
        self.enable_risk_layer = enable_risk_layer
        self.risk_layer = RiskLayer(max_position_size=max_position_size) if enable_risk_layer else None
    
    def compute_signal(
        self,
        rsi_14: Optional[float],
        macd_line: Optional[float],
        macd_signal: Optional[float],
        bb_upper: Optional[float],
        bb_lower: Optional[float],
        live_price: Optional[float],
        price_change_pct: Optional[float],
        volatility_annualized: Optional[float],
        trend_state: str = "neutral",  # Changed from regime to trend_state
        sharpe: Optional[float] = None,
    ) -> MoEOutput:
        """
        Compute combined signal using MoE.
        """
        # Bollinger breakout detection for gating
        bb_breakout = False
        if bb_upper is not None and bb_lower is not None and live_price is not None:
            bandwidth = bb_upper - bb_lower
            if abs(bandwidth) > 1e-8:
                raw_pct_b = (live_price - bb_lower) / bandwidth
                bb_breakout = raw_pct_b > 1.0 or raw_pct_b < 0.0
        
        # Get gating weights
        weights = self.gating_network.compute_gate_weights(
            trend_state=trend_state,
            volatility_annualized=volatility_annualized,
            rsi_14=rsi_14,
            bb_breakout=bb_breakout,
        )
        
        # Soft selection: keep all experts (no hard thresholding)
        # This ensures smooth regime transitions
        active_experts = weights
        
        # Get outputs from active experts
        expert_outputs = []
        for expert_type, weight in active_experts.items():
            expert = self.experts[expert_type]
            output = expert.compute_signal(
                rsi_14=rsi_14,
                macd_line=macd_line,
                macd_signal=macd_signal,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                live_price=live_price,
                price_change_pct=price_change_pct,
                volatility_annualized=volatility_annualized,
            )
            expert_outputs.append((output, weight))
        
        # Combine signals using confidence-weighted aggregation
        # Fix: separate signal aggregation from confidence aggregation to avoid quadratic amplification
        weighted_signal = 0.0
        total_effective_weight = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for output, weight in expert_outputs:
            effective_weight = weight * output.confidence
            weighted_signal += output.signal * effective_weight
            total_effective_weight += effective_weight
            # Linear confidence aggregation (not quadratic)
            weighted_confidence += output.confidence * weight
            total_weight += weight
        
        if total_effective_weight > 0:
            final_signal = weighted_signal / total_effective_weight
        else:
            final_signal = 0.0
        
        if total_weight > 0:
            final_confidence = weighted_confidence / total_weight
        else:
            final_confidence = 0.0
        
        # Clip
        final_signal = max(-1.0, min(1.0, final_signal))
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        selected_experts = [output.expert_type for output, _ in expert_outputs]
        
        gating_reason = f"Trend State: {trend_state}, Active experts: {[e.value for e in selected_experts]}"
        
        # No-trade mechanism: if confidence is too low, force no trade
        no_trade_threshold = 0.4
        if final_confidence < no_trade_threshold:
            gating_reason += " [NO-TRADE: low confidence]"
            # Apply risk layer with no-trade override
            if self.enable_risk_layer and self.risk_layer is not None:
                position_size, risk_adjusted_signal = self.risk_layer.compute_position_size(
                    signal=final_signal,
                    confidence=final_confidence,
                    volatility_annualized=volatility_annualized,
                    sharpe=sharpe,
                )
                position_size = 0.0  # Override to zero
                risk_adjusted_signal = 0.0
            else:
                position_size = 0.0
                risk_adjusted_signal = 0.0
        else:
            # Apply risk layer normally
            if self.enable_risk_layer and self.risk_layer is not None:
                position_size, risk_adjusted_signal = self.risk_layer.compute_position_size(
                    signal=final_signal,
                    confidence=final_confidence,
                    volatility_annualized=volatility_annualized,
                    sharpe=sharpe,
                )
            else:
                position_size = 0.0
                risk_adjusted_signal = 0.0
        
        return MoEOutput(
            final_signal=final_signal,
            final_confidence=final_confidence,
            selected_experts=selected_experts,
            expert_weights=active_experts,
            gating_reason=gating_reason,
            position_size=position_size,
            risk_adjusted_signal=risk_adjusted_signal,
        )


# Unit tests
if __name__ == "__main__":
    print("Testing MoE layer...")
    
    moe = MoESignalLayer()
    
    # Test trending regime
    output = moe.compute_signal(
        rsi_14=50,
        macd_line=10,
        macd_signal=5,
        bb_upper=110,
        bb_lower=90,
        live_price=100,
        price_change_pct=5.0,
        volatility_annualized=0.3,
        regime="trending",
        sharpe=1.5,
    )
    
    print(f"Signal: {output.final_signal:.3f}")
    print(f"Confidence: {output.final_confidence:.3f}")
    print(f"Position Size: {output.position_size:.3f}")
    print(f"Risk-Adjusted Signal: {output.risk_adjusted_signal:.3f}")
    print(f"Experts: {[e.value for e in output.selected_experts]}")
    print(f"Weights: {output.expert_weights}")
    print(f"Reason: {output.gating_reason}")
    
    print("\nMoE layer tests passed.")
