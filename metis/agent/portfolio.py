"""
Portfolio & Exposure Layer
────────────────────────────────────────────────
Institutional-grade portfolio management with correlation tracking,
aggregate exposure calculation, and marginal risk assessment.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import math


class AssetClass(str, Enum):
    """Asset classification for portfolio management."""
    CRYPTO = "crypto"
    EQUITY = "equity"
    FX = "fx"
    COMMODITY = "commodity"
    FIXED_INCOME = "fixed_income"


@dataclass
class Position:
    """Single position in the portfolio."""
    symbol: str
    asset_class: AssetClass
    quantity: float          # positive for long, negative for short
    entry_price: float
    current_price: float
    notional: float = 0.0    # abs(quantity) * current_price
    weight: float = 0.0      # % of portfolio (absolute value)
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def __post_init__(self):
        self.update_prices(self.current_price)

    def update_prices(self, new_price: float) -> None:
        """Recalculate notional, PnL, and weight (caller must recalc portfolio totals)."""
        if new_price <= 0:
            raise ValueError(f"Price must be positive, got {new_price}")
        self.current_price = new_price
        self.notional = abs(self.quantity) * new_price
        # PnL: (current - entry) * quantity (positive quantity for long, negative for short)
        self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        self.unrealized_pnl_pct = self.unrealized_pnl / (abs(self.quantity) * self.entry_price) if self.entry_price > 0 else 0.0


@dataclass
class PortfolioState:
    """Complete portfolio state."""
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0       # cash + sum(abs(notional))
    cash: float = 0.0
    exposure_by_asset_class: Dict[AssetClass, float] = field(default_factory=dict)
    aggregate_exposure: float = 0.0   # sum(abs(weight)) = total leveraged exposure
    net_exposure: float = 0.0         # long_weight - short_weight
    leverage: float = 1.0             # aggregate_exposure / (total_value - cash)? Actually gross notional / NAV

    def recalc(self) -> None:
        """Recalculate all derived fields from positions."""
        if self.total_value <= 0:
            # Initialize with cash only
            self.total_value = self.cash
            for pos in self.positions.values():
                self.total_value += pos.notional
        else:
            # Use existing total_value as base (NAV) – weights are relative to NAV
            nav = self.total_value
            gross_notional = 0.0
            long_notional = 0.0
            short_notional = 0.0
            exposure_by_class = {cls: 0.0 for cls in AssetClass}

            for pos in self.positions.values():
                gross_notional += pos.notional
                if pos.quantity > 0:
                    long_notional += pos.notional
                else:
                    short_notional += pos.notional
                # Weight is notional / NAV (absolute)
                pos.weight = pos.notional / nav if nav > 0 else 0.0
                exposure_by_class[pos.asset_class] += pos.weight

            self.aggregate_exposure = gross_notional / nav if nav > 0 else 0.0
            self.net_exposure = (long_notional - short_notional) / nav if nav > 0 else 0.0
            self.leverage = self.aggregate_exposure  # gross exposure is leverage in long-only; for long/short it's gross/NAV
            self.exposure_by_asset_class = exposure_by_class


@dataclass
class CorrelationMatrix:
    """Correlation matrix for portfolio assets."""
    matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    assets: Set[str] = field(default_factory=set)

    def get_correlation(self, asset_a: str, asset_b: str) -> float:
        """Return correlation between two assets, default 0.5 if missing."""
        if asset_a not in self.matrix or asset_b not in self.matrix[asset_a]:
            return 0.5
        return self.matrix[asset_a][asset_b]

    def validate(self) -> bool:
        """Basic validation: symmetric, diagonal = 1, values in [-1,1]."""
        for a, row in self.matrix.items():
            for b, corr in row.items():
                if not -1.0 <= corr <= 1.0:
                    return False
                if a == b and abs(corr - 1.0) > 1e-6:
                    return False
                if b in self.matrix and a in self.matrix[b]:
                    if abs(corr - self.matrix[b][a]) > 1e-6:
                        return False
        return True


@dataclass
class PortfolioConstraints:
    """Institutional portfolio constraints."""
    max_position_size: float = 0.10           # 10% of NAV per position (absolute weight)
    max_portfolio_exposure: float = 0.30      # 30% total gross exposure (leverage)
    max_asset_class_exposure: float = 0.20    # 20% per asset class
    max_correlation_exposure: float = 0.15    # 15% for highly correlated assets (corr > 0.7)
    max_leverage: float = 2.0                 # maximum gross exposure / NAV
    correlation_threshold: float = 0.7        # correlation above this is "high"


# ---------- Portfolio Management Functions ----------

def update_portfolio_prices(portfolio: PortfolioState, price_updates: Dict[str, float]) -> None:
    """
    Update current prices for multiple symbols and recalc portfolio.
    """
    for symbol, new_price in price_updates.items():
        if symbol in portfolio.positions:
            portfolio.positions[symbol].update_prices(new_price)
    # Recalculate totals – must recompute NAV first
    total_notional = sum(pos.notional for pos in portfolio.positions.values())
    portfolio.total_value = portfolio.cash + total_notional
    portfolio.recalc()


def add_position(
    portfolio: PortfolioState,
    symbol: str,
    asset_class: AssetClass,
    quantity: float,
    entry_price: float,
    constraints: Optional[PortfolioConstraints] = None,
    correlation_matrix: Optional[CorrelationMatrix] = None,
) -> tuple[bool, List[str]]:
    """
    Add a new position or increase existing one (netting by symbol).
    Returns (success, violations_list).
    """
    if constraints is None:
        constraints = PortfolioConstraints()

    if quantity == 0:
        return False, ["Quantity cannot be zero"]

    if entry_price <= 0:
        return False, [f"Entry price must be positive, got {entry_price}"]

    # Compute proposed notional and weight relative to current NAV
    current_nav = portfolio.total_value if portfolio.total_value > 0 else portfolio.cash
    proposed_notional = abs(quantity) * entry_price
    proposed_weight = proposed_notional / current_nav if current_nav > 0 else 1.0

    # Determine net position change (handle existing position)
    existing = portfolio.positions.get(symbol)
    if existing:
        # Net quantity change
        new_quantity = existing.quantity + quantity
        # For weight check, we need to simulate new weight after adding
        # Simplified: use proposed_weight as increment (not exact but conservative)
        new_weight = existing.weight + proposed_weight
    else:
        new_weight = proposed_weight

    # Run constraint checks
    violations = []
    if new_weight > constraints.max_position_size:
        violations.append(
            f"Position size {new_weight:.1%} exceeds max {constraints.max_position_size:.1%}"
        )

    # Asset class exposure (sum of weights for that class after addition)
    current_class_exposure = portfolio.exposure_by_asset_class.get(asset_class, 0.0)
    new_class_exposure = current_class_exposure + proposed_weight
    if new_class_exposure > constraints.max_asset_class_exposure:
        violations.append(
            f"Asset class {asset_class.value} exposure {new_class_exposure:.1%} exceeds max {constraints.max_asset_class_exposure:.1%}"
        )

    # Correlation exposure: sum of weights of assets with correlation > threshold
    if correlation_matrix and symbol in correlation_matrix.assets:
        correlated_weight = 0.0
        for sym, pos in portfolio.positions.items():
            if sym == symbol:
                continue
            corr = correlation_matrix.get_correlation(symbol, sym)
            if corr > constraints.correlation_threshold:
                correlated_weight += pos.weight
        new_corr_exposure = correlated_weight + proposed_weight
        if new_corr_exposure > constraints.max_correlation_exposure:
            violations.append(
                f"Correlated exposure {new_corr_exposure:.1%} exceeds max {constraints.max_correlation_exposure:.1%}"
            )

    # Aggregate gross exposure after addition
    current_gross = portfolio.aggregate_exposure
    new_gross = current_gross + proposed_weight
    if new_gross > constraints.max_portfolio_exposure:
        violations.append(
            f"Aggregate exposure {new_gross:.1%} exceeds max {constraints.max_portfolio_exposure:.1%}"
        )
    if new_gross > constraints.max_leverage:
        violations.append(
            f"Leverage {new_gross:.2f}x exceeds max {constraints.max_leverage:.2f}x"
        )

    if violations:
        return False, violations

    # Apply the trade
    if existing:
        # Update existing position
        new_quantity = existing.quantity + quantity
        # Average entry price (weighted by quantity)
        total_cost = existing.quantity * existing.entry_price + quantity * entry_price
        if new_quantity != 0:
            new_entry = total_cost / new_quantity
        else:
            new_entry = 0.0
        existing.quantity = new_quantity
        existing.entry_price = new_entry
        existing.update_prices(existing.current_price)  # recalc notional, PnL
    else:
        # Create new position
        new_pos = Position(
            symbol=symbol,
            asset_class=asset_class,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
        )
        portfolio.positions[symbol] = new_pos

    # Recalculate portfolio totals
    total_notional = sum(pos.notional for pos in portfolio.positions.values())
    portfolio.total_value = portfolio.cash + total_notional
    portfolio.recalc()

    return True, []


def remove_position(portfolio: PortfolioState, symbol: str) -> bool:
    """Remove a position entirely from portfolio."""
    if symbol not in portfolio.positions:
        return False
    # Close position: add cash from notional at current price (simplified)
    pos = portfolio.positions[symbol]
    portfolio.cash += pos.quantity * pos.current_price  # positive if long, negative if short
    del portfolio.positions[symbol]
    # Recalc portfolio
    total_notional = sum(p.notional for p in portfolio.positions.values())
    portfolio.total_value = portfolio.cash + total_notional
    portfolio.recalc()
    return True


def calculate_portfolio_exposure(portfolio: PortfolioState) -> PortfolioState:
    """
    Recalculate aggregate portfolio exposure by asset class.
    This is essentially portfolio.recalc() but kept for compatibility.
    """
    portfolio.recalc()
    return portfolio


def check_portfolio_constraints(
    portfolio: PortfolioState,
    proposed_symbol: str,
    proposed_size: float,          # absolute weight (e.g., 0.05 for 5%)
    constraints: PortfolioConstraints,
    correlation_matrix: Optional[CorrelationMatrix] = None,
    asset_class: Optional[AssetClass] = None,
) -> tuple[bool, List[str]]:
    """
    Check if proposed position (by weight) violates constraints.
    This is a pre-trade validation without actually modifying portfolio.
    """
    violations = []

    if proposed_size <= 0:
        violations.append("Proposed size must be positive")

    # Position size
    if proposed_size > constraints.max_position_size:
        violations.append(
            f"Position size {proposed_size:.1%} exceeds max {constraints.max_position_size:.1%}"
        )

    # Aggregate exposure
    new_exposure = portfolio.aggregate_exposure + proposed_size
    if new_exposure > constraints.max_portfolio_exposure:
        violations.append(
            f"Aggregate exposure {new_exposure:.1%} exceeds max {constraints.max_portfolio_exposure:.1%}"
        )
    if new_exposure > constraints.max_leverage:
        violations.append(
            f"Leverage {new_exposure:.2f}x exceeds max {constraints.max_leverage:.2f}x"
        )

    # Asset class exposure (if provided)
    if asset_class:
        current_class_exp = portfolio.exposure_by_asset_class.get(asset_class, 0.0)
        new_class_exp = current_class_exp + proposed_size
        if new_class_exp > constraints.max_asset_class_exposure:
            violations.append(
                f"Asset class {asset_class.value} exposure {new_class_exp:.1%} exceeds max {constraints.max_asset_class_exposure:.1%}"
            )

    # Correlation exposure
    if correlation_matrix and proposed_symbol in correlation_matrix.assets:
        correlated_weight = 0.0
        for sym, pos in portfolio.positions.items():
            if sym == proposed_symbol:
                continue
            corr = correlation_matrix.get_correlation(proposed_symbol, sym)
            if corr > constraints.correlation_threshold:
                correlated_weight += pos.weight
        new_corr_exp = correlated_weight + proposed_size
        if new_corr_exp > constraints.max_correlation_exposure:
            violations.append(
                f"Correlated exposure {new_corr_exp:.1%} exceeds max {constraints.max_correlation_exposure:.1%}"
            )

    return len(violations) == 0, violations


def calculate_marginal_risk(
    portfolio: PortfolioState,
    proposed_symbol: str,
    proposed_size: float,          # notional weight (decimal)
    volatility: float,             # annualized volatility of proposed asset (decimal)
    correlation_matrix: Optional[CorrelationMatrix] = None,
) -> float:
    """
    Calculate marginal contribution to portfolio VaR (simplified using marginal VaR).
    Returns the approximate increase in portfolio volatility (in same units as volatility).
    """
    if proposed_size <= 0 or volatility <= 0:
        return 0.0

    # Current portfolio volatility (assuming weights are notional, correlation matrix needed)
    if not correlation_matrix or len(portfolio.positions) == 0:
        # No correlation info: treat as independent, marginal risk = proposed_size * volatility
        return proposed_size * volatility

    # Compute current portfolio variance (simplified: only existing assets)
    # For marginal risk, we compute derivative of portfolio std with respect to new position weight
    # d(σ_p)/dw_new = (w_new * σ_new^2 + Σ w_i * σ_i * σ_new * ρ_i_new) / σ_p
    # We'll return the marginal contribution to portfolio volatility (not VaR directly)
    # Use a conservative approximation: correlation-weighted average

    total_corr_contrib = 0.0
    for sym, pos in portfolio.positions.items():
        corr = correlation_matrix.get_correlation(proposed_symbol, sym)
        total_corr_contrib += pos.weight * corr

    # Current portfolio volatility (if we had it, we'd use; else estimate as weighted average)
    # For marginal risk we often use: ΔVaR ≈ proposed_size * volatility * (portfolio average correlation)
    avg_correlation = total_corr_contrib / max(1, len(portfolio.positions)) if portfolio.positions else 0.5
    marginal_risk = proposed_size * volatility * avg_correlation

    return marginal_risk


def get_portfolio_summary(portfolio: PortfolioState) -> Dict:
    """Generate portfolio summary for reporting."""
    return {
        "total_value": portfolio.total_value,
        "cash": portfolio.cash,
        "aggregate_exposure": portfolio.aggregate_exposure,
        "net_exposure": portfolio.net_exposure,
        "leverage": portfolio.leverage,
        "position_count": len(portfolio.positions),
        "exposure_by_asset_class": {k.value: v for k, v in portfolio.exposure_by_asset_class.items()},
        "positions": {sym: {"weight": pos.weight, "pnl_pct": pos.unrealized_pnl_pct}
                      for sym, pos in portfolio.positions.items()}
    }


# ---------- Unit Tests ----------
if __name__ == "__main__":
    print("Running portfolio layer tests...")

    # Create empty portfolio with cash
    port = PortfolioState(cash=100000.0)
    port.total_value = 100000.0
    port.recalc()

    constraints = PortfolioConstraints()

    # Test add long position
    success, errs = add_position(
        port, "BTC", AssetClass.CRYPTO, quantity=0.5, entry_price=50000.0,
        constraints=constraints
    )
    assert success, f"Add failed: {errs}"
    assert port.positions["BTC"].quantity == 0.5
    assert abs(port.aggregate_exposure - 0.25) < 0.01  # 0.5*50000 / 100000 = 0.25

    # Test update price
    update_portfolio_prices(port, {"BTC": 60000.0})
    assert abs(port.positions["BTC"].unrealized_pnl - 5000.0) < 1.0  # (60000-50000)*0.5 = 5000

    # Test constraint violation: exceed max position size
    success, errs = add_position(port, "ETH", AssetClass.CRYPTO, quantity=100, entry_price=2000.0,
                                 constraints=constraints)
    # 100*2000 = 200k notional, weight = 200k / (current NAV ~105k) ≈ 1.9 → violation
    assert not success
    assert any("exceeds max" in e for e in errs)

    # Test correlation exposure
    corr_mat = CorrelationMatrix(assets={"BTC", "ETH"})
    corr_mat.matrix = {
        "BTC": {"BTC": 1.0, "ETH": 0.85},
        "ETH": {"ETH": 1.0, "BTC": 0.85},
    }
    # Create new portfolio with just cash
    port2 = PortfolioState(cash=100000.0, total_value=100000.0)
    add_position(port2, "BTC", AssetClass.CRYPTO, 0.5, 50000.0, constraints)
    # Try to add ETH with size 0.12 (12%) – correlated exposure would be 0.25+0.12=0.37 > 0.15
    success, errs = check_portfolio_constraints(
        port2, "ETH", 0.12, constraints, correlation_matrix=corr_mat, asset_class=AssetClass.CRYPTO
    )
    assert not success
    assert any("Correlated exposure" in e for e in errs)

    # Test marginal risk
    marginal = calculate_marginal_risk(port2, "ETH", 0.10, volatility=0.5, correlation_matrix=corr_mat)
    assert marginal > 0

    # Test removal
    remove_position(port, "BTC")
    assert "BTC" not in port.positions
    assert abs(port.aggregate_exposure) < 1e-6

    print("All portfolio tests passed.")