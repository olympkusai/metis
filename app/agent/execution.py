"""
Execution Engine - Institutional-Grade Order Execution
────────────────────────────────────────────────
Realistic execution modeling with VWAP/TWAP strategies,
slippage estimation, and market impact calculation.
"""

import math
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ExecutionStrategy(str, Enum):
    """Execution strategy types."""
    MARKET = "market"          # Immediate execution (aggressive)
    VWAP = "vwap"              # Volume-Weighted Average Price
    TWAP = "twap"              # Time-Weighted Average Price
    LIMIT = "limit"            # Limit order at specific price
    ICEBERG = "iceberg"        # Hidden large orders (only visible portion)


@dataclass
class Order:
    """Order specification."""
    symbol: str
    side: str                  # "buy" or "sell"
    quantity: float            # Positive quantity
    price: Optional[float] = None   # Required for LIMIT orders
    strategy: ExecutionStrategy = ExecutionStrategy.MARKET
    time_horizon: Optional[float] = None  # Minutes for TWAP/VWAP
    slices: Optional[int] = None          # Number of slices for algo orders


@dataclass
class SlippageEstimate:
    """Slippage estimation for order execution."""
    base_slippage_bps: float          # Half-spread in bps
    market_impact_bps: float          # Market impact in bps
    total_slippage_bps: float         # Total slippage (bps)
    estimated_execution_price: float
    confidence_interval: Tuple[float, float]  # (lower, upper) price range


@dataclass
class ExecutionPlan:
    """Complete execution plan."""
    orders: List[Order]
    strategy: ExecutionStrategy
    expected_avg_price: float
    total_slippage_bps: float
    execution_time_estimate: float    # Minutes
    liquidity_score: float            # 0 to 1, higher is better


def calculate_market_impact(
    order_size: float,          # Notional value of order (currency)
    avg_daily_volume: float,    # Notional average daily volume
    volatility: float,          # Annualized volatility (decimal, e.g., 0.5 = 50%)
    spread_bps: float = 10.0,   # Current bid-ask spread in basis points
    eta_base: float = 0.3,      # Base impact coefficient (Almgren-Chriss)
    alpha: float = 0.7,         # Exponent for participation rate
    max_impact_bps: float = 500.0
) -> float:
    """
    Calculate market impact using Almgren-Chriss model (simplified).
    Institutional: models price impact based on order size and liquidity.

    Returns market impact in basis points.
    """
    # Validate inputs
    if order_size <= 0 or avg_daily_volume <= 0:
        return 0.0
    if volatility <= 0:
        volatility = 0.2  # Assume 20% if unknown

    participation_rate = min(1.0, order_size / avg_daily_volume)

    # Higher volatility increases market impact (wider distribution of liquidity)
    eta = eta_base * (volatility / 0.2)  # Normalized to 20% vol baseline
    eta = min(eta, 1.0)  # Cap at 1.0

    market_impact_bps = eta * (participation_rate ** alpha) * 10000.0

    # Spread component: larger participation eats more spread layers
    spread_impact = spread_bps * participation_rate

    total_impact = market_impact_bps + spread_impact
    return min(total_impact, max_impact_bps)


def calculate_vwap_execution(
    symbol: str,
    side: str,
    total_quantity: float,
    current_price: float,
    avg_daily_volume: float,
    volatility: float,
    spread_bps: float = 10.0,
    execution_window_minutes: float = 60.0,
    num_slices: int = 10
) -> ExecutionPlan:
    """
    Calculate VWAP execution plan (Volume-Weighted Average Price).
    Institutional: slices order over time proportional to expected volume profile.
    Simplified here as equal slices but could be enhanced with volume curve.
    """
    if total_quantity <= 0 or current_price <= 0:
        raise ValueError("Quantity and price must be positive")
    if avg_daily_volume <= 0:
        avg_daily_volume = 1_000_000  # Fallback for unknown liquidity

    slice_size = total_quantity / num_slices
    slice_interval = execution_window_minutes / num_slices

    orders = []
    for i in range(num_slices):
        orders.append(Order(
            symbol=symbol,
            side=side,
            quantity=slice_size,
            strategy=ExecutionStrategy.VWAP,
            time_horizon=slice_interval
        ))

    # Estimate market impact for total order
    order_notional = total_quantity * current_price
    slippage_bps = calculate_market_impact(
        order_size=order_notional,
        avg_daily_volume=avg_daily_volume,
        volatility=volatility,
        spread_bps=spread_bps
    )

    # VWAP typically reduces impact by 40-60% vs market order
    vwap_slippage_bps = slippage_bps * 0.5

    if side == "buy":
        expected_avg_price = current_price * (1 + vwap_slippage_bps / 10000.0)
    else:
        expected_avg_price = current_price * (1 - vwap_slippage_bps / 10000.0)

    # Liquidity score: how easily the order can be absorbed
    participation_rate = order_notional / avg_daily_volume
    liquidity_score = max(0.0, min(1.0, 1.0 - participation_rate))

    return ExecutionPlan(
        orders=orders,
        strategy=ExecutionStrategy.VWAP,
        expected_avg_price=expected_avg_price,
        total_slippage_bps=vwap_slippage_bps,
        execution_time_estimate=execution_window_minutes,
        liquidity_score=liquidity_score
    )


def calculate_twap_execution(
    symbol: str,
    side: str,
    total_quantity: float,
    current_price: float,
    avg_daily_volume: float,
    volatility: float,
    spread_bps: float = 10.0,
    execution_window_minutes: float = 60.0,
    num_slices: int = 12
) -> ExecutionPlan:
    """
    Calculate TWAP execution plan (Time-Weighted Average Price).
    Institutional: executes equal slices over time regardless of volume profile.
    """
    if total_quantity <= 0 or current_price <= 0:
        raise ValueError("Quantity and price must be positive")
    if avg_daily_volume <= 0:
        avg_daily_volume = 1_000_000

    slice_size = total_quantity / num_slices
    slice_interval = execution_window_minutes / num_slices

    orders = []
    for i in range(num_slices):
        orders.append(Order(
            symbol=symbol,
            side=side,
            quantity=slice_size,
            strategy=ExecutionStrategy.TWAP,
            time_horizon=slice_interval
        ))

    order_notional = total_quantity * current_price
    slippage_bps = calculate_market_impact(
        order_size=order_notional,
        avg_daily_volume=avg_daily_volume,
        volatility=volatility,
        spread_bps=spread_bps
    )

    # TWAP reduces impact less than VWAP (typically 30-50% reduction)
    twap_slippage_bps = slippage_bps * 0.65

    if side == "buy":
        expected_avg_price = current_price * (1 + twap_slippage_bps / 10000.0)
    else:
        expected_avg_price = current_price * (1 - twap_slippage_bps / 10000.0)

    participation_rate = order_notional / avg_daily_volume
    liquidity_score = max(0.0, min(1.0, 1.0 - participation_rate * 0.8))

    return ExecutionPlan(
        orders=orders,
        strategy=ExecutionStrategy.TWAP,
        expected_avg_price=expected_avg_price,
        total_slippage_bps=twap_slippage_bps,
        execution_time_estimate=execution_window_minutes,
        liquidity_score=liquidity_score
    )


def estimate_slippage(
    order: Order,
    current_price: float,
    avg_daily_volume: float,
    volatility: float,
    spread_bps: float = 10.0
) -> SlippageEstimate:
    """
    Estimate slippage for a single order.
    Institutional: provides confidence interval for execution price.
    """
    if order.quantity <= 0 or current_price <= 0:
        raise ValueError("Order quantity and current price must be positive")

    order_notional = order.quantity * current_price
    base_slippage = spread_bps / 2.0   # Half the spread is typical for marketable orders

    market_impact = calculate_market_impact(
        order_size=order_notional,
        avg_daily_volume=avg_daily_volume,
        volatility=volatility,
        spread_bps=spread_bps
    )

    total_slippage_bps = base_slippage + market_impact

    # Calculate execution price with slippage
    price_adjustment = total_slippage_bps / 10000.0
    if order.side == "buy":
        estimated_price = current_price * (1 + price_adjustment)
        lower_bound = current_price * (1 + price_adjustment * 0.5)
        upper_bound = current_price * (1 + price_adjustment * 1.5)
    else:
        estimated_price = current_price * (1 - price_adjustment)
        lower_bound = current_price * (1 - price_adjustment * 1.5)
        upper_bound = current_price * (1 - price_adjustment * 0.5)

    return SlippageEstimate(
        base_slippage_bps=base_slippage,
        market_impact_bps=market_impact,
        total_slippage_bps=total_slippage_bps,
        estimated_execution_price=estimated_price,
        confidence_interval=(lower_bound, upper_bound)
    )


def recommend_execution_strategy(
    order_size: float,          # Notional value
    avg_daily_volume: float,
    volatility: float,
    urgency: str = "normal",    # "low", "normal", "high"
    spread_bps: float = 10.0
) -> ExecutionStrategy:
    """
    Recommend optimal execution strategy based on order characteristics.
    Institutional: algorithmic strategy selection.
    """
    if order_size <= 0 or avg_daily_volume <= 0:
        return ExecutionStrategy.MARKET

    participation_rate = order_size / avg_daily_volume

    # High urgency overrides everything
    if urgency == "high":
        return ExecutionStrategy.MARKET

    # Low urgency: prefer algorithms to minimize impact
    if urgency == "low":
        if participation_rate > 0.05:
            return ExecutionStrategy.TWAP
        else:
            return ExecutionStrategy.VWAP

    # Normal urgency
    if participation_rate < 0.01:      # Very small order
        return ExecutionStrategy.MARKET
    elif participation_rate < 0.05:    # Small-medium
        return ExecutionStrategy.VWAP
    elif participation_rate < 0.15:    # Large
        return ExecutionStrategy.TWAP
    else:                              # Very large
        return ExecutionStrategy.ICEBERG


def calculate_optimal_slice_count(
    total_quantity: float,
    avg_daily_volume: float,
    execution_window_minutes: float,
    min_slice_seconds: float = 30.0
) -> int:
    """
    Calculate optimal number of slices for algorithmic execution.
    Balances execution speed vs market impact.
    """
    if total_quantity <= 0 or avg_daily_volume <= 0:
        return 5

    participation_rate = total_quantity / avg_daily_volume

    # Base slices on participation rate
    if participation_rate < 0.02:
        base_slices = 5
    elif participation_rate < 0.05:
        base_slices = 10
    elif participation_rate < 0.10:
        base_slices = 15
    else:
        base_slices = 20

    # Adjust for execution window: longer window allows more slices
    max_slices_by_time = max(1, int(execution_window_minutes / (min_slice_seconds / 60.0)))
    optimal = min(base_slices, max_slices_by_time)

    return max(1, optimal)


def simulate_market_execution(
    order: Order,
    current_price: float,
    avg_daily_volume: float,
    volatility: float,
    spread_bps: float = 10.0
) -> Tuple[float, float]:
    """
    Simulate execution of a market order.
    Returns (executed_price, realized_slippage_bps).
    """
    estimate = estimate_slippage(order, current_price, avg_daily_volume, volatility, spread_bps)
    # Use the estimated price as the expected execution price
    executed_price = estimate.estimated_execution_price
    realized_slippage = estimate.total_slippage_bps
    return executed_price, realized_slippage


# ---------- Unit Tests ----------
if __name__ == "__main__":
    print("Running execution engine tests...")

    # Test market impact
    impact = calculate_market_impact(1_000_000, 10_000_000, 0.5, spread_bps=10)
    assert 0 < impact < 500, f"Impact out of range: {impact}"

    # Test VWAP execution
    plan = calculate_vwap_execution(
        symbol="BTC",
        side="buy",
        total_quantity=0.5,
        current_price=50000,
        avg_daily_volume=1_000_000_000,  # 1B notional
        volatility=0.6,
        spread_bps=15,
        execution_window_minutes=120,
        num_slices=20
    )
    assert len(plan.orders) == 20
    assert plan.total_slippage_bps >= 0
    assert plan.liquidity_score > 0.9

    # Test TWAP execution
    plan2 = calculate_twap_execution(
        symbol="ETH",
        side="sell",
        total_quantity=100,
        current_price=3000,
        avg_daily_volume=500_000_000,
        volatility=0.4,
        spread_bps=12,
        execution_window_minutes=90
    )
    assert plan2.strategy == ExecutionStrategy.TWAP
    assert plan2.expected_avg_price < 3000  # sell side

    # Test slippage estimate
    order = Order(symbol="AAPL", side="buy", quantity=1000)
    slippage = estimate_slippage(
        order, current_price=150.0,
        avg_daily_volume=50_000_000,
        volatility=0.25,
        spread_bps=5
    )
    assert slippage.total_slippage_bps > 0
    assert slippage.confidence_interval[0] < slippage.estimated_execution_price < slippage.confidence_interval[1]

    # Test strategy recommendation
    strat = recommend_execution_strategy(1_000_000, 100_000_000, 0.3, urgency="normal")
    assert strat == ExecutionStrategy.VWAP  # 1% participation -> VWAP

    strat_high = recommend_execution_strategy(10_000_000, 100_000_000, 0.3, urgency="high")
    assert strat_high == ExecutionStrategy.MARKET

    # Test optimal slice count
    slices = calculate_optimal_slice_count(100_000, 1_000_000, 60)
    assert 5 <= slices <= 20

    print("All execution engine tests passed.")