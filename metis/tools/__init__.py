"""
Tools for LangChain - LOCAL CALCULATIONS (NO EXTERNAL API CALLS).
Uses CalculationEngine and MarketCandleQueries for all calculations.
"""

# Import local tools (use these for production - no external API calls)
from metis.tools.local import (
    get_live_price,
    get_indicators,
    calculate_risk,
    get_feature_rsi,
    get_feature_macd,
    get_feature_bollinger,
    get_feature_volatility,
    get_feature_sharpe,
    get_feature_cvar,
    get_feature_max_drawdown,
    get_feature_sma,
    get_feature_ema_return,
    set_db_pool,
)

# Finance tools — separate list, never merged into all_tools (crypto). Keeps
# the crypto pipeline's blast radius at zero and avoids exposing finance
# tools to the LLM in nodes that have nothing to do with personal finance.
from metis.tools.finance import (
    finance_tools,
    get_spending_by_category,
    get_cashflow,
    get_budget_progress,
    get_goal_summary,
    get_recurrences_due,
    list_transactions_filtered,
    set_auth_token,
)

# Export local tools as the default
all_tools = [
    get_live_price,
    get_indicators,
    calculate_risk,
    get_feature_rsi,
    get_feature_macd,
    get_feature_bollinger,
    get_feature_volatility,
    get_feature_sharpe,
    get_feature_cvar,
    get_feature_max_drawdown,
    get_feature_sma,
    get_feature_ema_return,
]

__all__ = [
    "all_tools",
    "set_db_pool",
    "get_live_price",
    "get_indicators",
    "calculate_risk",
    "get_feature_rsi",
    "get_feature_macd",
    "get_feature_bollinger",
    "get_feature_volatility",
    "get_feature_sharpe",
    "get_feature_cvar",
    "get_feature_max_drawdown",
    "get_feature_sma",
    "get_feature_ema_return",
    "finance_tools",
    "get_spending_by_category",
    "get_cashflow",
    "get_budget_progress",
    "get_goal_summary",
    "get_recurrences_due",
    "list_transactions_filtered",
    "set_auth_token",
]
