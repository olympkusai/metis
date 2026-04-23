"""
Tools for LangChain - LOCAL CALCULATIONS (NO EXTERNAL API CALLS).
Uses CalculationEngine and MarketCandleQueries for all calculations.
"""

# Import local tools (use these for production - no external API calls)
from app.tools.local import (
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
]
