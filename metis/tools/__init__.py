"""
Tools for LangChain — Finance tools (Pluto report lookups).
"""

from __future__ import annotations

from typing import Any

from metis.mcp_client import close_hermes_client, discover_hermes_tools
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

__all__ = [
    "finance_tools",
    "get_spending_by_category",
    "get_cashflow",
    "get_budget_progress",
    "get_goal_summary",
    "get_recurrences_due",
    "list_transactions_filtered",
    "set_auth_token",
    "build_tool_catalog",
    "close_hermes_client",
]


async def build_tool_catalog(auth_token: str) -> tuple[list, Any]:
    """Build a unified tool catalog: finance read tools + Hermes write tools.

    This is the single source of truth for tools available to the AgentRuntime
    in the v2 agentic loop. The LLM sees ALL tools and decides freely which
    to call.

    Args:
        auth_token: User's JWT, forwarded to Hermes for MCP tool discovery.
                    Also set as ContextVar for finance tools (via set_auth_token).

    Returns:
        (tools, hermes_client) — the unified list of LangChain BaseTool
        instances, plus the Hermes MCP client (which must be kept alive
        while tools are in use, and closed via close_hermes_client when done).
        If Hermes is unreachable, returns (finance_tools, None) — read-only
        mode, don't break the agent.
    """
    # Set the ContextVar so finance read tools can authenticate Pluto calls.
    set_auth_token(auth_token)

    # Discover Hermes write tools (best-effort: returns ([], None) if down).
    hermes_tools, hermes_client = await discover_hermes_tools(auth_token)

    combined = finance_tools + hermes_tools
    return combined, hermes_client
