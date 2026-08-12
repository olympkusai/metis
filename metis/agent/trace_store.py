"""Trace persistence — saves agent execution traces to the database.

Stores structured traces in the `agent_traces` table for debugging,
analytics, and evaluation. Best-effort: failures don't break the chat.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from metis.agent.tracing import AgentTrace
from metis.memory.conversation_history import _get_conversation_db_pool

logger = logging.getLogger(__name__)


async def save_trace(trace: AgentTrace) -> None:
    """Save a trace to the agent_traces table.

    Best-effort: logs a warning on failure but doesn't raise.
    """
    pool = _get_conversation_db_pool()
    summary = trace.summary()
    full = trace.full_trace()

    await pool.execute(
        """
        INSERT INTO agent_traces
            (trace_id, user_id, session_id, user_message, final_answer,
             total_time_ms, iterations, llm_calls, tool_calls,
             input_tokens, output_tokens, cost_usd, effort, model,
             status, summary)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16::jsonb)
        ON CONFLICT (trace_id) DO UPDATE SET
            final_answer = EXCLUDED.final_answer,
            total_time_ms = EXCLUDED.total_time_ms,
            iterations = EXCLUDED.iterations,
            llm_calls = EXCLUDED.llm_calls,
            tool_calls = EXCLUDED.tool_calls,
            input_tokens = EXCLUDED.input_tokens,
            output_tokens = EXCLUDED.output_tokens,
            cost_usd = EXCLUDED.cost_usd,
            status = EXCLUDED.status,
            summary = EXCLUDED.summary
        """,
        trace.trace_id,
        trace.user_id,
        trace.session_id,
        summary["user_message"],
        full["final_answer"],
        int(summary["total_time_ms"]),
        summary["iterations"],
        summary["llm_calls"],
        summary["tool_calls"],
        summary["input_tokens"],
        summary["output_tokens"],
        summary["cost_usd"],
        summary["effort"],
        summary["model"],
        summary["status"],
        json.dumps(full),
    )


async def get_trace(trace_id: str) -> dict[str, Any] | None:
    """Get a single trace by ID."""
    pool = _get_conversation_db_pool()
    rows = await pool.fetch(
        "SELECT * FROM agent_traces WHERE trace_id = $1",
        trace_id,
    )
    if not rows:
        return None
    row = rows[0]
    return _row_to_dict(row)


async def list_traces(
    user_id: str = "",
    session_id: str = "",
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List traces, optionally filtered by user or session."""
    pool = _get_conversation_db_pool()

    if user_id:
        rows = await pool.fetch(
            "SELECT * FROM agent_traces WHERE user_id = $1 "
            "ORDER BY created_at DESC LIMIT $2 OFFSET $3",
            user_id, limit, offset,
        )
    elif session_id:
        rows = await pool.fetch(
            "SELECT * FROM agent_traces WHERE session_id = $1 "
            "ORDER BY created_at DESC LIMIT $2 OFFSET $3",
            session_id, limit, offset,
        )
    else:
        rows = await pool.fetch(
            "SELECT * FROM agent_traces "
            "ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            limit, offset,
        )

    return [_row_to_dict(r) for r in rows]


async def get_trace_stats(user_id: str = "", limit: int = 100) -> dict[str, Any]:
    """Aggregate stats from recent traces."""
    pool = _get_conversation_db_pool()

    if user_id:
        rows = await pool.fetch(
            "SELECT total_time_ms, iterations, llm_calls, tool_calls, "
            "input_tokens, output_tokens, cost_usd, effort, status "
            "FROM agent_traces WHERE user_id = $1 "
            "ORDER BY created_at DESC LIMIT $2",
            user_id, limit,
        )
    else:
        rows = await pool.fetch(
            "SELECT total_time_ms, iterations, llm_calls, tool_calls, "
            "input_tokens, output_tokens, cost_usd, effort, status "
            "FROM agent_traces ORDER BY created_at DESC LIMIT $1",
            limit,
        )

    if not rows:
        return {"count": 0}

    n = len(rows)
    total_time = sum(r["total_time_ms"] or 0 for r in rows)
    total_input = sum(r["input_tokens"] or 0 for r in rows)
    total_output = sum(r["output_tokens"] or 0 for r in rows)
    total_cost = sum(float(r["cost_usd"] or 0) for r in rows)
    errors = sum(1 for r in rows if r["status"] == "error")

    # Effort distribution
    effort_counts: dict[str, int] = {}
    for r in rows:
        e = r["effort"] or "unknown"
        effort_counts[e] = effort_counts.get(e, 0) + 1

    return {
        "count": n,
        "avg_time_ms": round(total_time / n, 0) if n else 0,
        "avg_iterations": round(sum(r["iterations"] or 0 for r in rows) / n, 1) if n else 0,
        "avg_llm_calls": round(sum(r["llm_calls"] or 0 for r in rows) / n, 1) if n else 0,
        "avg_tool_calls": round(sum(r["tool_calls"] or 0 for r in rows) / n, 1) if n else 0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": round(total_cost, 4),
        "error_rate": round(errors / n, 2) if n else 0,
        "effort_distribution": effort_counts,
    }


def _row_to_dict(row) -> dict[str, Any]:
    """Convert a database row to a dict, parsing JSONB fields."""
    result = dict(row)
    # Parse summary JSONB
    summary = result.get("summary")
    if summary and isinstance(summary, str):
        result["summary"] = json.loads(summary)
    return result
