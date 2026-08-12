"""Structured tracing for agent executions.

Each request gets an AgentTrace that collects events throughout the
ReAct loop: LLM calls, tool calls, tool results, errors, effort selection.
The trace is persisted to the database and a summary is sent via SSE.

The trace_id comes from the Nike gateway's X-Request-ID header, so traces
correlate with gateway logs.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class TraceEvent:
    """A single event in the agent execution trace."""
    ts: float          # seconds since trace start
    type: str          # "effort" | "llm_call" | "tool_call" | "tool_result" | "error" | "prefetch" | "action" | "final"
    iteration: int     # ReAct loop iteration (0-indexed)
    data: dict[str, Any] = field(default_factory=dict)


class AgentTrace:
    """Collects structured events for a single agent execution.

    Usage:
        trace = AgentTrace(trace_id="abc-123", user_id="user-1", session_id="sess-1")
        trace.add_event("effort", 0, level="low", model="gpt-4o-mini")
        trace.add_event("llm_call", 0, input_tokens=1200, output_tokens=45, latency_ms=850)
        trace.add_event("tool_call", 0, tool_name="list_transactions_filtered", args={...})
        trace.add_event("tool_result", 0, tool_name="list_transactions_filtered", result_size=1200, latency_ms=320)
        trace.add_event("final", 1, answer_len=250)
        summary = trace.summary()
    """

    def __init__(
        self,
        trace_id: str = "",
        user_id: str = "",
        session_id: str = "",
        user_message: str = "",
    ):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.user_id = user_id
        self.session_id = session_id
        self.user_message = user_message
        self.events: list[TraceEvent] = []
        self.start_time = time.time()
        self.final_answer: str = ""
        self.status: str = "running"

    def add_event(self, type: str, iteration: int = 0, **data: Any) -> None:
        """Add an event to the trace."""
        self.events.append(TraceEvent(
            ts=round(time.time() - self.start_time, 3),
            type=type,
            iteration=iteration,
            data=data,
        ))

    def set_final_answer(self, answer: str, status: str = "success") -> None:
        """Set the final answer and mark the trace as complete."""
        self.final_answer = answer
        self.status = status

    def summary(self) -> dict[str, Any]:
        """Build a summary dict with aggregated metrics."""
        total_time = time.time() - self.start_time

        llm_calls = [e for e in self.events if e.type == "llm_call"]
        tool_calls = [e for e in self.events if e.type == "tool_call"]
        tool_results = [e for e in self.events if e.type == "tool_result"]
        errors = [e for e in self.events if e.type == "error"]
        effort_events = [e for e in self.events if e.type == "effort"]

        input_tokens = sum(e.data.get("input_tokens", 0) for e in llm_calls)
        output_tokens = sum(e.data.get("output_tokens", 0) for e in llm_calls)
        cost_usd = sum(e.data.get("cost_usd", 0.0) for e in llm_calls)

        max_iteration = max((e.iteration for e in self.events), default=0)

        # Tool latency breakdown
        tool_latencies = {}
        for tr in tool_results:
            name = tr.data.get("tool_name", "unknown")
            lat = tr.data.get("latency_ms", 0)
            if name not in tool_latencies:
                tool_latencies[name] = []
            tool_latencies[name].append(lat)

        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "status": self.status,
            "total_time_ms": round(total_time * 1000, 0),
            "iterations": max_iteration + 1,
            "llm_calls": len(llm_calls),
            "tool_calls": len(tool_calls),
            "tool_names": [e.data.get("tool_name") for e in tool_calls],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": round(cost_usd, 6),
            "effort": effort_events[0].data.get("level", "unknown") if effort_events else "unknown",
            "model": effort_events[0].data.get("model", "unknown") if effort_events else "unknown",
            "errors": len(errors),
            "error_details": [e.data.get("message", "") for e in errors],
            "tool_latencies_ms": {
                name: {
                    "calls": len(lats),
                    "total_ms": sum(lats),
                    "avg_ms": round(sum(lats) / len(lats), 0) if lats else 0,
                }
                for name, lats in tool_latencies.items()
            },
            "user_message": self.user_message[:200],
            "answer_len": len(self.final_answer),
        }

    def full_trace(self) -> dict[str, Any]:
        """Full trace with all events (for storage/debugging)."""
        return {
            **self.summary(),
            "events": [asdict(e) for e in self.events],
            "final_answer": self.final_answer[:2000],
        }

    def sse_summary(self) -> dict[str, Any]:
        """Compact summary for the SSE trace event (sent to frontend)."""
        s = self.summary()
        return {
            "type": "trace",
            "trace_id": s["trace_id"],
            "total_time_ms": s["total_time_ms"],
            "iterations": s["iterations"],
            "llm_calls": s["llm_calls"],
            "tool_calls": s["tool_calls"],
            "tool_names": s["tool_names"],
            "input_tokens": s["input_tokens"],
            "output_tokens": s["output_tokens"],
            "cost_usd": s["cost_usd"],
            "effort": s["effort"],
            "model": s["model"],
            "status": s["status"],
        }
