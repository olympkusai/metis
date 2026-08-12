"""
Metis — Personal Finance Agent v2 (ReAct loop)
────────────────────────────────────────────────
Single-node graph that uses AgentRuntime for a free-form ReAct loop.
The LLM decides which tools to call (read + write) and when to stop.

  ┌─ finance_agent_v2 ──────────────────────────────┐
  │  1. Build tool catalog (finance read + Hermes)  │
  │  2. Fetch user profile (pre-step, not a tool)   │
  │  3. Run AgentRuntime with unified prompt        │
  │  4. Close Hermes client                         │
  │  5. Return final answer                         │
  └──────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from metis.agent.graph import FinanceAgentState, NextAction
from metis.agent.runtime import AgentRuntime, AgentResult, NodeStatus, _append_reasoning
from metis.agent.effort import EffortConfig, get_effort_config, get_effort_config_async
from metis.tools import build_tool_catalog, close_hermes_client, set_auth_token, set_user_id, set_session_id
from metis.pluto_client import PlutoApiError, get_pluto_client
from metis.soter_client import SoterApiError, get_soter_client
from metis.agent.finance_prompts import (
    _FINANCE_AGENT_V2_SYSTEM,
    _FINANCE_AGENT_V2_COMPRESSED,
    build_personalization_directives,
)

from metis.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. SINGLE NODE — finance_agent_v2
# ─────────────────────────────────────────────

async def finance_agent_v2_node(
    state: FinanceAgentState,
    config: RunnableConfig,
) -> dict:
    """ReAct loop node: fetches context, builds tools, runs AgentRuntime.

    This single node replaces the v1 fixed pipeline (orchestrator → context →
    data_gathering → analysis → synthesis). The LLM decides freely which tools
    to call and when to produce the final answer.
    """
    # ── Auth guard ──────────────────────────────────────────────
    if not state.auth_token:
        error_msg = (
            "Não consegui acessar seus dados financeiros — sessão não "
            "autenticada. Tente fazer login novamente."
        )
        return {
            **state.model_dump(),
            "final_answer": error_msg,
            "next_action": NextAction.FINALIZE,
            "messages": state.messages + [AIMessage(content=error_msg)],
        }

    # ── 1. Set auth token + build tool catalog ──────────────────
    set_auth_token(state.auth_token)
    set_user_id(state.user_id)
    # Extract metadata from config (session_id, trace, callbacks)
    _config_metadata = {}
    if config:
        _config_metadata = config.get("metadata", {}) if hasattr(config, "get") else {}
        if isinstance(_config_metadata, dict):
            set_session_id(_config_metadata.get("session_id", ""))
    tools, hermes_client = await build_tool_catalog(state.auth_token)

    # ── 1b. Determine effort level ──────────────────────────────
    # Extract the user's latest message for auto-selection
    user_message = ""
    if state.messages:
        for m in reversed(state.messages):
            if isinstance(m, HumanMessage):
                user_message = getattr(m, "content", "") or ""
                break
            elif isinstance(m, dict) and m.get("type") in ("human", "user"):
                user_message = m.get("content", "") or ""
                break

    # Effort level from config metadata (resolved by chat.py) or fallback to auto
    effort_level = "auto"
    if config:
        metadata = config.get("metadata", {}) if hasattr(config, "get") else {}
        if isinstance(metadata, dict):
            effort_level = metadata.get("effort", "auto")

    # If chat.py already resolved the effort (low/medium/high), use it directly.
    # If still "auto" (e.g. called outside chat.py), use regex fallback.
    effort = get_effort_config(effort_level, user_message)
    logger.info(f"[finance_agent_v2] effort={effort.level} model={effort.model} "
                f"msg='{user_message[:60]}'")

    # ── 2. Fetch user profile + accounts + categories (pre-step, not a tool) ─
    # Pre-fetch reports based on effort level to reduce ReAct iterations
    try:
        client = get_pluto_client()
        soter = get_soter_client()
        # Categories is best-effort: if the endpoint doesn't exist or fails,
        # we still proceed with profile + accounts.
        profile, accounts = await asyncio.gather(
            client.get_financial_profile(token=state.auth_token),
            client.list_accounts(token=state.auth_token),
        )
        categories: dict | list = []
        try:
            categories = await client.list_categories(token=state.auth_token)
        except Exception:
            pass

        # Pre-fetch reports based on effort config (reduces ReAct iterations)
        spending_data = None
        cashflow_data = None
        budget_data = None
        goals_data = None
        prefetch_tasks = []
        if effort.prefetch_spending:
            prefetch_tasks.append(("spending", client.spending_by_category(token=state.auth_token)))
        if effort.prefetch_cashflow:
            prefetch_tasks.append(("cashflow", client.cashflow(token=state.auth_token)))
        if effort.prefetch_budget:
            prefetch_tasks.append(("budget", client.budget_progress(token=state.auth_token)))
        if effort.prefetch_goals:
            prefetch_tasks.append(("goals", client.goal_summary(token=state.auth_token)))

        if prefetch_tasks:
            prefetch_results = await asyncio.gather(
                *[t[1] for t in prefetch_tasks], return_exceptions=True,
            )
            for (name, _), result in zip(prefetch_tasks, prefetch_results):
                if isinstance(result, Exception):
                    logger.warning(f"[finance_agent_v2] prefetch {name} failed: {result}")
                elif name == "spending":
                    spending_data = result
                elif name == "cashflow":
                    cashflow_data = result
                elif name == "budget":
                    budget_data = result
                elif name == "goals":
                    goals_data = result

        # AI personalization from Soter (best-effort)
        personalization: dict = {}
        try:
            app_id = await soter.get_app_id_by_client_id(
                client_id="pluto", token=state.auth_token,
            )
            pers = await soter.get_personalization(
                token=state.auth_token, app_id=app_id,
            )
            if pers:
                personalization = pers
        except SoterApiError as e:
            print(f"[finance_agent_v2] Soter personalization unavailable: {e}")
        except Exception as e:
            print(f"[finance_agent_v2] Soter error: {e}")

    except PlutoApiError as e:
        # Sanitize: don't leak internal API names, UUIDs, or status codes
        error_detail = str(e)
        if "404" in error_detail or "not found" in error_detail.lower():
            error_msg = (
                "Ainda não tenho seus dados financeiros cadastrados. "
                "Configure seu perfil financeiro no app para que eu possa "
                "te ajudar com contas, gastos e orçamento."
            )
        else:
            error_msg = (
                "Não consegui acessar seus dados financeiros agora. "
                "Tente novamente em alguns instantes."
            )
        print(f"[finance_agent_v2] PlutoApiError: {error_detail}")
        if hermes_client is not None:
            await close_hermes_client(hermes_client)
        return {
            **state.model_dump(),
            "final_answer": error_msg,
            "next_action": NextAction.FINALIZE,
            "messages": state.messages + [AIMessage(content=error_msg)],
        }

    accounts_data = (
        accounts.get("accounts", accounts) if isinstance(accounts, dict) else accounts
    )

    # ── 3. Build extra_context ──────────────────────────────────
    now_utc = datetime.now(timezone.utc)
    today = now_utc.strftime("%Y-%m-%d")
    time_utc = now_utc.strftime("%H:%M")

    accounts_json = json.dumps(accounts_data, ensure_ascii=False, default=str)
    profile_json = json.dumps(profile, ensure_ascii=False, default=str)

    # Build a compact category list (id + name only) for the context
    categories_data = (
        categories.get("categories", categories)
        if isinstance(categories, dict)
        else categories
    ) if categories else []
    categories_compact = []
    if isinstance(categories_data, list):
        for cat in categories_data:
            if isinstance(cat, dict):
                categories_compact.append({
                    "id": cat.get("id", cat.get("category_id", "")),
                    "name": cat.get("name", ""),
                })
    categories_json = json.dumps(categories_compact, ensure_ascii=False, default=str)

    extra_context = (
        f"[DATA ATUAL] {today} (UTC) — use esta data quando o usuário não "
        f"especificar uma data. NUNCA use data futura.\n"
        f"[HORÁRIO ATUAL] {time_utc} (UTC)\n\n"
        f"[CONTAS DO USUÁRIO]\n{accounts_json}\n\n"
        f"[CATEGORIAS DO USUÁRIO]\n{categories_json}\n\n"
        f"[PERFIL FINANCEIRO DO USUÁRIO]\n{profile_json}"
    )

    # Add pre-fetched reports to context (saves ReAct iterations)
    if spending_data is not None:
        extra_context += f"\n\n[GASTOS POR CATEGORIA (mês corrente)]\n{json.dumps(spending_data, ensure_ascii=False, default=str)}"
    if cashflow_data is not None:
        extra_context += f"\n\n[FLUXO DE CAIXA (mês corrente)]\n{json.dumps(cashflow_data, ensure_ascii=False, default=str)}"
    if budget_data is not None:
        extra_context += f"\n\n[PROGRESSO DO ORÇAMENTO]\n{json.dumps(budget_data, ensure_ascii=False, default=str)}"
    if goals_data is not None:
        extra_context += f"\n\n[RESUMO DE METAS]\n{json.dumps(goals_data, ensure_ascii=False, default=str)}"

    # ── 3b. RAG memory recall (auto-injection) ──────────────────
    # For medium/high effort, search previous conversations for relevant
    # memories. Excludes the current session (already in context).
    # Low effort skips this (simple queries don't need historical context).
    _trace = _config_metadata.get("trace") if isinstance(_config_metadata, dict) else None
    if effort.level != "low" and user_message and state.user_id:
        try:
            from metis.agent.memory_rag import recall_similar_messages
            session_id = _config_metadata.get("session_id", "") if isinstance(_config_metadata, dict) else ""
            memories = await recall_similar_messages(
                query=user_message,
                user_id=state.user_id,
                session_id=session_id,
                limit=3,
                threshold=0.75,
            )
            if memories:
                memory_parts = []
                for m in memories:
                    date_str = m["created_at"].strftime("%d/%m/%Y")
                    role_label = "Usuário" if m["role"] == "user" else "Assistente"
                    memory_parts.append(
                        f"[{date_str}] {role_label}: {m['content'][:200]}"
                    )
                extra_context += (
                    "\n\n[MEMÓRIAS RELEVANTES DE CONVERSAS ANTERIORES]\n"
                    + "\n".join(memory_parts)
                )
                logger.info(
                    f"[finance_agent_v2] RAG: {len(memories)} memories injected "
                    f"(effort={effort.level})"
                )
                if _trace is not None:
                    _trace.add_event("memory_recall", 0,
                        memories_found=len(memories),
                        top_similarity=memories[0]["similarity"] if memories else 0)
        except Exception as e:
            logger.warning(f"[finance_agent_v2] RAG memory recall failed: {e}")

    # ── 4. Apply personalization directives to system prompt ────
    # Use compressed prompt for low effort, full prompt otherwise
    system_prompt = (
        _FINANCE_AGENT_V2_COMPRESSED if effort.compressed_prompt
        else _FINANCE_AGENT_V2_SYSTEM
    )
    pers = personalization
    profile_currency = profile.get("currency") if isinstance(profile, dict) else None
    if pers and isinstance(pers, dict):
        directives = build_personalization_directives(
            tone=pers.get("tone", "friendly"),
            display_name=pers.get("display_name"),
            personality_notes=pers.get("personality_notes"),
            language=pers.get("language", "pt_BR"),
            obfuscation_level=pers.get("obfuscation_level", "none"),
            profile_currency=profile_currency,
        )
        if directives:
            system_prompt = system_prompt + "\n\n" + directives

    # ── 5. Run AgentRuntime ─────────────────────────────────────
    # Extract callbacks from RunnableConfig.metadata if present.
    # chat.py injects them so the runtime can:
    # - stream_callback: stream final answer tokens in real time via SSE
    # - reasoning_callback: emit tool-call reasoning in real time so the
    #   UI shows "Consultando X..." BEFORE the final answer appears
    # Both bypass LangGraph's stream_mode (which mixes intermediate ReAct
    # reasoning with the final answer and fires node_execution too late).
    stream_callback = None
    reasoning_callback = None
    action_callback = None
    trace = None
    if config:
        metadata = config.get("metadata", {}) if hasattr(config, "get") else {}
        if isinstance(metadata, dict):
            stream_callback = metadata.get("stream_callback")
            reasoning_callback = metadata.get("reasoning_callback")
            action_callback = metadata.get("action_callback")
            trace = metadata.get("trace")

    try:
        runtime = AgentRuntime(
            system_prompt=system_prompt,
            tools=tools,
            model=effort.model,
            temperature=effort.temperature,
            max_iterations=effort.max_iterations,
            enable_cot=False,
            node_name="finance_agent_v2",
            stream_callback=stream_callback,
            reasoning_callback=reasoning_callback,
            action_callback=action_callback,
            trace=trace,
            effort=effort.level,
        )

        result: AgentResult = await runtime.run(
            messages=state.messages,
            extra_context=extra_context,
            reasoning_trail=state.reasoning_trail,
            config=config,
        )
    except Exception as e:
        error_msg = (
            "Não consegui processar sua solicitação agora. "
            "Tente novamente em alguns instantes."
        )
        print(f"[finance_agent_v2] AgentRuntime error: {e}")
        if trace is not None:
            trace.add_event("error", 0, message=str(e), phase="runtime")
            trace.set_final_answer(error_msg, status="error")
        if hermes_client is not None:
            await close_hermes_client(hermes_client)
        return {
            **state.model_dump(),
            "final_answer": error_msg,
            "next_action": NextAction.FINALIZE,
            "messages": state.messages + [AIMessage(content=error_msg)],
        }
    finally:
        # ── 6. Close Hermes client ──────────────────────────────
        if hermes_client is not None:
            try:
                await close_hermes_client(hermes_client)
            except Exception as e:
                print(f"[finance_agent_v2] Error closing Hermes client: {e}")

    # ── 7. Return state update ──────────────────────────────────
    answer = result.answer or "Não consegui gerar uma resposta. Tente reformular sua pergunta."

    print(
        f"[finance_agent_v2] tools_called={len(result.steps)}, "
        f"answer_len={len(answer)}, status={result.status.value}"
    )

    return {
        **state.model_dump(),
        "final_answer": answer,
        "cot": result.cot,
        "reasoning_trail": _append_reasoning(
            state.reasoning_trail, "finance_agent_v2", result.cot, result.status,
        ),
        "intermediate_steps_agent": result.steps,
        "next_action": NextAction.FINALIZE,
        "messages": state.messages + [AIMessage(content=answer)],
    }


# ─────────────────────────────────────────────
# 2. GRAPH ASSEMBLY
# ─────────────────────────────────────────────

def build_finance_graph_v2() -> Any:
    workflow = StateGraph(FinanceAgentState)
    workflow.add_node("finance_agent_v2", finance_agent_v2_node)
    workflow.set_entry_point("finance_agent_v2")
    workflow.add_edge("finance_agent_v2", END)
    return workflow.compile()


# ─────────────────────────────────────────────
# 3. SINGLETON + GETTER
# ─────────────────────────────────────────────

_finance_graph_v2: Any | None = None


def get_finance_agent_graph_v2() -> Any:
    global _finance_graph_v2
    if _finance_graph_v2 is None:
        _finance_graph_v2 = build_finance_graph_v2()
    return _finance_graph_v2
