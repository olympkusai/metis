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
from metis.tools import build_tool_catalog, close_hermes_client, set_auth_token
from metis.pluto_client import PlutoApiError, get_pluto_client
from metis.soter_client import SoterApiError, get_soter_client
from metis.agent.finance_prompts import (
    _FINANCE_AGENT_V2_SYSTEM,
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
    tools, hermes_client = await build_tool_catalog(state.auth_token)

    # ── 2. Fetch user profile + accounts (pre-step, not a tool) ─
    try:
        client = get_pluto_client()
        soter = get_soter_client()
        profile, accounts = await asyncio.gather(
            client.get_financial_profile(token=state.auth_token),
            client.list_accounts(token=state.auth_token),
        )

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

    extra_context = (
        f"[DATA ATUAL] {today} (UTC) — use esta data quando o usuário não "
        f"especificar uma data. NUNCA use data futura.\n"
        f"[HORÁRIO ATUAL] {time_utc} (UTC)\n\n"
        f"[CONTAS DO USUÁRIO]\n{accounts_json}\n\n"
        f"[PERFIL FINANCEIRO DO USUÁRIO]\n{profile_json}"
    )

    # ── 4. Apply personalization directives to system prompt ────
    system_prompt = _FINANCE_AGENT_V2_SYSTEM
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
    try:
        runtime = AgentRuntime(
            system_prompt=system_prompt,
            tools=tools,
            model="gpt-4o",
            temperature=0.1,
            max_iterations=20,
            enable_cot=False,
            node_name="finance_agent_v2",
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
