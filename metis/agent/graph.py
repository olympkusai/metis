"""
Metis — Personal Finance Agent
────────────────────────────────
LangGraph multi-agent system for personal finance assistance.

  ┌─ FinanceOrchestrator ──────────────────────────────────────┐
  │  Classifies intent (greeting / in-scope / out-of-scope)    │
  │  ↓                                                          │
  │  FinanceContext  — fetches profile + accounts from Pluto   │
  │  ↓                                                          │
  │  FinanceReasoning — LLM with finance_tools (Pluto reports) │
  │  ↓                                                          │
  │  Finalize        — terminal node                            │
  └────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ConfigDict

from metis.tools import finance_tools, set_auth_token
from metis.pluto_client import PlutoApiError, get_pluto_client
from metis.soter_client import SoterApiError, get_soter_client
from metis.mcp_client import discover_hermes_tools, close_hermes_client
from metis.agent.finance_prompts import (
    _FINANCE_GREETING,
    _FINANCE_OUT_OF_SCOPE,
    _FINANCE_ORCHESTRATOR_SYSTEM,
    _FINANCE_ACTION_SYSTEM,
    _FINANCE_DATA_GATHERING_SYSTEM,
    _FINANCE_ANALYSIS_SYSTEM,
    _FINANCE_SYNTHESIS_SYSTEM,
    build_personalization_directives,
)
from metis.config import get_settings
from metis.utils.cost_tracker import get_cost_tracker


# ─────────────────────────────────────────────
# 1. DOMAIN ENUMS
# ─────────────────────────────────────────────

class NextAction(str, Enum):
    FINANCE_CONTEXT   = "finance_context"
    FINANCE_REASONING = "finance_reasoning"
    ACTION            = "action"
    FINALIZE          = "finalize"


# ─────────────────────────────────────────────
# 2. TYPED STATE  (Pydantic-backed)
# ─────────────────────────────────────────────

class FinanceAgentState(BaseModel):
    # Conversation
    messages:           list[Any]          = Field(default_factory=list)
    user_id:            str                = ""
    # Bearer JWT forwarded from the frontend, per-request only — never persisted
    # to conversation history (see metis/memory/conversation_history.py).
    auth_token:         str                = ""

    # Finance persona context (fetched by finance_context_node from Pluto)
    finance_profile:    dict               = Field(default_factory=dict)
    finance_accounts:   list               = Field(default_factory=list)

    # AI personalization (fetched by finance_context_node from Soter)
    personalization:    dict               = Field(default_factory=dict)

    # Data gathered by data_gathering_node (tool results)
    gathered_data:      str                = ""

    # Analysis produced by analysis_node (technical reasoning)
    analysis_text:      str                = ""

    # Routing
    next_action:        NextAction         = NextAction.FINANCE_CONTEXT
    intermediate_steps_global: list[tuple[str, str]] = Field(default_factory=list)
    intermediate_steps_agent: list[tuple[str, str]] = Field(default_factory=list)
    final_answer:       str                = ""

    # Tool call deduplication cache (per run)
    tool_cache:         dict[str, str]     = Field(default_factory=dict)

    # Chain-of-thought (per node, for streaming)
    cot:                str                = ""
    # Reasoning trail: cumulative <thought> de cada nó, alimenta nós downstream.
    reasoning_trail:    list[tuple[str, str, str]] = Field(default_factory=list)

    # MCP client for Hermes (kept alive during the run, closed in finalize)
    _hermes_client:     Any                = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ─────────────────────────────────────────────
# 3. SHARED INFRASTRUCTURE
# ─────────────────────────────────────────────

_BASE_MODEL = "gpt-4o"


def _make_llm(model: str = _BASE_MODEL, temperature: float = 0.1, **kw) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.openai_api_key,
        streaming=True,
        **kw,
    )


finance_tool_map = {t.name: t for t in finance_tools}

logger = logging.getLogger(__name__)


def _latest_user_message(messages: list) -> HumanMessage | None:
    """Retorna a ÚLTIMA `HumanMessage` da lista, ou None se não houver."""
    return next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )


def _extract_cot_and_answer(content: str) -> tuple[str, str]:
    """Extrai CoT e answer de uma resposta formatada como:
    <thought>...</thought><answer>...</answer>

    Retorna (cot, answer). Se não encontrar formato, retorna ("", content).
    Fallback: se answer vazio, usa o conteúdo sem as tags thought.
    """
    thought_match = re.search(r'<thought>(.*?)</thought>', content, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)

    if thought_match and answer_match:
        cot = thought_match.group(1).strip()
        answer = answer_match.group(1).strip()
        # Fallback: if answer is empty, use content without thought tags
        if not answer:
            answer = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL).strip()
            answer = re.sub(r'</?answer>', '', answer).strip()
        return cot, answer
    elif thought_match:
        cot = thought_match.group(1).strip()
        answer = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL).strip()
        # Also strip any answer tags
        answer = re.sub(r'</?answer>', '', answer).strip()
        if not answer:
            answer = cot  # last resort: use the thought as the answer
        return cot, answer
    else:
        return "", content


class NodeStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    NO_COT = "no_cot"


def _failure_cot(node: str, reason: str) -> str:
    """Pseudo-thought para nós que falharam tecnicamente."""
    reason = (reason or "").strip() or "motivo desconhecido"
    return f"Nó '{node}' não pôde concluir: {reason}"


def _synthesize_cot_from_steps(steps: list[tuple[str, str]]) -> str:
    """Fallback: sintetiza um CoT a partir das tool calls executadas quando
    o LLM não produziu as tags <thought>/<answer> espontaneamente."""
    if not steps:
        return ""
    tool_names = []
    for label, _result in steps:
        name = label.split(" [cached]")[0].strip()
        if name not in tool_names:
            tool_names.append(name)
    if not tool_names:
        return ""
    if len(tool_names) == 1:
        return f"Consultei {tool_names[0]} para obter os dados necessários."
    return f"Consultei {', '.join(tool_names[:-1])} e {tool_names[-1]} para reunir os dados necessários."


def _append_reasoning(
    trail: list[tuple[str, str, str]],
    node_name: str,
    cot: str,
    status: NodeStatus = NodeStatus.OK,
) -> list[tuple[str, str, str]]:
    """Append a node's CoT to the reasoning trail."""
    cot = (cot or "").strip()
    if not cot and status == NodeStatus.OK:
        return trail
    if not cot:
        cot = f"(sem detalhe — status={status.value})"
    return trail + [(node_name, cot, status.value)]


def _cache_key(name: str, args: dict) -> str:
    try:
        return f"{name}|{json.dumps(args, sort_keys=True, default=str)}"
    except Exception:
        return f"{name}|{repr(sorted(args.items()))}"


# ─────────────────────────────────────────────
# 4. TOOL EXECUTION
# ─────────────────────────────────────────────

async def _execute_tools(
    last_ai: AIMessage,
    steps: list,
    cache: dict[str, str] | None = None,
    tool_map_override: dict | None = None,
) -> tuple[list[ToolMessage], list]:
    """Executa tool calls em paralelo, com dedup cache (por-run).

    `tool_map_override` permite a um agente resolver tool calls contra um
    conjunto de tools diferente do global.
    """
    tool_messages: list[ToolMessage] = []
    steps = list(steps)
    cache = cache if cache is not None else {}
    active_tool_map = tool_map_override if tool_map_override is not None else finance_tool_map

    async def _run_one(tc):
        name = tc["name"]
        args = dict(tc.get("args") or {})
        key = _cache_key(name, args)
        if key in cache:
            return name, tc["id"], cache[key], True
        tool = active_tool_map.get(name)
        if tool is None:
            return name, tc["id"], f"[ERR] Tool '{name}' não encontrada.", False
        result = await tool.ainvoke(args)
        cache[key] = str(result)
        return name, tc["id"], result, False

    results = await asyncio.gather(
        *[_run_one(tc) for tc in last_ai.tool_calls],
        return_exceptions=True,
    )

    for item, tc in zip(results, last_ai.tool_calls):
        if isinstance(item, Exception):
            name = tc["name"]
            result = f"[ERR] Tool '{name}' exceção: {item}"
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            steps.append((name, str(result)))
        else:
            name, call_id, result, was_cached = item
            content = str(result)
            tool_messages.append(ToolMessage(content=content, tool_call_id=call_id))
            label = f"{name} [cached]" if was_cached else name
            steps.append((label, content))
    return tool_messages, steps


async def _run_agent_loop(
    state: FinanceAgentState,
    llm,
    system_prompt: str,
    extra_context: str = "",
    clear_steps: bool = False,
    node_name: str = "unknown",
    enable_cot: bool = False,
    tool_map_override: dict | None = None,
    config: RunnableConfig | None = None,
) -> tuple[str, str, NodeStatus, list[ToolMessage], list]:
    """Loop LLM → tool call → result para um agente especializado."""
    original_query = _latest_user_message(state.messages)

    if node_name == "action":
        tool_instruction = (
            "\n\nIMPORTANTE: A mensagem do usuário é um pedido de AÇÃO. "
            "Interprete o que ele quer e chame a ferramenta apropriada "
            "IMEDIATAMENTE. Não pergunte qual operação ele quer realizar — "
            "a mensagem já é o pedido. Chame a tool e depois confirme."
        )
    else:
        tool_instruction = (
            "\n\nIMPORTANTE: Você tem acesso a ferramentas. Você DEVE chamar as "
            "ferramentas apropriadas antes de fornecer sua análise final. Não "
            "responda sem chamar as ferramentas primeiro."
        )

    cot_instruction = ""
    if enable_cot:
        cot_instruction = (
            "\n\nFORMATO DE RESPOSTA OBRIGATÓRIO (único):\n"
            "<thought>Resumo CONCISO do seu raciocínio (max 2-3 frases). "
            "Ex: 'Obtive o indicador X que está em Y, indicando Z.'</thought>\n"
            "<answer>Sua resposta final em Markdown aqui (siga as "
            "diretrizes de estilo acima)</answer>"
        )

    enhanced_prompt = system_prompt + tool_instruction + cot_instruction

    msgs: list = [SystemMessage(content=enhanced_prompt)]

    if state.reasoning_trail:
        trail_block = (
            "RACIOCÍNIO DOS AGENTES ANTERIORES (use como contexto, não copie literalmente):\n"
            + "\n".join(
                f"  [{name}]({status}) {thought}"
                for name, thought, status in state.reasoning_trail
            )
        )
        msgs.append(HumanMessage(content=trail_block))

    if extra_context:
        msgs.append(HumanMessage(content=f"[CONTEXTO DO PIPELINE]\n{extra_context}"))

    # User's message goes LAST so the LLM treats it as the current request
    if original_query:
        msgs.append(original_query)

    # Debug: log messages for action node
    if node_name == "action":
        for i, m in enumerate(msgs):
            content_preview = str(m.content)[:150] if hasattr(m, 'content') and m.content else "(empty)"
            logger.info(f"[action_node] msg[{i}] type={type(m).__name__}: {content_preview}")

    all_tool_msgs: list[ToolMessage] = []
    steps = [] if clear_steps else list(state.intermediate_steps_agent)
    MAX_ITERATIONS = 6
    MAX_RETRIES = 3

    previous_step_count = len(steps)
    no_progress_count = 0

    # Accumulate each tool call as a reasoning step so the CoT is a real
    # chain (step 1 → step 2 → … → final thought) instead of just the
    # final summary.
    reasoning_lines: list[str] = []

    # Human-readable names for each finance tool, so the CoT shows
    # "Consultando gastos por categoria" instead of "Consultando get_spending_by_category".
    _TOOL_LABELS: dict[str, str] = {
        "get_spending_by_category": "gastos por categoria",
        "get_cashflow": "fluxo de caixa",
        "get_budget_progress": "progresso do orçamento",
        "get_goal_summary": "resumo de metas",
        "get_recurrences_due": "contas recorrentes a vencer",
        "list_transactions_filtered": "transações filtradas",
    }

    def _build_cot(final_cot: str) -> str:
        """Combine accumulated reasoning lines with the final thought."""
        if final_cot and reasoning_lines:
            return "\n".join(reasoning_lines + [final_cot])
        if final_cot:
            return final_cot
        if reasoning_lines:
            return "\n".join(reasoning_lines)
        return ""

    for iteration in range(MAX_ITERATIONS):
        response = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await llm.ainvoke(msgs, config=config, timeout=30)

                cost_tracker = get_cost_tracker()
                if hasattr(response, 'response_metadata'):
                    metadata = response.response_metadata
                    input_tokens = metadata.get('token_usage', {}).get('prompt_tokens', 0)
                    output_tokens = metadata.get('token_usage', {}).get('completion_tokens', 0)
                    model_name = getattr(llm, 'model_name', 'unknown') or getattr(llm, 'model', 'unknown')
                    if input_tokens > 0 or output_tokens > 0:
                        cost_tracker.add_call(model_name, input_tokens, output_tokens, node_name)

                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    error_msg = f"[LLM ERROR after {MAX_RETRIES} retries]: {str(e)}"
                    return error_msg, _failure_cot(node_name, error_msg), NodeStatus.ERROR, all_tool_msgs, steps
                backoff_time = 0.5 * (2 ** attempt)
                await asyncio.sleep(backoff_time)
                continue

        if response is None:
            error_msg = "[LLM ERROR] Failed to get response after retries"
            return error_msg, _failure_cot(node_name, error_msg), NodeStatus.ERROR, all_tool_msgs, steps

        # Debug: log LLM response for action node
        if node_name == "action":
            has_tc = hasattr(response, 'tool_calls') and bool(response.tool_calls)
            content_preview = str(response.content)[:200] if response.content else "(empty)"
            logger.info(f"[action_node] LLM response: has_tool_calls={has_tc}, content_preview={content_preview}")
            if has_tc:
                logger.info(f"[action_node] tool_calls: {response.tool_calls}")

        msgs.append(response)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Record each tool call as a reasoning step
            for tc in response.tool_calls:
                tool_name = tc.get("name", "unknown")
                friendly = _TOOL_LABELS.get(tool_name, tool_name.replace("_", " "))
                args = dict(tc.get("args") or {})
                # Build a short human-readable description of the call
                if args:
                    arg_str = ", ".join(
                        f"{k}={v}" for k, v in args.items()
                        if k not in ("auth_token",) and v
                    )
                    reasoning_lines.append(f"Consultando {friendly}({arg_str})" if arg_str else f"Consultando {friendly}")
                else:
                    reasoning_lines.append(f"Consultando {friendly}")
            tool_msgs, steps = await _execute_tools(
                response, steps, cache=state.tool_cache,
                tool_map_override=tool_map_override,
            )
            msgs.extend(tool_msgs)
            all_tool_msgs.extend(tool_msgs)
        elif hasattr(response, 'content'):
            # AIMessage / AIMessageChunk — use .content (the actual text),
            # NOT model_dump() which serializes the entire message object
            # (metadata, usage, id, …) as JSON.
            cot, answer = _extract_cot_and_answer(response.content) if enable_cot else ("", response.content)
            if enable_cot and not cot and steps:
                cot = _synthesize_cot_from_steps(steps)
            final_cot = _build_cot(cot) if enable_cot else ""
            status = NodeStatus.OK if (final_cot or not enable_cot) else NodeStatus.NO_COT
            return answer, final_cot, status, all_tool_msgs, steps
        elif isinstance(response, (str, dict)):
            if isinstance(response, dict):
                content = json.dumps(response, ensure_ascii=False, indent=2)
            else:
                content = str(response)
            cot, answer = _extract_cot_and_answer(content) if enable_cot else ("", content)
            if enable_cot and not cot and steps:
                cot = _synthesize_cot_from_steps(steps)
            final_cot = _build_cot(cot) if enable_cot else ""
            status = NodeStatus.OK if (final_cot or not enable_cot) else NodeStatus.NO_COT
            return answer, final_cot, status, all_tool_msgs, steps
        else:
            # Last-resort fallback for unexpected response types
            content = str(response)
            cot, answer = _extract_cot_and_answer(content) if enable_cot else ("", content)
            if enable_cot and not cot and steps:
                cot = _synthesize_cot_from_steps(steps)
            final_cot = _build_cot(cot) if enable_cot else ""
            status = NodeStatus.OK if (final_cot or not enable_cot) else NodeStatus.NO_COT
            return answer, final_cot, status, all_tool_msgs, steps

        if len(steps) == previous_step_count:
            no_progress_count += 1
            if no_progress_count >= 2:
                error_msg = "[AGENT LOOP] No progress detected - aborting to prevent infinite loop"
                return error_msg, _failure_cot(node_name, error_msg), NodeStatus.ERROR, all_tool_msgs, steps
        else:
            no_progress_count = 0
            previous_step_count = len(steps)

    # Força finalização se exceder iterações
    try:
        final = await llm.ainvoke(msgs + [HumanMessage(content="Sintetize os resultados obtidos até agora.")], config=config)
    except Exception as e:
        error_msg = f"[LLM ERROR on forced finalize]: {str(e)}"
        return error_msg, _failure_cot(node_name, error_msg), NodeStatus.ERROR, all_tool_msgs, steps
    cot, answer = _extract_cot_and_answer(final.content) if enable_cot else ("", final.content)
    if enable_cot and not cot and steps:
        cot = _synthesize_cot_from_steps(steps)
    final_cot = _build_cot(cot) if enable_cot else ""
    status = NodeStatus.OK if (final_cot or not enable_cot) else NodeStatus.NO_COT
    return answer, final_cot, status, all_tool_msgs, steps


# ─────────────────────────────────────────────
# 5. INTENT CLASSIFICATION (deterministic)
# ─────────────────────────────────────────────

_PURE_GREETINGS: frozenset[str] = frozenset({
    # PT-BR
    "eae", "eai", "e ai", "oi", "oii", "ola", "opa", "salve", "fala", "fala ai", "fala mano",
    "tudo bem", "tudo bom", "como vai", "como esta", "como voce esta", "como vc esta",
    "beleza", "blz", "tmj",
    "eae tudo bem", "eae tudo bom", "oi tudo bem", "oi tudo bom",
    "ola tudo bem", "ola tudo bom", "opa tudo bem", "opa tudo bom",
    "eae mano", "oi mano", "fala mano",
    "bom dia", "boa tarde", "boa noite",
    # EN
    "hi", "hi there", "hello", "hello there", "hey", "hey there", "sup", "yo",
    "how are you", "how are you doing", "whats up", "what is up",
    "good morning", "good afternoon", "good evening",
})

_GREETING_STARTERS: tuple[str, ...] = (
    "eae", "eai", "oi", "ola", "opa", "salve", "fala",
    "hey", "hi", "hello", "yo", "sup",
    "bom dia", "boa tarde", "boa noite",
    "good morning", "good afternoon", "good evening",
    "tudo bem", "tudo bom", "como vai", "como esta", "como voce esta", "como vc esta",
    "beleza", "blz",
)


def _normalize_text(text: str) -> str:
    """Normaliza texto: lowercase + sem acentos + sem pontuação + whitespace colapsado."""
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _classify_intent(question: Any) -> str:
    """Classifica a intenção do usuário de forma determinística.

    Retorna:
        "greeting": saudação pura
        "task":     pergunta com conteúdo real
    """
    if not question or not str(question).strip():
        return "greeting"

    normalized = _normalize_text(str(question))
    if not normalized:
        return "greeting"

    tokens = set(normalized.split())

    if normalized in _PURE_GREETINGS:
        return "greeting"

    for starter in _GREETING_STARTERS:
        if normalized == starter or normalized.startswith(starter + " "):
            if len(tokens) <= 6:
                return "greeting"

    return "task"


# ─────────────────────────────────────────────
# 6. AGENT NODES
# ─────────────────────────────────────────────

async def finance_orchestrator_node(state: FinanceAgentState) -> dict:
    """Classifica intenção: greeting, out-of-scope, analysis (FINANCE_OK)
    ou action (ACTION). Action vai direto para o action_node, pulando o
    pipeline de análise."""
    user_question = state.messages[-1].content if state.messages else ""

    intent = _classify_intent(user_question)
    if intent == "greeting":
        return {
            **state.model_dump(),
            "final_answer": _FINANCE_GREETING,
            "next_action": NextAction.FINALIZE,
            "messages": state.messages + [AIMessage(content=_FINANCE_GREETING)],
        }

    simple_llm = _make_llm(model="gpt-4o-mini", temperature=0)
    scope_validation = await simple_llm.ainvoke([
        SystemMessage(content=_FINANCE_ORCHESTRATOR_SYSTEM),
        HumanMessage(content=user_question),
    ])
    scope_result = scope_validation.content.strip().upper()

    if scope_result == "ACTION":
        return {**state.model_dump(), "next_action": NextAction.ACTION}

    if scope_result != "FINANCE_OK":
        return {
            **state.model_dump(),
            "final_answer": _FINANCE_OUT_OF_SCOPE,
            "next_action": NextAction.FINALIZE,
            "messages": state.messages + [AIMessage(content=_FINANCE_OUT_OF_SCOPE)],
        }

    return {**state.model_dump(), "next_action": NextAction.FINANCE_CONTEXT}


async def finance_context_node(state: FinanceAgentState) -> dict:
    """Busca o perfil financeiro + contas do usuário no Pluto."""
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

    try:
        client = get_pluto_client()
        soter = get_soter_client()
        profile, accounts = await asyncio.gather(
            client.get_financial_profile(token=state.auth_token),
            client.list_accounts(token=state.auth_token),
        )

        # Fetch AI personalization from Soter (best-effort — don't fail
        # the whole node if Soter is unreachable or preferences don't
        # exist yet).
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
            print(f"[finance_context_node] Soter personalization unavailable: {e}")
        except Exception as e:
            print(f"[finance_context_node] Soter error: {e}")

        cot_text = "Buscando seu perfil financeiro e contas..."
        return {
            **state.model_dump(),
            "finance_profile": profile,
            "finance_accounts": accounts.get("accounts", accounts) if isinstance(accounts, dict) else accounts,
            "personalization": personalization,
            "cot": cot_text,
            "reasoning_trail": _append_reasoning(state.reasoning_trail, "finance_context", cot_text),
        }
    except PlutoApiError as e:
        # Sanitize: don't leak internal API names, UUIDs, or status codes
        # to the end user. Log the full error server-side instead.
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
        print(f"[finance_context_node] PlutoApiError: {error_detail}")
        return {
            **state.model_dump(),
            "final_answer": error_msg,
            "next_action": NextAction.FINALIZE,
            "messages": state.messages + [AIMessage(content=error_msg)],
        }


async def action_node(state: FinanceAgentState, config: RunnableConfig) -> dict:
    """Executa operações de escrita/gestão (criar, atualizar, excluir, pagar,
    etc.) usando as tools do Hermes (MCP server).

    Este nó é ativado quando o orchestrator classifica a intenção como ACTION.
    Diferente do pipeline de análise (data_gathering → analysis → synthesis),
    este nó chama a tool apropriada e responde diretamente ao usuário — sem
    passar por análise financeira.
    """
    if not state.auth_token:
        return {
            **state.model_dump(),
            "final_answer": "Não consegui executar essa operação — sessão não autenticada. Tente fazer login novamente.",
            "next_action": NextAction.FINALIZE,
            "messages": state.messages + [AIMessage(content="Não consegui executar essa operação — sessão não autenticada.")],
        }

    set_auth_token(state.auth_token)

    # Discover write tools from Hermes
    hermes_tools, hermes_client = await discover_hermes_tools(state.auth_token)

    if not hermes_tools:
        return {
            **state.model_dump(),
            "final_answer": "Não consegui conectar ao serviço de operações financeiras agora. Tente novamente em alguns instantes.",
            "next_action": NextAction.FINALIZE,
            "messages": state.messages + [AIMessage(content="Não consegui conectar ao serviço de operações financeiras agora.")],
        }

    tool_map = {t.name: t for t in hermes_tools}
    llm = _make_llm(model="gpt-4o", temperature=0.1).bind_tools(hermes_tools)

    # Debug: log tool binding info
    tool_names = [t.name for t in hermes_tools]
    logger.info(f"[action_node] bound {len(hermes_tools)} tools: {tool_names[:5]}...")
    if hermes_tools:
        first = hermes_tools[0]
        logger.info(f"[action_node] first tool: name={first.name}, desc={first.description[:80] if first.description else 'None'}")
        if hasattr(first, 'args_schema') and first.args_schema:
            try:
                schema = first.args_schema.model_json_schema()
                logger.info(f"[action_node] first tool schema keys: {list(schema.get('properties', {}).keys())}")
            except Exception as e:
                logger.info(f"[action_node] schema error: {e}")

    # Build context with the user's accounts so the LLM can resolve
    # account_id automatically when the user says "minha conta principal".
    context_block = json.dumps(
        {"accounts": state.finance_accounts},
        ensure_ascii=False, default=str,
    )

    content, cot, status, tool_msgs, steps = await _run_agent_loop(
        state,
        llm,
        system_prompt=_FINANCE_ACTION_SYSTEM,
        extra_context=f"[CONTAS DO USUÁRIO]\n{context_block}",
        clear_steps=True,
        node_name="action",
        enable_cot=False,
        tool_map_override=tool_map,
        config=config,
    )

    await close_hermes_client(hermes_client)

    # The action node produces the final answer directly — no analysis/synthesis
    answer = content or "Não consegui executar a operação. Pode tentar novamente?"

    print(f"[action_node] tools_called={len(steps)}, answer_len={len(answer)}")

    return {
        **state.model_dump(),
        "final_answer": answer,
        "cot": "",
        "reasoning_trail": _append_reasoning(state.reasoning_trail, "action", "", status),
        "intermediate_steps_agent": steps,
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "next_action": NextAction.FINALIZE,
        "messages": state.messages + [AIMessage(content=answer)],
    }


async def data_gathering_node(state: FinanceAgentState, config: RunnableConfig) -> dict:
    """Coleta dados via ferramentas de LEITURA. Não analisa, não responde ao
    usuário. Apenas chama as ferramentas necessárias e devolve os dados brutos.

    As tools de escrita (Hermes MCP) são tratadas pelo action_node, não aqui.
    """
    set_auth_token(state.auth_token)

    llm = _make_llm(model="gpt-4o", temperature=0.1).bind_tools(finance_tools)

    context_block = json.dumps(
        {"financial_profile": state.finance_profile, "accounts": state.finance_accounts},
        ensure_ascii=False, default=str,
    )

    content, cot, status, tool_msgs, steps = await _run_agent_loop(
        state,
        llm,
        system_prompt=_FINANCE_DATA_GATHERING_SYSTEM,
        extra_context=f"[PERFIL FINANCEIRO E CONTAS DO USUÁRIO]\n{context_block}",
        clear_steps=True,
        node_name="data_gathering",
        enable_cot=True,
        tool_map_override=finance_tool_map,
        config=config,
    )

    # Gather all tool results as a single data block for the analysis node
    gathered = "\n\n".join(
        f"[{label}]\n{result[:3000]}" for label, result in steps
    )
    if not gathered:
        # No tools called — use profile + accounts as the data
        gathered = context_block

    print(f"[data_gathering_node] tools_called={len(steps)}, cot_len={len(cot)}, data_len={len(gathered)}")

    return {
        **state.model_dump(),
        "gathered_data": gathered,
        "cot": cot,
        "reasoning_trail": _append_reasoning(state.reasoning_trail, "data_gathering", cot, status),
        "intermediate_steps_agent": steps,
        "intermediate_steps_global": state.intermediate_steps_global + steps,
        "next_action": NextAction.FINALIZE,
    }


async def analysis_node(state: FinanceAgentState, config: RunnableConfig) -> dict:
    """Analisa os dados coletados com frameworks financeiros. Sem tools, sem
    Markdown. Produz raciocínio técnico estruturado para o nó de síntese.
    """
    llm = _make_llm(model="gpt-4o", temperature=0.2)

    context_block = json.dumps(
        {"financial_profile": state.finance_profile, "accounts": state.finance_accounts},
        ensure_ascii=False, default=str,
    )

    extra = (
        f"[PERFIL FINANCEIRO E CONTAS DO USUÁRIO]\n{context_block}\n\n"
        f"[DADOS COLETADOS PELAS FERRAMENTAS]\n{state.gathered_data}"
    )

    # Apply personalization (currency + obfuscation) to the analysis prompt
    # too, so the technical reasoning uses the correct currency symbol and
    # respects the user's privacy level — the synthesis node may quote
    # numbers from this analysis verbatim.
    system_prompt = _FINANCE_ANALYSIS_SYSTEM
    pers = state.personalization
    profile_currency = state.finance_profile.get("currency") if isinstance(state.finance_profile, dict) else None
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

    content, cot, status, _tool_msgs, steps = await _run_agent_loop(
        state,
        llm,
        system_prompt=system_prompt,
        extra_context=extra,
        clear_steps=True,
        node_name="analysis",
        enable_cot=True,
        tool_map_override=None,  # no tools for analysis
        config=config,
    )

    # The analysis is the <thought> content (the <answer> should be empty)
    analysis_text = cot if cot else content

    print(f"[analysis_node] cot_len={len(cot)}, analysis_len={len(analysis_text)}")

    return {
        **state.model_dump(),
        "analysis_text": analysis_text,
        "cot": cot,
        "reasoning_trail": _append_reasoning(state.reasoning_trail, "analysis", cot, status),
        "next_action": NextAction.FINALIZE,
    }


async def synthesis_node(state: FinanceAgentState, config: RunnableConfig) -> dict:
    """Transforma a análise técnica em resposta final em Markdown para o
    usuário. Token streaming acontece aqui — o usuário vê a resposta sendo
    digitada em tempo real.
    """
    llm = _make_llm(model="gpt-4o-mini", temperature=0.3)

    extra = f"[ANÁLISE DO AGENTE ESPECIALISTA]\n{state.analysis_text}"

    # Build a personalization directive from the user's Soter preferences.
    # This adjusts tone, display name, language, currency and obfuscation
    # of the final response — applied consistently across all nodes.
    system_prompt = _FINANCE_SYNTHESIS_SYSTEM
    pers = state.personalization
    profile_currency = None
    if isinstance(state.finance_profile, dict):
        profile_currency = state.finance_profile.get("currency")
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

    content, cot, status, _tool_msgs, steps = await _run_agent_loop(
        state,
        llm,
        system_prompt=system_prompt,
        extra_context=extra,
        clear_steps=True,
        node_name="synthesis",
        enable_cot=False,
        tool_map_override=None,
        config=config,
    )

    print(f"[synthesis_node] answer_len={len(content)}")

    # Safety: never return an empty answer
    if not content or not content.strip():
        content = state.analysis_text or "Não consegui gerar uma resposta. Tente reformular sua pergunta."

    return {
        **state.model_dump(),
        "final_answer": content,
        "cot": "",
        "reasoning_trail": _append_reasoning(state.reasoning_trail, "synthesis", "", NodeStatus.OK),
        "next_action": NextAction.FINALIZE,
        "messages": state.messages + [AIMessage(content=content)],
    }


def finalize_node(state: FinanceAgentState) -> dict:
    """Extrai a resposta final — nó terminal."""
    return {**state.model_dump(), "next_action": "done"}


# ─────────────────────────────────────────────
# 7. GRAPH ASSEMBLY
# ─────────────────────────────────────────────

def build_finance_graph() -> Any:
    workflow = StateGraph(FinanceAgentState)

    workflow.add_node("finance_orchestrator", finance_orchestrator_node)
    workflow.add_node("finance_context",      finance_context_node)
    workflow.add_node("action",              action_node)
    workflow.add_node("data_gathering",       data_gathering_node)
    workflow.add_node("analysis",             analysis_node)
    workflow.add_node("synthesis",            synthesis_node)
    workflow.add_node("finalize",             finalize_node)

    workflow.set_entry_point("finance_orchestrator")

    # orchestrator → finalize (greeting/out-of-scope)
    # orchestrator → finance_context (FINANCE_OK or ACTION — both need context)
    workflow.add_conditional_edges(
        "finance_orchestrator",
        lambda s: "finalize" if s.next_action == NextAction.FINALIZE else "finance_context",
        {"finalize": "finalize", "finance_context": "finance_context"},
    )

    # finance_context → finalize (error)
    # finance_context → action (if next_action == ACTION)
    # finance_context → data_gathering (if next_action == FINANCE_CONTEXT)
    workflow.add_conditional_edges(
        "finance_context",
        lambda s: (
            "finalize" if s.next_action == NextAction.FINALIZE
            else "action" if s.next_action == NextAction.ACTION
            else "data_gathering"
        ),
        {"finalize": "finalize", "action": "action", "data_gathering": "data_gathering"},
    )

    workflow.add_edge("action", "finalize")
    workflow.add_edge("data_gathering", "analysis")
    workflow.add_edge("analysis", "synthesis")
    workflow.add_edge("synthesis", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


_finance_graph: Any | None = None


def get_finance_agent_graph() -> Any:
    global _finance_graph
    if _finance_graph is None:
        _finance_graph = build_finance_graph()
    return _finance_graph


# Backward-compat alias — chat.py imports this name.
get_agent_graph = get_finance_agent_graph
