"""
Metis — AgentRuntime
─────────────────────
Generic ReAct loop extracted from the v1 fixed pipeline.

The runtime is agnostic to the domain: it receives a system prompt and a
unified catalog of LangChain tools, then lets the LLM decide freely which
tools to call, in which order, and when to stop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from metis.config import get_settings
from metis.utils.cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)

_BASE_MODEL = "gpt-4o"

# Human-readable names for each finance tool, so the CoT shows
# "Consultando gastos por categoria" instead of "Consultando get_spending_by_category".
_DEFAULT_TOOL_LABELS: dict[str, str] = {
    "get_spending_by_category": "gastos por categoria",
    "get_cashflow": "fluxo de caixa",
    "get_budget_progress": "progresso do orçamento",
    "get_goal_summary": "resumo de metas",
    "get_recurrences_due": "contas recorrentes a vencer",
    "list_transactions_filtered": "transações filtradas",
}


# ─────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────

def _latest_user_message(messages: list) -> HumanMessage | None:
    """Retorna a ÚLTIMA `HumanMessage` da lista, ou None se não houver.

    Handles both HumanMessage objects and dict representations (which occur
    after state serialization via model_dump()).
    """
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m
        if isinstance(m, dict):
            # Check if it's a serialized HumanMessage (has 'content' and
            # type indicator like 'role' or 'type' key)
            if m.get("type") == "human" or m.get("role") == "user":
                return HumanMessage(content=m.get("content", ""))
            # Fallback: if it has content and looks like a user message
            # (no 'tool_calls' or 'type' indicating AI/tool)
            if "content" in m and not m.get("tool_calls") and m.get("type") != "ai":
                return HumanMessage(content=m["content"])
    return None


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
# 2. TOOL EXECUTION
# ─────────────────────────────────────────────

async def _execute_tools(
    last_ai: AIMessage,
    steps: list,
    cache: dict[str, str] | None = None,
    tool_map: dict | None = None,
) -> tuple[list[ToolMessage], list]:
    """Executa tool calls em paralelo, com dedup cache (por-run).

    `tool_map` permite a um agente resolver tool calls contra um
    conjunto de tools específico.
    """
    tool_messages: list[ToolMessage] = []
    steps = list(steps)
    cache = cache if cache is not None else {}
    active_tool_map = tool_map if tool_map is not None else {}

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


# ─────────────────────────────────────────────
# 3. RESULT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class AgentResult:
    """Result of a single AgentRuntime.run() execution."""
    answer: str
    cot: str
    status: NodeStatus
    tool_messages: list[ToolMessage] = field(default_factory=list)
    steps: list[tuple[str, str]] = field(default_factory=list)


# ─────────────────────────────────────────────
# 4. AGENT RUNTIME
# ─────────────────────────────────────────────

class AgentRuntime:
    """Generic ReAct loop: LLM decides which tools to call, runtime executes them.

    The LLM has access to ALL tools in the catalog and decides freely:
    - which tools to call
    - when to stop (emit response without tool_calls)
    - the order of operations
    """

    def __init__(
        self,
        system_prompt: str,
        tools: list,  # list of LangChain BaseTool — unified catalog
        *,
        model: str = _BASE_MODEL,
        temperature: float = 0.1,
        max_iterations: int = 20,
        enable_cot: bool = False,
        node_name: str = "agent",
        tool_labels: dict[str, str] | None = None,
        stream_callback: Callable[[str], Awaitable[None]] | None = None,
    ):
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.enable_cot = enable_cot
        self.node_name = node_name
        self.tool_labels = tool_labels if tool_labels is not None else dict(_DEFAULT_TOOL_LABELS)
        self.stream_callback = stream_callback

        settings = get_settings()
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=settings.openai_api_key,
            streaming=True,
        )
        if tools:
            self.llm = self.llm.bind_tools(tools)

        # Per-run state (reset on each run() call)
        self.tool_cache: dict[str, str] = {}
        self.steps: list[tuple[str, str]] = []
        self.reasoning_lines: list[str] = []

    async def _astream_and_collect(
        self,
        msgs: list,
        *,
        config: RunnableConfig | None = None,
    ) -> tuple[AIMessage, str | None]:
        """Stream the LLM response, collecting chunks into a full AIMessage.

        Only pure-text chunks (no tool_call_chunks) are forwarded to
        stream_callback in real time. When the LLM is calling tools, the
        chunks contain tool_call_chunks and are NOT forwarded — intermediate
        reasoning stays internal. When the LLM produces the final answer
        (no tool calls), all chunks are pure text and stream live.

        In practice, with bind_tools the LLM either produces tool_calls OR
        text in a given turn, rarely both. So this effectively streams only
        the final answer.

        Returns (assembled_aimessage, streamed_text_or_None).
        """
        chunks: list = []
        streamed_parts: list[str] = []

        async for chunk in self.llm.astream(msgs, config=config):
            chunks.append(chunk)
            content = getattr(chunk, "content", "")
            # Only forward non-empty text chunks without tool_call_chunks
            if content and self.stream_callback is not None:
                has_tool_call = bool(
                    getattr(chunk, "tool_call_chunks", None)
                )
                if not has_tool_call:
                    streamed_parts.append(content)
                    await self.stream_callback(content)

        # Reassemble the full AIMessage from chunks
        if chunks:
            response = chunks[0]
            for chunk in chunks[1:]:
                response = response + chunk
        else:
            response = AIMessage(content="")

        streamed_text = "".join(streamed_parts) if streamed_parts else None
        return response, streamed_text

    async def run(
        self,
        messages: list,  # conversation history (HumanMessage, AIMessage, etc.)
        *,
        extra_context: str = "",
        reasoning_trail: list[tuple[str, str, str]] | None = None,
        config: RunnableConfig | None = None,
    ) -> AgentResult:
        """Execute the ReAct loop. Returns AgentResult with answer, cot, status, tool_msgs, steps."""
        # Reset per-run state
        self.tool_cache = {}
        self.steps = []
        self.reasoning_lines = []

        original_query = _latest_user_message(messages)

        tool_instruction = (
            "\n\nIMPORTANTE: Você tem acesso a ferramentas. Você DEVE chamar as "
            "ferramentas apropriadas antes de fornecer sua análise final. Não "
            "responda sem chamar as ferramentas primeiro."
        )

        cot_instruction = ""
        if self.enable_cot:
            cot_instruction = (
                "\n\nFORMATO DE RESPOSTA OBRIGATÓRIO (único):\n"
                "<thought>Resumo CONCISO do seu raciocínio (max 2-3 frases). "
                "Ex: 'Obtive o indicador X que está em Y, indicando Z.'</thought>\n"
                "<answer>Sua resposta final em Markdown aqui (siga as "
                "diretrizes de estilo acima)</answer>"
            )

        enhanced_prompt = self.system_prompt + tool_instruction + cot_instruction

        msgs: list = [SystemMessage(content=enhanced_prompt)]

        if reasoning_trail:
            trail_block = (
                "RACIOCÍNIO DOS AGENTES ANTERIORES (use como contexto, não copie literalmente):\n"
                + "\n".join(
                    f"  [{name}]({status}) {thought}"
                    for name, thought, status in reasoning_trail
                )
            )
            msgs.append(HumanMessage(content=trail_block))

        if extra_context:
            msgs.append(HumanMessage(content=f"[CONTEXTO DO PIPELINE]\n{extra_context}"))

        # User's message goes LAST so the LLM treats it as the current request
        if original_query:
            msgs.append(original_query)
        else:
            logger.warning(
                f"[{self.node_name}] No user message found in messages (len={len(messages)})"
            )

        all_tool_msgs: list[ToolMessage] = []
        steps: list[tuple[str, str]] = []
        MAX_RETRIES = 3

        previous_step_count = len(steps)
        no_progress_count = 0

        # Track the sequence of tool calls (name + args) for repetition/cycle detection.
        tool_call_history: list[tuple[str, str]] = []

        # Accumulate each tool call as a reasoning step so the CoT is a real
        # chain (step 1 → step 2 → … → final thought) instead of just the
        # final summary.
        reasoning_lines: list[str] = []

        def _build_cot(final_cot: str) -> str:
            """Combine accumulated reasoning lines with the final thought."""
            if final_cot and reasoning_lines:
                return "\n".join(reasoning_lines + [final_cot])
            if final_cot:
                return final_cot
            if reasoning_lines:
                return "\n".join(reasoning_lines)
            return ""

        def _detect_repetition() -> bool:
            """Detect repeated tool calls or cycles.

            - Same tool + args 3 times in a row → abort.
            - Cycle of 2-3 tools repeating 3 times → abort.
            """
            if not tool_call_history:
                return False
            n = len(tool_call_history)

            # Same tool+args 3 times in a row
            if n >= 3:
                last_three = tool_call_history[-3:]
                if last_three[0] == last_three[1] == last_three[2]:
                    return True

            # Cycle of length 2 repeating 3 times (6 calls)
            if n >= 6:
                cycle_len = 2
                last_block = tool_call_history[-(cycle_len * 3):]
                if (
                    last_block[0:cycle_len]
                    == last_block[cycle_len:cycle_len * 2]
                    == last_block[cycle_len * 2:cycle_len * 3]
                ):
                    return True

            # Cycle of length 3 repeating 3 times (9 calls)
            if n >= 9:
                cycle_len = 3
                last_block = tool_call_history[-(cycle_len * 3):]
                if (
                    last_block[0:cycle_len]
                    == last_block[cycle_len:cycle_len * 2]
                    == last_block[cycle_len * 2:cycle_len * 3]
                ):
                    return True

            return False

        for iteration in range(self.max_iterations):
            response = None
            for attempt in range(MAX_RETRIES):
                try:
                    if self.stream_callback is not None:
                        # Use astream so we can forward tokens to the callback
                        # in real time. We collect chunks to reassemble the
                        # full AIMessage (needed for tool_calls detection and
                        # for appending to msgs).
                        response, streamed_text = await self._astream_and_collect(
                            msgs, config=config,
                        )
                    else:
                        response = await self.llm.ainvoke(msgs, config=config, timeout=30)
                        streamed_text = None

                    cost_tracker = get_cost_tracker()
                    if hasattr(response, 'response_metadata'):
                        metadata = response.response_metadata
                        input_tokens = metadata.get('token_usage', {}).get('prompt_tokens', 0)
                        output_tokens = metadata.get('token_usage', {}).get('completion_tokens', 0)
                        model_name = getattr(self.llm, 'model_name', 'unknown') or getattr(self.llm, 'model', 'unknown')
                        if input_tokens > 0 or output_tokens > 0:
                            cost_tracker.add_call(model_name, input_tokens, output_tokens, self.node_name)

                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        error_msg = f"[LLM ERROR after {MAX_RETRIES} retries]: {str(e)}"
                        return AgentResult(
                            answer=error_msg,
                            cot=_failure_cot(self.node_name, error_msg),
                            status=NodeStatus.ERROR,
                            tool_messages=all_tool_msgs,
                            steps=steps,
                        )
                    backoff_time = 0.5 * (2 ** attempt)
                    await asyncio.sleep(backoff_time)
                    continue

            if response is None:
                error_msg = "[LLM ERROR] Failed to get response after retries"
                return AgentResult(
                    answer=error_msg,
                    cot=_failure_cot(self.node_name, error_msg),
                    status=NodeStatus.ERROR,
                    tool_messages=all_tool_msgs,
                    steps=steps,
                )

            msgs.append(response)

            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Record each tool call as a reasoning step
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    friendly = self.tool_labels.get(tool_name, tool_name.replace("_", " "))
                    args = dict(tc.get("args") or {})
                    # Build a short human-readable description of the call
                    if args:
                        arg_str = ", ".join(
                            f"{k}={v}" for k, v in args.items()
                            if k not in ("auth_token",) and v
                        )
                        reasoning_lines.append(
                            f"Consultando {friendly}({arg_str})" if arg_str else f"Consultando {friendly}"
                        )
                    else:
                        reasoning_lines.append(f"Consultando {friendly}")
                    # Track for repetition/cycle detection
                    try:
                        args_key = json.dumps(args, sort_keys=True, default=str)
                    except Exception:
                        args_key = repr(sorted(args.items()))
                    tool_call_history.append((tool_name, args_key))

                # Repetition / cycle detection
                if _detect_repetition():
                    error_msg = (
                        "[AGENT LOOP] Repetitive tool calls detected - "
                        "aborting to prevent infinite loop"
                    )
                    return AgentResult(
                        answer=error_msg,
                        cot=_failure_cot(self.node_name, error_msg),
                        status=NodeStatus.ERROR,
                        tool_messages=all_tool_msgs,
                        steps=steps,
                    )

                tool_msgs, steps = await _execute_tools(
                    response, steps, cache=self.tool_cache,
                    tool_map=self.tool_map,
                )
                msgs.extend(tool_msgs)
                all_tool_msgs.extend(tool_msgs)
            elif hasattr(response, 'content'):
                # AIMessage / AIMessageChunk — use .content (the actual text),
                # NOT model_dump() which serializes the entire message object
                # (metadata, usage, id, …) as JSON.
                cot, answer = (
                    _extract_cot_and_answer(response.content) if self.enable_cot
                    else ("", response.content)
                )
                if self.enable_cot and not cot and steps:
                    cot = _synthesize_cot_from_steps(steps)
                final_cot = _build_cot(cot) if self.enable_cot else ""
                status = NodeStatus.OK if (final_cot or not self.enable_cot) else NodeStatus.NO_COT
                return AgentResult(
                    answer=answer,
                    cot=final_cot,
                    status=status,
                    tool_messages=all_tool_msgs,
                    steps=steps,
                )
            elif isinstance(response, (str, dict)):
                if isinstance(response, dict):
                    content = json.dumps(response, ensure_ascii=False, indent=2)
                else:
                    content = str(response)
                cot, answer = (
                    _extract_cot_and_answer(content) if self.enable_cot
                    else ("", content)
                )
                if self.enable_cot and not cot and steps:
                    cot = _synthesize_cot_from_steps(steps)
                final_cot = _build_cot(cot) if self.enable_cot else ""
                status = NodeStatus.OK if (final_cot or not self.enable_cot) else NodeStatus.NO_COT
                return AgentResult(
                    answer=answer,
                    cot=final_cot,
                    status=status,
                    tool_messages=all_tool_msgs,
                    steps=steps,
                )
            else:
                # Last-resort fallback for unexpected response types
                content = str(response)
                cot, answer = (
                    _extract_cot_and_answer(content) if self.enable_cot
                    else ("", content)
                )
                if self.enable_cot and not cot and steps:
                    cot = _synthesize_cot_from_steps(steps)
                final_cot = _build_cot(cot) if self.enable_cot else ""
                status = NodeStatus.OK if (final_cot or not self.enable_cot) else NodeStatus.NO_COT
                return AgentResult(
                    answer=answer,
                    cot=final_cot,
                    status=status,
                    tool_messages=all_tool_msgs,
                    steps=steps,
                )

            if len(steps) == previous_step_count:
                no_progress_count += 1
                if no_progress_count >= 2:
                    error_msg = "[AGENT LOOP] No progress detected - aborting to prevent infinite loop"
                    return AgentResult(
                        answer=error_msg,
                        cot=_failure_cot(self.node_name, error_msg),
                        status=NodeStatus.ERROR,
                        tool_messages=all_tool_msgs,
                        steps=steps,
                    )
            else:
                no_progress_count = 0
                previous_step_count = len(steps)

        # Força finalização se exceder iterações
        try:
            if self.stream_callback is not None:
                final, _ = await self._astream_and_collect(
                    msgs + [HumanMessage(content="Sintetize os resultados obtidos até agora.")],
                    config=config,
                )
            else:
                final = await self.llm.ainvoke(
                    msgs + [HumanMessage(content="Sintetize os resultados obtidos até agora.")],
                    config=config,
                )
        except Exception as e:
            error_msg = f"[LLM ERROR on forced finalize]: {str(e)}"
            return AgentResult(
                answer=error_msg,
                cot=_failure_cot(self.node_name, error_msg),
                status=NodeStatus.ERROR,
                tool_messages=all_tool_msgs,
                steps=steps,
            )
        cot, answer = (
            _extract_cot_and_answer(final.content) if self.enable_cot
            else ("", final.content)
        )
        if self.enable_cot and not cot and steps:
            cot = _synthesize_cot_from_steps(steps)
        final_cot = _build_cot(cot) if self.enable_cot else ""
        status = NodeStatus.OK if (final_cot or not self.enable_cot) else NodeStatus.NO_COT
        return AgentResult(
            answer=answer,
            cot=final_cot,
            status=status,
            tool_messages=all_tool_msgs,
            steps=steps,
        )


__all__ = [
    "AgentRuntime",
    "AgentResult",
    "NodeStatus",
    "_latest_user_message",
    "_extract_cot_and_answer",
    "_failure_cot",
    "_synthesize_cot_from_steps",
    "_append_reasoning",
    "_cache_key",
    "_execute_tools",
]
