from fastapi import APIRouter, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from metis.agent.graph import get_finance_agent_graph, FinanceAgentState, NextAction
from metis.agent.graph_v2 import get_finance_agent_graph_v2
from metis.config import get_settings
from metis.memory.conversation_history import (
    get_conversation_history,
    MessageRole,
)
from metis.jwt_verifier import JWTVerifier, InvalidTokenError
from metis.api.deps import get_jwt_verifier
import json
import asyncio
import uuid

router = APIRouter()


def _get_agent_graph():
    """Select agent graph based on AGENT_VERSION feature flag."""
    settings = get_settings()
    if settings.agent_version == "v2":
        return get_finance_agent_graph_v2()
    return get_finance_agent_graph()


# ─────────────────────────────────────────────
# Helpers for fake-streaming static messages
# ─────────────────────────────────────────────

# When the orchestrator short-circuits with a static greeting or out-of-scope
# message, LangGraph emits the full AIMessage as a single "messages" chunk.
# This splits large chunks into word groups with small delays so the frontend
# gets a smooth streaming UX even for non-LLM responses.
_STATIC_CHUNK_THRESHOLD = 80  # chars — below this, emit as-is
_STATIC_CHUNK_DELAY = 0.02    # seconds between fake chunks


async def _yield_token_events(content: str, node: str):
    """Yield SSE token events, splitting large chunks for a streaming feel."""
    if len(content) <= _STATIC_CHUNK_THRESHOLD:
        yield f"data: {json.dumps({'type': 'token', 'node': node, 'content': content})}\n\n"
        return

    # Split into word groups of ~3-5 words for natural pacing
    words = content.split(" ")
    i = 0
    while i < len(words):
        group_size = 3 + (i % 3)  # 3, 4, 5, 3, 4, 5, ...
        chunk_words = words[i:i + group_size]
        chunk = " ".join(chunk_words)
        # Preserve leading space if we're not at the start
        if i > 0 and not chunk.startswith(" ") and not content[i - 1] == " ":
            pass  # words.join handles spacing
        yield f"data: {json.dumps({'type': 'token', 'node': node, 'content': chunk})}\n\n"
        i += group_size
        await asyncio.sleep(_STATIC_CHUNK_DELAY)


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None  # Deprecated — user_id is extracted from the JWT
    session_id: Optional[str] = None


def _extract_bearer_token(authorization: Optional[str]) -> str:
    """Extracts the raw JWT from an `Authorization: Bearer <token>` header."""
    if not authorization:
        return ""
    return authorization.removeprefix("Bearer ").strip()


async def _validate_and_get_user_id(
    authorization: Optional[str],
    verifier: JWTVerifier,
) -> tuple[str, str]:
    """Validate the JWT and return (user_id, raw_token).

    The raw token is returned so it can be forwarded to downstream services
    (e.g. Pluto) that need it. The user_id is extracted from the `sub` claim.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    try:
        identity = await verifier.verify(token)
        return identity.user_id, token
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.post("/chat")
async def chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
    verifier: JWTVerifier = Depends(get_jwt_verifier),
):
    user_id, auth_token = await _validate_and_get_user_id(authorization, verifier)
    session_id = request.session_id or f"{user_id}:{uuid.uuid4().hex}"

    conv_history = get_conversation_history()
    await conv_history.save_message(
        user_id=user_id,
        session_id=session_id,
        role=MessageRole.USER,
        content=request.message,
    )

    session_history = await conv_history.get_conversation_history(
        user_id=user_id,
        session_id=session_id,
        limit=20,
    )
    global_context = await conv_history.get_global_context(
        user_id=user_id,
        limit=10,
        exclude_session_id=session_id,
    )

    messages = []

    if global_context:
        context_summary = "Previous conversations context:\n"
        for msg in global_context[-5:]:
            context_summary += f"{msg['role']}: {msg['content'][:200]}...\n"
        messages.append(HumanMessage(content=context_summary))

    for msg in session_history[:-1]:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            messages.append(AIMessage(content=msg['content']))

    messages.append(HumanMessage(content=request.message))

    agent = _get_agent_graph()
    initial_state = FinanceAgentState(
        messages=messages,
        user_id=user_id,
        auth_token=auth_token,
    )

    final_state = await agent.ainvoke(initial_state, config={"recursion_limit": 60})
    final_answer = final_state.get("final_answer", "")

    reasoning_steps = [step[0] for step in final_state.get("intermediate_steps_agent", [])]
    cot = final_state.get("cot", "")

    await conv_history.save_message(
        user_id=user_id,
        session_id=session_id,
        role=MessageRole.ASSISTANT,
        content=final_answer,
        metadata={
            "reasoning_steps": reasoning_steps,
            "chain_of_thought": cot,
        },
    )

    return {
        "response": final_answer,
        "reasoning": reasoning_steps,
        "thought": cot,
        "session_id": session_id,
    }


@router.post("/streaming/chat")
async def streaming_chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
    verifier: JWTVerifier = Depends(get_jwt_verifier),
):
    """Streaming endpoint that emits SSE events in real time.

    Uses two LangGraph stream modes simultaneously:
    - "messages": token-by-token LLM output (real-time reasoning as it's generated)
    - "updates": node completion events (CoT, final answer, state changes)

    SSE event types emitted:
    - { type: "node_start", node } — a node started executing
    - { type: "token", node, content } — a token chunk from the LLM (live reasoning)
    - { type: "node_execution", node, thought } — a node completed with its CoT
    - { type: "completion", response, session_id } — final answer
    """
    user_id, auth_token = await _validate_and_get_user_id(authorization, verifier)
    session_id = request.session_id or f"{user_id}:{uuid.uuid4().hex}"

    conv_history = get_conversation_history()
    await conv_history.save_message(
        user_id=user_id,
        session_id=session_id,
        role=MessageRole.USER,
        content=request.message,
    )

    async def event_generator():
        session_history = await conv_history.get_conversation_history(
            user_id=user_id,
            session_id=session_id,
            limit=20,
        )
        global_context = await conv_history.get_global_context(
            user_id=user_id,
            limit=10,
            exclude_session_id=session_id,
        )

        messages = []

        if global_context:
            context_summary = "Previous conversations context:\n"
            for msg in global_context[-5:]:
                context_summary += f"{msg['role']}: {msg['content'][:200]}...\n"
            messages.append(HumanMessage(content=context_summary))

        for msg in session_history[:-1]:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))

        messages.append(HumanMessage(content=request.message))

        agent = _get_agent_graph()
        initial_state = FinanceAgentState(
            messages=messages,
            user_id=user_id,
            auth_token=auth_token,
        )

        accumulated_state: dict = {}
        current_node: str = ""

        # v2 real token streaming: the AgentRuntime calls stream_callback for
        # each text chunk of the final answer (bypassing LangGraph's
        # stream_mode which mixes intermediate ReAct reasoning). The callback
        # pushes tokens to a queue; we drain it after each agent.astream event.
        token_queue: asyncio.Queue = asyncio.Queue()
        v2_tokens_streamed = False

        async def stream_callback(content: str) -> None:
            token_queue.put_nowait(content)

        # stream_mode=["messages", "updates"] gives us both token-level
        # streaming AND node completion events in a single iteration.
        async for mode, payload in agent.astream(
            initial_state,
            config={
                "recursion_limit": 60,
                "metadata": {"stream_callback": stream_callback},
            },
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                # payload is (chunk, metadata) — chunk is an AIMessageChunk
                # with token content, metadata tells us which node it came from.
                chunk, metadata = payload
                node = metadata.get("langgraph_node", current_node) if isinstance(metadata, dict) else current_node
                current_node = node

                content = getattr(chunk, "content", "")
                if not content:
                    continue

                # Skip tool call chunks (they have no text content)
                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    continue

                # Skip tokens from nodes that don't produce user-facing text:
                # - finance_orchestrator: internal classification only
                # - finance_context: no LLM, static messages fake-streamed below
                # - data_gathering: tool calls + CoT (shown via node_execution)
                # - analysis: CoT only (shown via node_execution), not user text
                # - finance_agent_v2: tokens are streamed via stream_callback
                #   (real-time, only final answer) not via LangGraph's messages
                #   stream (which includes intermediate ReAct reasoning).
                if node in ("finance_orchestrator", "finance_context", "data_gathering", "analysis", "finance_agent_v2"):
                    continue

                # 'synthesis' streams the user-facing answer directly —
                # real token streaming, no fake-stream needed.
                async for sse_line in _yield_token_events(content, node):
                    yield sse_line

            elif mode == "updates":
                for node_name, state_update in payload.items():
                    accumulated_state.update(state_update)

                    # Skip terminal/pass-through nodes that don't produce new
                    # reasoning — they just re-emit inherited state (cot, etc).
                    if node_name in ("finalize", "__end__"):
                        continue

                    # When the orchestrator short-circuits (greeting /
                    # out-of-scope), it sets final_answer directly — no LLM
                    # tokens were streamed, so fake-stream the static message
                    # now for a smooth UX.
                    if (
                        node_name == "finance_orchestrator"
                        and state_update.get("final_answer")
                        and state_update.get("next_action") == NextAction.FINALIZE.value
                    ):
                        async for sse_line in _yield_token_events(
                            state_update["final_answer"], node_name
                        ):
                            yield sse_line

                    # Same for finance_context when it short-circuits on
                    # auth/Pluto errors — the error message is static text.
                    if (
                        node_name == "finance_context"
                        and state_update.get("final_answer")
                        and state_update.get("next_action") == NextAction.FINALIZE.value
                    ):
                        async for sse_line in _yield_token_events(
                            state_update["final_answer"], node_name
                        ):
                            yield sse_line

                    # v2: finance_agent_v2 — if stream_callback already
                    # streamed the final answer tokens in real time, skip
                    # fake-stream. Fall back to fake-stream only if no tokens
                    # were streamed (e.g. error before streaming started).
                    if (
                        node_name == "finance_agent_v2"
                        and state_update.get("final_answer")
                        and not v2_tokens_streamed
                    ):
                        async for sse_line in _yield_token_events(
                            state_update["final_answer"], node_name
                        ):
                            yield sse_line

                    # Emit node_execution with CoT if available
                    thought = ""
                    if "cot" in state_update and state_update["cot"]:
                        thought = state_update["cot"]

                    # Emit node_execution event (backward compat with afrodite)
                    step_data = {
                        "node": node_name,
                        "type": "node_execution",
                        "timestamp": asyncio.get_event_loop().time(),
                        "reasoning": "",
                        "thought": thought,
                    }
                    yield f"data: {json.dumps(step_data)}\n\n"

            # Drain the token queue after every event (messages or updates).
            # The AgentRuntime's stream_callback pushes final-answer tokens
            # here in real time; we emit them as SSE token events.
            while not token_queue.empty():
                content = token_queue.get_nowait()
                v2_tokens_streamed = True
                async for sse_line in _yield_token_events(content, "finance_agent_v2"):
                    yield sse_line

        # Final drain — catch any tokens pushed after the last astream event.
        while not token_queue.empty():
            content = token_queue.get_nowait()
            v2_tokens_streamed = True
            async for sse_line in _yield_token_events(content, "finance_agent_v2"):
                yield sse_line

        final_event = {
            "node": "final",
            "type": "completion",
            "response": accumulated_state.get("final_answer", ""),
            "timestamp": asyncio.get_event_loop().time(),
            "session_id": session_id,
        }

        await conv_history.save_message(
            user_id=user_id,
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=accumulated_state.get("final_answer", ""),
            metadata={
                "chain_of_thought": accumulated_state.get("cot", ""),
            },
        )

        yield f"data: {json.dumps(final_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx/Railway edge)
        },
    )
