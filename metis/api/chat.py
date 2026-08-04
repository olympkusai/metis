from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from metis.agent.graph import get_finance_agent_graph, FinanceAgentState
from metis.memory.conversation_history import (
    get_conversation_history,
    MessageRole,
)
import json
import asyncio
import uuid

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None


def _extract_bearer_token(authorization: Optional[str]) -> str:
    """Extracts the raw JWT from an `Authorization: Bearer <token>` header."""
    if not authorization:
        return ""
    return authorization.removeprefix("Bearer ").strip()


@router.post("/chat")
async def chat(request: ChatRequest, authorization: Optional[str] = Header(None)):
    auth_token = _extract_bearer_token(authorization)
    session_id = request.session_id or f"{request.user_id}:{uuid.uuid4().hex}"

    conv_history = get_conversation_history()
    await conv_history.save_message(
        user_id=request.user_id,
        session_id=session_id,
        role=MessageRole.USER,
        content=request.message,
    )

    session_history = await conv_history.get_conversation_history(
        user_id=request.user_id,
        session_id=session_id,
        limit=20,
    )
    global_context = await conv_history.get_global_context(
        user_id=request.user_id,
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

    agent = get_finance_agent_graph()
    initial_state = FinanceAgentState(
        messages=messages,
        user_id=request.user_id,
        auth_token=auth_token,
    )

    final_state = await agent.ainvoke(initial_state, config={"recursion_limit": 60})
    final_answer = final_state.get("final_answer", "")

    reasoning_steps = [step[0] for step in final_state.get("intermediate_steps_agent", [])]
    cot = final_state.get("cot", "")

    await conv_history.save_message(
        user_id=request.user_id,
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
async def streaming_chat(request: ChatRequest, authorization: Optional[str] = Header(None)):
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
    auth_token = _extract_bearer_token(authorization)

    session_id = request.session_id or f"{request.user_id}:{uuid.uuid4().hex}"

    conv_history = get_conversation_history()
    await conv_history.save_message(
        user_id=request.user_id,
        session_id=session_id,
        role=MessageRole.USER,
        content=request.message,
    )

    async def event_generator():
        session_history = await conv_history.get_conversation_history(
            user_id=request.user_id,
            session_id=session_id,
            limit=20,
        )
        global_context = await conv_history.get_global_context(
            user_id=request.user_id,
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

        agent = get_finance_agent_graph()
        initial_state = FinanceAgentState(
            messages=messages,
            user_id=request.user_id,
            auth_token=auth_token,
        )

        accumulated_state: dict = {}
        current_node: str = ""

        # stream_mode=["messages", "updates"] gives us both token-level
        # streaming AND node completion events in a single iteration.
        async for mode, payload in agent.astream(
            initial_state,
            config={"recursion_limit": 60},
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

                token_event = {
                    "type": "token",
                    "node": node,
                    "content": content,
                }
                yield f"data: {json.dumps(token_event)}\n\n"

            elif mode == "updates":
                for node_name, state_update in payload.items():
                    accumulated_state.update(state_update)

                    # Skip terminal/pass-through nodes that don't produce new
                    # reasoning — they just re-emit inherited state (cot, etc).
                    if node_name in ("finalize", "__end__"):
                        continue

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

        final_event = {
            "node": "final",
            "type": "completion",
            "response": accumulated_state.get("final_answer", ""),
            "timestamp": asyncio.get_event_loop().time(),
            "session_id": session_id,
        }

        await conv_history.save_message(
            user_id=request.user_id,
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=accumulated_state.get("final_answer", ""),
            metadata={
                "chain_of_thought": accumulated_state.get("cot", ""),
            },
        )

        yield f"data: {json.dumps(final_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
