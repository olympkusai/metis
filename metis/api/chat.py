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
    """Streaming endpoint that emits each node execution as an SSE event."""
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

        accumulated_state = {}

        async for event in agent.astream(
            initial_state,
            config={"recursion_limit": 60},
            stream_mode="updates"
        ):
            for node_name, state_update in event.items():
                accumulated_state.update(state_update)

                step_data = {
                    "node": node_name,
                    "type": "node_execution",
                    "timestamp": asyncio.get_event_loop().time(),
                    "reasoning": "",
                    "thought": ""
                }

                # Extract CoT (chain-of-thought) if available
                if "cot" in state_update and state_update["cot"]:
                    step_data["thought"] = state_update["cot"]

                # Extract reasoning from messages if available
                if "messages" in state_update:
                    msgs = state_update["messages"]
                    if msgs:
                        last_message = msgs[-1]
                        if hasattr(last_message, 'content') and last_message.content:
                            content = last_message.content
                            if isinstance(content, dict) and "content" in content:
                                content = content["content"]
                            elif isinstance(content, str):
                                try:
                                    parsed = json.loads(content)
                                    if isinstance(parsed, dict) and "content" in parsed:
                                        content = parsed["content"]
                                except:
                                    pass

                            if isinstance(content, str) and len(content) > 500:
                                content = content[:500] + "..."
                            step_data["reasoning"] = content

                # Emit SSE event
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
